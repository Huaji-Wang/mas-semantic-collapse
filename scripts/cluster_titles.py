"""Stage 3: embed headlines, cluster them, compare to nine motion labels.

    python scripts/cluster_titles.py --tag chik200
    python scripts/cluster_titles.py --tag chik200 --title-backend openai
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mas_collapse.config import load_config
from mas_collapse.embed.backend import build_embedder

MOTIONS = [
    "translation",
    "diffusion",
    "condensation",
    "fission",
    "crystallization",
    "vortex",
    "orbit",
    "breathing",
    "cascade",
]
MOTION_ZH = {
    "translation": "平移",
    "diffusion": "扩散",
    "condensation": "凝聚",
    "fission": "裂变",
    "crystallization": "结晶",
    "vortex": "涡流",
    "orbit": "轨道",
    "breathing": "呼吸",
    "cascade": "级联",
}
UNUSABLE_RE = re.compile(
    r"^\s*$|^\[\s*(removed by moderator|removed|deleted)\s*\]\s*$",
    re.I,
)


def _is_unusable(title: str | None) -> bool:
    return UNUSABLE_RE.match(title or "") is not None


def _title_cfg(cfg: dict, backend: str | None, allow_cuda: bool) -> dict:
    merged = dict(cfg["embedding"])
    extra = cfg.get("title_embedding") or {}
    merged.update({k: v for k, v in extra.items() if v is not None})
    if backend:
        merged["backend"] = backend
    device = str(merged.get("device", "cpu")).lower()
    if device == "cuda" and not allow_cuda:
        raise SystemExit(
            "Refusing CUDA for title embedding on this display GPU. "
            "Use --title-device cpu, or --allow-laptop-cuda on a compute box."
        )
    return {"embedding": merged}


def _remap(labels: np.ndarray) -> np.ndarray:
    mapping = {old: new for new, old in enumerate(sorted(set(int(x) for x in labels.tolist())), start=1)}
    return np.asarray([mapping[int(x)] for x in labels], dtype=int)


def _spherical_kmeans(x: np.ndarray, k: int, rng: np.random.Generator, n_iter: int = 40) -> np.ndarray:
    n = x.shape[0]
    k = min(k, n)
    cents = x[rng.choice(n, k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        labels = (x @ cents.T).argmax(axis=1)
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                cents[j] = x[int(rng.integers(0, n))]
                continue
            m = x[mask].mean(axis=0)
            cents[j] = m / max(float(np.linalg.norm(m)), 1e-12)
    return labels


def _merge_small(x: np.ndarray, labels: np.ndarray, min_size: int) -> np.ndarray:
    labels = labels.copy()
    while True:
        ids, counts = np.unique(labels, return_counts=True)
        if len(ids) <= 2 or int(counts.min()) >= min_size:
            break
        small = int(ids[int(np.argmin(counts))])
        cents = []
        for i in ids:
            m = x[labels == i].mean(axis=0)
            cents.append(m / max(float(np.linalg.norm(m)), 1e-12))
        cents = np.stack(cents)
        si = int(np.where(ids == small)[0][0])
        sim = cents[si] @ cents.T
        sim[si] = -1.0
        target = int(ids[int(np.argmax(sim))])
        labels[labels == small] = target
    return _remap(labels)


def _choose_k_linkage(vecs: np.ndarray, k_min: int, k_max: int) -> tuple[int, np.ndarray, list[dict]]:
    sim = np.clip(vecs @ vecs.T, -1.0, 1.0)
    dist = np.maximum(1.0 - sim, 0.0)
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method="average")
    rows = []
    best_k, best_s, best_lab = k_min, -1.0, None
    n = vecs.shape[0]
    k_hi = min(k_max, max(k_min, n - 1))
    for k in range(k_min, k_hi + 1):
        labels = fcluster(z, k, criterion="maxclust")
        n_lab = len(set(labels.tolist()))
        if n_lab < 2:
            rows.append({"k": k, "n_clusters": n_lab, "silhouette": None})
            continue
        sil = float(silhouette_score(vecs, labels, metric="cosine"))
        rows.append({"k": k, "n_clusters": n_lab, "silhouette": round(sil, 4)})
        if sil > best_s:
            best_k, best_s, best_lab = k, sil, labels
    if best_lab is None:
        best_lab = fcluster(z, k_min, criterion="maxclust")
        best_k = int(len(set(best_lab.tolist())))
    return best_k, _remap(best_lab), rows


def _choose_k_spherical(
    vecs: np.ndarray, k_min: int, k_max: int, min_size: int, seed: int = 42
) -> tuple[int, np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    n = vecs.shape[0]
    k_hi = min(k_max, max(k_min, n // max(min_size, 1)))
    k_hi = max(k_hi, k_min)
    rows = []
    best_s, best_lab = -1.0, None
    for k in range(k_min, k_hi + 1):
        raw = _spherical_kmeans(vecs, k, rng)
        labels = _merge_small(vecs, raw, min_size)
        n_lab = len(set(labels.tolist()))
        sizes = [int(np.sum(labels == c)) for c in sorted(set(labels.tolist()))]
        sil = float(silhouette_score(vecs, labels, metric="cosine")) if n_lab >= 2 else None
        rows.append(
            {
                "k_init": k,
                "n_clusters": n_lab,
                "sizes": sizes,
                "min_size": min(sizes) if sizes else 0,
                "silhouette": None if sil is None else round(sil, 4),
            }
        )
        if sil is not None and sil > best_s:
            best_s, best_lab = sil, labels
    if best_lab is None:
        best_lab = _merge_small(vecs, _spherical_kmeans(vecs, k_min, rng), min_size)
    k_final = int(len(set(best_lab.tolist())))
    return k_final, best_lab, rows


def _choose_k(
    vecs: np.ndarray, k_min: int, k_max: int, method: str, min_size: int
) -> tuple[int, np.ndarray, list[dict]]:
    if method == "average_linkage":
        return _choose_k_linkage(vecs, k_min, k_max)
    if method == "kmeans":
        return _choose_k_spherical(vecs, k_min, k_max, min_size)
    raise SystemExit(f"unknown --cluster-method {method}")



def _keywords(titles: list[str], labels: np.ndarray, top_n: int = 8) -> dict[int, list[str]]:
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_features=4000, stop_words="english")
    try:
        x = vec.fit_transform(titles)
    except ValueError:
        return {int(c): [] for c in set(labels.tolist())}
    names = np.asarray(vec.get_feature_names_out())
    out: dict[int, list[str]] = {}
    global_mean = np.asarray(x.mean(axis=0)).ravel()
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        mean = np.asarray(x[idx].mean(axis=0)).ravel() - global_mean
        order = np.argsort(mean)[::-1]
        words = []
        for j in order:
            tok = str(names[j])
            if tok.isdigit():
                continue
            words.append(tok)
            if len(words) >= top_n:
                break
        out[int(c)] = words
    return out


def _exemplars(titles: list[str], vecs: np.ndarray, labels: np.ndarray, n: int = 5) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        cent = vecs[idx].mean(axis=0)
        cent = cent / max(float(np.linalg.norm(cent)), 1e-12)
        sims = vecs[idx] @ cent
        order = np.argsort(sims)[::-1][:n]
        out[int(c)] = [
            {"title": titles[int(idx[j])], "sim_to_center": round(float(sims[j]), 4)}
            for j in order
        ]
    return out


def _bh(pvals: list[float]) -> list[float]:
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for rank in range(n, 0, -1):
        i = int(order[rank - 1])
        prev = min(prev, pvals[i] * n / rank)
        q[i] = min(prev, 1.0)
    return [round(float(x), 6) for x in q]


def _association(cluster_ids: list[str], motions: list[str]) -> dict:
    usable = [(c, m) for c, m in zip(cluster_ids, motions) if c != "unusable"]
    clusters = sorted({c for c, _ in usable}, key=lambda x: int(x) if str(x).isdigit() else 0)
    table = []
    for c in clusters:
        row = [sum(1 for cc, mm in usable if cc == c and mm == mot) for mot in MOTIONS]
        table.append(row)
    arr = np.asarray(table, dtype=np.int64)
    n = int(arr.sum())
    independence = None
    col_ok = arr.sum(axis=0) > 0
    row_ok = arr.sum(axis=1) > 0
    sub = arr[np.ix_(row_ok, col_ok)] if arr.size else arr
    if sub.size and sub.shape[0] >= 2 and sub.shape[1] >= 2 and int(sub.sum()) >= 8:
        try:
            chi2, p, dof, expected = chi2_contingency(sub, correction=False)
            k = min(sub.shape) - 1
            n_sub = int(sub.sum())
            v = float(np.sqrt(chi2 / (n_sub * k))) if k > 0 and n_sub > 0 else 0.0
            independence = {
                "test": "chi2_independence",
                "chi2": round(float(chi2), 4),
                "p": round(float(p), 6),
                "dof": int(dof),
                "cramers_v": round(v, 4),
                "dropped_zero_rows_or_cols": True,
                "note": "空运动列已去掉。表仍稀，p 值只当脚注。看格子偏离。",
                "expected": np.round(expected, 2).tolist(),
            }
        except ValueError as exc:
            independence = {"test": "chi2_independence", "error": str(exc)}
    elif arr.size:
        independence = {
            "test": "chi2_independence",
            "error": "too_sparse",
            "note": "去掉全零行列之后仍无法做 χ²。只看计数表。",
        }

    baseline = {mot: sum(1 for _, mm in usable if mm == mot) / max(len(usable), 1) for mot in MOTIONS}
    cells = []
    pvals = []
    meta = []
    n_u = len(usable)
    for i, c in enumerate(clusters):
        n_c = int(arr[i].sum())
        for j, mot in enumerate(MOTIONS):
            a = int(arr[i, j])
            b = n_c - a
            n_m = int(arr[:, j].sum())
            c_off = n_m - a
            d = n_u - n_c - c_off
            oddsratio, p = fisher_exact([[a, b], [c_off, d]], alternative="two-sided")
            exp = n_c * (n_m / n_u) if n_u else 0.0
            resid = (a - exp) / np.sqrt(exp) if exp > 0 else 0.0
            meta.append((c, mot, a, n_c, exp, resid, float(oddsratio), float(p)))
            pvals.append(float(p))
    qvals = _bh(pvals)
    for (c, mot, a, n_c, exp, resid, oratio, p), q in zip(meta, qvals):
        cells.append(
            {
                "cluster_id": str(c),
                "motion": mot,
                "motion_zh": MOTION_ZH[mot],
                "count": a,
                "cluster_size": n_c,
                "share_in_cluster": round(a / n_c, 4) if n_c else 0.0,
                "share_baseline": round(baseline[mot], 4),
                "expected": round(float(exp), 3),
                "std_residual": round(float(resid), 3),
                "odds_ratio": None if not np.isfinite(oratio) else round(oratio, 4),
                "fisher_p": round(p, 6),
                "bh_q": q,
                "flag": bool(q < 0.05 and abs(resid) >= 1.5),
            }
        )
    flags = [x for x in cells if x["flag"]]
    return {
        "n_usable": n_u,
        "clusters": clusters,
        "motions": MOTIONS,
        "counts": arr.tolist(),
        "baseline_share": {k: round(v, 4) for k, v in baseline.items()},
        "independence": independence,
        "cells": cells,
        "flagged_cells": flags,
        "flag_rule": "BH q<0.05 且 |标准化残差|>=1.5。平移占多数是基线，不是发现。",
    }


def _label_with_deepseek(cfg: dict, clusters: list[dict]) -> list[dict]:
    key = (cfg.get("simulation") or {}).get("api_key") or ""
    if not key:
        for c in clusters:
            c["label_auto"] = None
            c["definition_auto"] = None
            c["label_source"] = "skipped_no_deepseek_key"
        return clusters
    from openai import OpenAI

    client = OpenAI(
        api_key=key,
        base_url=cfg["simulation"].get("base_url") or "https://api.deepseek.com",
    )
    model = cfg["simulation"].get("model") or "deepseek-chat"
    for c in clusters:
        exemplars = [e["title"] for e in c.get("exemplars", [])]
        words = ", ".join(c.get("keywords", []))
        user = (
            "These Reddit post titles were clustered together.\n"
            f"Keywords: {words}\n"
            "Examples:\n- " + "\n- ".join(exemplars[:5]) + "\n"
            "Reply with JSON only: "
            '{"label": "2-6 word name", "definition": "one sentence, what this cluster asks"}'
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.2,
                max_tokens=120,
                messages=[
                    {
                        "role": "system",
                        "content": "Name a cluster of forum titles. No speculation about comment threads.",
                    },
                    {"role": "user", "content": user},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.I | re.M).strip()
            parsed = json.loads(text)
            c["label_auto"] = str(parsed.get("label") or "").strip() or None
            c["definition_auto"] = str(parsed.get("definition") or "").strip() or None
            c["label_source"] = f"deepseek:{model}"
        except Exception as exc:
            c["label_auto"] = None
            c["definition_auto"] = None
            c["label_source"] = f"failed:{type(exc).__name__}"
    return clusters


def _html(
    tag: str,
    assoc: dict,
    clusters: list[dict],
    embedder_name: str,
    k_search: list,
    method: str,
    min_size: int,
) -> str:
    def esc(s: object) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    head_m = "".join(f"<th>{esc(MOTION_ZH[m])}</th>" for m in MOTIONS)
    body_rows = []
    for i, cid in enumerate(assoc["clusters"]):
        cells = "".join(f"<td>{assoc['counts'][i][j]}</td>" for j in range(len(MOTIONS)))
        n_c = sum(assoc["counts"][i])
        lab = next((c.get("label_auto") or f"簇 {cid}") for c in clusters if str(c["cluster_id"]) == str(cid))
        body_rows.append(f"<tr><th>{esc(lab)} <code>{esc(cid)}</code> n={n_c}</th>{cells}</tr>")
    base = "".join(
        f"<td>{100 * assoc['baseline_share'][m]:.1f}%</td>" for m in MOTIONS
    )
    flags = assoc.get("flagged_cells") or []
    if flags:
        flag_html = "<ul>" + "".join(
            f"<li>簇 {esc(f['cluster_id'])} × {esc(f['motion_zh'])}："
            f"簇内 {100 * f['share_in_cluster']:.1f}% ，全体 {100 * f['share_baseline']:.1f}% "
            f"（残差 {f['std_residual']:+.2f}，q={f['bh_q']}）</li>"
            for f in flags
        ) + "</ul>"
    else:
        flag_html = "<p>没有格子同时满足 BH q&lt;0.05 和 |残差|≥1.5。多数簇仍以平移为主，这是基线。</p>"

    cluster_blocks = []
    for c in clusters:
        ex = "</li><li>".join(esc(e["title"]) for e in c.get("exemplars", []))
        cluster_blocks.append(
            f"<h3>簇 {esc(c['cluster_id'])}　{esc(c.get('label_auto') or '（尚无自动名）')}</h3>"
            f"<p>{esc(c.get('definition_auto') or '没有生成定义。')} "
            f"<span style='color:#656d76'>n={c['size']}　{esc(c.get('label_source'))}</span></p>"
            f"<p>突出词：{esc(', '.join(c.get('keywords') or []))}</p>"
            f"<ol><li>{ex}</li></ol>"
            if ex
            else f"<h3>簇 {esc(c['cluster_id'])}</h3>"
        )
    indep = assoc.get("independence") or {}
    indep_line = (
        f"χ²={indep.get('chi2')}　p={indep.get('p')}　Cramér's V={indep.get('cramers_v')}。"
        if indep.get("chi2") is not None
        else esc(indep.get("error") or indep.get("note") or "")
    )
    sizes = [int(c.get("size") or 0) for c in clusters]
    blob = max(sizes) if sizes else 0
    sil = None
    if k_search:
        sil = max((x.get("silhouette") or -1) for x in k_search)
    cluster_warn = (
        f"<p><strong>切法：</strong>{esc(method)}，最小簇 {min_size}。"
        f"轮廓系数最高 {sil:.3f}，最大簇 {blob} / {sum(sizes)}。"
        f"簇大小：{esc(sizes)}。</p>"
        if sil is not None
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>标题簇 × 运动类 · {esc(tag)}</title>
<style>
body {{ font-family: "Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; margin: 0; background:#f6f8fa; color:#1f2328; }}
.wrap {{ max-width: 1080px; margin: 0 auto; padding: 32px 24px 64px; background:#fff; box-shadow: 0 0 0 1px #d0d7de; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; margin: 1rem 0; }}
th, td {{ border: 1px solid #d0d7de; padding: 5px 8px; }}
th {{ background: #f6f8fa; }}
code {{ background:#f6f8fa; padding: 0.1em 0.35em; }}
h1 {{ margin-top: 0; }}
</style>
</head>
<body>
<div class="wrap">
<h1>标题簇 × 九类运动 · {esc(tag)}</h1>
<p>第 3 级。只嵌了标题（{esc(embedder_name)}），再对上第 2 级已经打好的运动标签。
标题向量没有写进评论云。</p>
{cluster_warn}
<p>轮廓系数选 k：{esc(k_search)}。不可用标题不进表。</p>
<p>独立性（脚注）：{indep_line}</p>
<table>
<thead><tr><th>簇 \\ 运动</th>{head_m}</tr></thead>
<tbody>
<tr><th>全体可用帖的占比（基线）</th>{base}</tr>
{"".join(body_rows)}
</tbody>
</table>
<h2>相对基线明显偏离的格子</h2>
{flag_html}
<h2>每簇在问什么</h2>
{"".join(cluster_blocks)}
<p style="color:#656d76">契约见 docs/pipeline.md。重新生成：python scripts/run_pipeline.py --from 3 --tag {esc(tag)}</p>
</div>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tag", default="chik200")
    ap.add_argument("--title-backend", default=None)
    ap.add_argument("--title-device", default=None)
    ap.add_argument("--allow-laptop-cuda", action="store_true")
    ap.add_argument("--k-min", type=int, default=5)
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument(
        "--cluster-method",
        choices=("kmeans", "average_linkage"),
        default="kmeans",
        help="kmeans = spherical k-means then merge tiny clusters",
    )
    ap.add_argument("--min-cluster-size", type=int, default=15)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.title_device:
        cfg.setdefault("title_embedding", {})["device"] = args.title_device
    tcfg = _title_cfg(cfg, args.title_backend, args.allow_laptop_cuda)
    embedder = build_embedder(tcfg)

    lab_path = ROOT / "outputs" / "labels" / f"thread_motions_{args.tag}.jsonl"
    labeled = [json.loads(l) for l in lab_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    usable_idx = []
    usable_titles = []
    for i, row in enumerate(labeled):
        if _is_unusable(row.get("title")):
            continue
        usable_idx.append(i)
        usable_titles.append(row.get("title") or "")

    if len(usable_titles) < args.k_min:
        raise SystemExit(f"usable titles={len(usable_titles)} < k_min={args.k_min}")

    print(
        f"embed {len(usable_titles)} titles with {embedder.name} "
        f"method={args.cluster_method} min_size={args.min_cluster_size}",
        flush=True,
    )
    vecs = embedder.embed(usable_titles)
    k, labels, k_search = _choose_k(
        vecs, args.k_min, args.k_max, args.cluster_method, args.min_cluster_size
    )
    keywords = _keywords(usable_titles, labels)
    exemplars = _exemplars(usable_titles, vecs, labels, n=8)

    cluster_ids = ["unusable"] * len(labeled)
    for i, lab in zip(usable_idx, labels):
        cluster_ids[i] = str(int(lab))
    motions = [r.get("motion") or "" for r in labeled]

    cluster_docs = []
    for cid in sorted({int(x) for x in labels.tolist()}):
        size = int(np.sum(labels == cid))
        cluster_docs.append(
            {
                "cluster_id": str(cid),
                "size": size,
                "keywords": keywords.get(cid, []),
                "exemplars": exemplars.get(cid, []),
            }
        )
    cluster_docs = _label_with_deepseek(cfg, cluster_docs)

    assoc = _association(cluster_ids, motions)

    out_dir = ROOT / "outputs" / "title_clusters"
    met_dir = ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    per_thread = out_dir / f"title_clusters_{args.tag}.jsonl"
    with per_thread.open("w", encoding="utf-8") as f:
        for row, cid in zip(labeled, cluster_ids):
            f.write(
                json.dumps(
                    {
                        "post_id": row["post_id"],
                        "title": row.get("title"),
                        "cluster_id": cid,
                        "motion": row.get("motion"),
                        "embedder": embedder.name,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    clusters_path = out_dir / f"clusters_{args.tag}.json"
    clusters_payload = {
        "tag": args.tag,
        "embedder": embedder.name,
        "k": k,
        "cluster_method": args.cluster_method,
        "min_cluster_size": args.min_cluster_size,
        "k_search": k_search,
        "n_threads": len(labeled),
        "n_usable": len(usable_titles),
        "n_unusable": len(labeled) - len(usable_titles),
        "clusters": cluster_docs,
    }
    clusters_path.write_text(json.dumps(clusters_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    assoc_path = met_dir / f"title_cluster_vs_motion_{args.tag}.json"
    assoc_out = {
        "tag": args.tag,
        "embedder": embedder.name,
        "k": k,
        "cluster_method": args.cluster_method,
        "min_cluster_size": args.min_cluster_size,
        "input_labels": str(lab_path),
        **assoc,
    }
    assoc_path.write_text(json.dumps(assoc_out, indent=2, ensure_ascii=False), encoding="utf-8")

    html_path = ROOT / "docs" / f"title_clusters_{args.tag}.html"
    html_path.write_text(
        _html(
            args.tag,
            assoc,
            cluster_docs,
            embedder.name,
            k_search,
            args.cluster_method,
            args.min_cluster_size,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "k": k,
                "cluster_method": args.cluster_method,
                "sizes": [c["size"] for c in cluster_docs],
                "n_usable": len(usable_titles),
                "n_unusable": len(labeled) - len(usable_titles),
                "embedder": embedder.name,
                "n_flagged_cells": len(assoc["flagged_cells"]),
                "independence": assoc.get("independence"),
                "wrote": [str(per_thread), str(clusters_path), str(assoc_path), str(html_path)],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
