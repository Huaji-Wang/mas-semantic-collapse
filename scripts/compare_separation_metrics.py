"""Compare Vendi, average converge distance, and effective dim on human extremes, then score AI sims."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mas_collapse.config import load_config
from mas_collapse.data.threads import iter_threads, linearize_comments
from mas_collapse.embed.backend import build_embedder
from mas_collapse.metrics.semantic import (
    _vendi_from_embeddings,
    average_converge_distance,
    late_mean,
    participation_ratio,
    window_similarities_to_first,
)


def _windows(bodies: list[str], window_size: int) -> list[list[str]]:
    return [bodies[i : i + window_size] for i in range(0, len(bodies), window_size) if bodies[i : i + window_size]]


def _sample_mean(embs: np.ndarray, fn, sample_size: int, repeats: int, rng: np.random.Generator) -> float:
    if len(embs) == 0:
        return float("nan")
    m = min(sample_size, len(embs))
    vals = []
    for _ in range(repeats):
        idx = rng.choice(len(embs), size=m, replace=False) if len(embs) > m else np.arange(len(embs))
        vals.append(float(fn(embs[idx])))
    return float(np.mean(vals))


def score_bodies(bodies: list[str], embedder, *, window_size: int, late_windows: int, sample_size: int, repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    chunks = _windows(bodies, window_size)
    all_embs = embedder.embed(bodies)
    offset = 0
    window_embs = []
    utt_per_window = []
    for chunk in chunks:
        embs = all_embs[offset : offset + len(chunk)]
        offset += len(chunk)
        utt_per_window.append(embs)
        window_embs.append(embs.mean(axis=0))
    window_embs = np.stack(window_embs, axis=0)
    # re-normalize window means
    norms = np.linalg.norm(window_embs, axis=1, keepdims=True)
    window_embs = window_embs / np.maximum(norms, 1e-12)

    sims = window_similarities_to_first(window_embs)
    dist = average_converge_distance(sims)

    vendi_by_w = [
        _sample_mean(e, _vendi_from_embeddings, sample_size, repeats, rng) for e in utt_per_window
    ]
    vendi_late = late_mean(vendi_by_w, late_windows)

    k = min(late_windows, len(utt_per_window))
    late_embs = np.concatenate(utt_per_window[-k:], axis=0) if k else utt_per_window[0]
    effdim = _sample_mean(late_embs, participation_ratio, sample_size, repeats, rng)

    return {
        "n_comments": len(bodies),
        "n_windows": len(sims),
        "s_end": late_mean(sims, late_windows),
        "avg_converge_distance": dist,
        "vendi_late": vendi_late,
        "effective_dim": effdim,
        "sims_to_first": [float(x) for x in sims],
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    labels_path = ROOT / "outputs" / "labels" / "thread_labels_q80q20.jsonl"
    rows = [json.loads(l) for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    wanted = {r["post_id"]: r["label_A"] for r in rows if r["label_A"] in {"converge", "non_converge"}}

    embedder = build_embedder(cfg)
    window_size = int(cfg["windowing"]["window_size"])
    late_windows = int(cfg["windowing"]["late_windows"])
    sample_size = int(cfg["metrics"].get("vendi_sample_size", 30))
    repeats = int(cfg["metrics"].get("vendi_repeats", 20))
    seed = int(cfg["project"].get("seed", 42))

    scored: list[dict] = []
    found = 0
    pbar = tqdm(total=len(wanted), desc="scored", unit="thread")
    for th in iter_threads(
        cfg["data"]["threads_path"],
        min_comments=int(cfg["data"]["min_comments"]),
        max_comments=None,
    ):
        if th.post_id not in wanted:
            continue
        bodies = [c.body for c in linearize_comments(th)]
        metrics = score_bodies(
            bodies,
            embedder,
            window_size=window_size,
            late_windows=late_windows,
            sample_size=sample_size,
            repeats=repeats,
            seed=seed,
        )
        metrics["post_id"] = th.post_id
        metrics["label_A"] = wanted[th.post_id]
        metrics["title"] = th.title
        metrics["subreddit"] = th.subreddit
        scored.append(metrics)
        found += 1
        pbar.set_postfix(label=wanted[th.post_id], pid=th.post_id)
        pbar.update(1)
        if found >= len(wanted):
            break
    pbar.close()

    out_dir = ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "human_separation_80.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in scored),
        encoding="utf-8",
    )

    metric_names = ["avg_converge_distance", "vendi_late", "effective_dim", "s_end"]
    summary = {"n": len(scored), "by_metric": {}}
    conv = [r for r in scored if r["label_A"] == "converge"]
    non = [r for r in scored if r["label_A"] == "non_converge"]
    for name in metric_names:
        a = np.array([r[name] for r in conv], dtype=float)
        b = np.array([r[name] for r in non], dtype=float)
        d = cohens_d(a, b)
        summary["by_metric"][name] = {
            "converge_mean": float(np.mean(a)),
            "non_converge_mean": float(np.mean(b)),
            "converge_std": float(np.std(a, ddof=1)),
            "non_converge_std": float(np.std(b, ddof=1)),
            "cohens_d_conv_minus_non": d,
            "abs_d": abs(d),
        }

    ranked = sorted(summary["by_metric"].items(), key=lambda kv: kv[1]["abs_d"], reverse=True)
    summary["winner"] = ranked[0][0]
    summary["ranking"] = [(k, v["abs_d"]) for k, v in ranked]

    # AI smoke sims (80-comment)
    sim_dir = ROOT / "outputs" / "sims" / "smoke_flat_post_only"
    ai_rows = []
    sim_files = [fp for fp in sorted(sim_dir.glob("*.json"))]
    for fp in tqdm(sim_files, desc="ai_sims"):
        data = json.loads(fp.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "comments" not in data:
            continue
        bodies = [c["body"] for c in data["comments"]]
        if len(bodies) < 50:
            continue
        metrics = score_bodies(
            bodies,
            embedder,
            window_size=window_size,
            late_windows=late_windows,
            sample_size=sample_size,
            repeats=repeats,
            seed=seed,
        )
        metrics["source"] = "ai_sim"
        metrics["human_label_A"] = data.get("human_label_A")
        metrics["post_id"] = data.get("post_id")
        ai_rows.append(metrics)

    summary["ai_sims"] = ai_rows
    (out_dir / "separation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in ("n", "winner", "ranking", "by_metric")}, indent=2, ensure_ascii=False))
    print("ai_n", len(ai_rows))
    print("wrote", out_dir / "separation_summary.json")


if __name__ == "__main__":
    main()
