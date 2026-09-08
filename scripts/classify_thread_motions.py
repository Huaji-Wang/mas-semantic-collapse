"""Whole-thread motion labels from the five time series.

One thread, one class. Each teacher-table row is scored by how well the
whole trajectories match its arrows; the highest score wins. Filamentation
is not scored. Chi is the new polar series, k is range-cluster count.

    python scripts/classify_thread_motions.py --tag chik200
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans SC", "Segoe UI"],
        "axes.unicode_minus": False,
    }
)

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
COLORS = {
    "translation": "#64b5cd",
    "diffusion": "#4c72b0",
    "condensation": "#c44e52",
    "fission": "#dd8452",
    "crystallization": "#6b4c9a",
    "vortex": "#8172b3",
    "orbit": "#937860",
    "breathing": "#55a868",
    "cascade": "#ccb974",
}


def _arr(vals) -> np.ndarray:
    return np.asarray([np.nan if x is None else float(x) for x in vals], dtype=np.float64)


def _finite(a: np.ndarray) -> np.ndarray:
    return a[np.isfinite(a)]


def _head_tail(a: np.ndarray, late: int = 3) -> tuple[float, float, float]:
    a = _finite(a)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    n_early = max(1, a.size // 3)
    early = float(a[:n_early].mean())
    late_v = float(a[-min(late, a.size) :].mean())
    return early, late_v, late_v - early


def _stable(delta: float, scale: float) -> float:
    if not np.isfinite(delta) or scale <= 1e-12:
        return 0.0
    return float(np.exp(-abs(delta) / scale))


def _signed(delta: float, scale: float) -> tuple[float, float]:
    """Return (up, down) in [0, 1]."""
    if not np.isfinite(delta) or scale <= 1e-12:
        return 0.0, 0.0
    z = delta / scale
    return float(1.0 / (1.0 + np.exp(-z))), float(1.0 / (1.0 + np.exp(z)))


def _dip_recover(y: np.ndarray) -> float:
    """Leave-and-return on a high-at-start series (sims_to_first)."""
    y = _finite(y)
    if y.size < 4:
        return 0.0
    amin = float(y.min())
    imin = int(np.argmin(y))
    drop = float(y[0] - amin)
    rec = float(y[-1] - amin)
    if drop < 0.04 or imin == 0 or imin >= y.size - 1:
        return 0.0
    return float(np.clip((rec / drop) * np.clip(drop / 0.12, 0.0, 1.0), 0.0, 1.0))


def _rise_fall(y: np.ndarray) -> float:
    """Peak in the middle, ends low (chi vs first on an orbit)."""
    y = _finite(y)
    if y.size < 4:
        return 0.0
    amax = float(y.max())
    imax = int(np.argmax(y))
    if amax < 0.12 or imax == 0 or imax >= y.size - 1:
        return 0.0
    ends = 0.5 * (float(y[0]) + float(y[-1]))
    return float(np.clip((amax - ends) / amax, 0.0, 1.0))


def _step(y: np.ndarray) -> float:
    """One big adjacent jump, then a flatter tail."""
    y = _finite(y)
    if y.size < 4:
        return 0.0
    d = np.abs(np.diff(y))
    if d.sum() <= 1e-12:
        return 0.0
    j = int(np.argmax(d))
    share = float(d[j] / d.sum())
    tail = y[j + 1 :]
    tail_flat = 1.0
    if tail.size >= 2:
        tail_flat = float(np.exp(-tail.std() / (d[j] + 1e-9)))
    return float(np.clip(share * tail_flat, 0.0, 1.0))


def features(row: dict) -> dict[str, float]:
    sims = _arr(row["sims_to_first"])
    sigma = _arr(row["sigma"])
    dim = _arr(row["eff_dim"])
    k = _arr(row["k_modes"])
    chi = _arr(row.get("chi", []))
    chi_step = _arr(row.get("chi_step", []))

    _, _, ds = _head_tail(sigma)
    _, _, dd = _head_tail(dim)
    _, _, dk = _head_tail(k)
    _, _, dc = _head_tail(chi)
    m_leave = float(1.0 - row.get("s_end", sims[-1] if sims.size else 1.0))
    step_mean = float(np.nanmean(chi_step[1:])) if chi_step.size > 1 else float("nan")
    chi_late_std = float(np.nanstd(_finite(chi)[-3:])) if _finite(chi).size else float("nan")

    return {
        "m_leave": m_leave,
        "m_wave": _dip_recover(sims),
        "m_step": _step(1.0 - sims),
        "d_sigma": ds,
        "d_dim": dd,
        "d_k": dk,
        "d_chi": dc,
        "sigma_wave": _rise_fall(sigma - np.nanmin(sigma)) if sigma.size else 0.0,
        "chi_wave": _rise_fall(chi),
        "chi_step_shape": _step(chi),
        "chi_step_mean": step_mean,
        "chi_late_std": chi_late_std,
        "adj_sim_mean": float(row.get("adj_sim_mean", np.nan)),
        "s_end": float(row.get("s_end", np.nan)),
        "k_first": float(row.get("k_modes_first", k[0] if k.size else np.nan)),
        "k_late": float(row.get("k_modes_late", k[-1] if k.size else np.nan)),
    }


def _scales(feats: list[dict]) -> dict[str, float]:
    def mad(key: str) -> float:
        a = _finite(np.asarray([f[key] for f in feats], dtype=np.float64))
        if a.size < 4:
            return 1.0
        s = float(np.median(np.abs(a - np.median(a))) * 1.4826)
        return s if s > 1e-6 else max(float(np.std(a)), 1e-3)

    return {
        "sigma": mad("d_sigma"),
        "dim": mad("d_dim"),
        "k": max(mad("d_k"), 0.35),
        "chi": mad("d_chi"),
        "m": mad("m_leave"),
        "step": mad("chi_step_mean"),
        "chi_std": mad("chi_late_std"),
    }


def score_row(f: dict[str, float], sc: dict[str, float]) -> dict[str, float]:
    sig_up, sig_dn = _signed(f["d_sigma"], sc["sigma"])
    k_up, k_dn = _signed(f["d_k"], sc["k"])
    chi_up, chi_dn = _signed(f["d_chi"], sc["chi"])
    sig_st = _stable(f["d_sigma"], sc["sigma"])
    dim_st = _stable(f["d_dim"], sc["dim"])
    k_st = _stable(f["d_k"], sc["k"])
    chi_st = _stable(f["d_chi"], sc["chi"])
    m_move = float(np.clip(f["m_leave"] / (sc["m"] * 2.0 + 1e-9), 0.0, 1.0))
    m_stay = 1.0 - m_move
    freeze = _stable(f["chi_step_mean"], sc["step"]) * _stable(f["chi_late_std"], sc["chi_std"])
    flow = 1.0 - freeze
    mw, cw = f["m_wave"], f["chi_wave"]
    sw, mstep, cstep = f["sigma_wave"], f["m_step"], f["chi_step_shape"]

    return {
        "translation": 1.4 * m_move + 0.6 * sig_st + 0.4 * dim_st + 0.4 * k_st + 0.8 * chi_st - 0.8 * mw,
        "diffusion": 1.2 * sig_up + 1.0 * chi_up + 0.4 * m_stay + 0.3 * k_st,
        "condensation": 1.2 * sig_dn + 1.0 * chi_dn + 0.4 * m_stay,
        "fission": 1.0 * sig_up + 1.4 * k_up + 0.8 * chi_dn + 0.3 * m_stay,
        "crystallization": 1.6 * freeze + 0.5 * sig_st + 0.4 * k_st + 0.4 * m_stay - 0.6 * flow,
        "vortex": 1.4 * flow + 0.5 * m_stay + 0.4 * sig_st + 0.4 * k_st - 0.7 * cw - 0.5 * freeze,
        "orbit": 1.3 * mw + 1.3 * cw + 0.3 * sig_st,
        "breathing": 1.2 * sw + 1.2 * cw + 0.3 * m_stay,
        "cascade": 1.3 * mstep + 1.1 * cstep + 0.3 * sig_st,
    }


def classify(f: dict[str, float], sc: dict[str, float]) -> tuple[str, dict[str, float]]:
    scores = score_row(f, sc)
    label = max(scores, key=scores.get)
    return label, scores


def draw_counts(counter: Counter, n: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    xs = np.arange(len(MOTIONS))
    ys = [counter.get(k, 0) / n for k in MOTIONS]
    ax.bar(xs, ys, color=[COLORS[k] for k in MOTIONS])
    ax.set_xticks(xs)
    ax.set_xticklabels([MOTION_ZH[k] for k in MOTIONS], rotation=20, ha="right")
    ax.set_ylabel("帖占比")
    ax.set_title(f"整帖运动分类（新 χ / 范围 k，n={n}）")
    ax.grid(axis="y", alpha=0.3)
    for i, y in enumerate(ys):
        ax.text(i, y + 0.008, f"{counter.get(MOTIONS[i], 0)}\n{100 * y:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def draw_vs_send(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    rng = np.random.default_rng(0)
    for mot in MOTIONS:
        pts = [r for r in rows if r["motion"] == mot]
        if not pts:
            continue
        y = np.asarray([r["s_end"] for r in pts], dtype=float)
        x = np.full(len(y), MOTIONS.index(mot), dtype=float) + rng.normal(0, 0.08, size=len(y))
        ax.scatter(x, y, s=18, alpha=0.6, color=COLORS[mot], label=MOTION_ZH[mot], edgecolors="none")
    ax.set_xticks(np.arange(len(MOTIONS)))
    ax.set_xticklabels([MOTION_ZH[k] for k in MOTIONS], rotation=20, ha="right")
    ax.set_ylabel("S_end")
    ax.set_title("整帖运动 vs 末期仍像开头（S_end）")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def draw_examples(src_rows: list[dict], labeled: list[dict], out: Path) -> None:
    by_id = {r["post_id"]: r for r in src_rows}
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.2), sharex=True)
    for ax, mot in zip(axes.ravel(), MOTIONS):
        hits = [x for x in labeled if x["motion"] == mot]
        ax.set_title(f"{MOTION_ZH[mot]}  n={len(hits)}")
        if not hits:
            ax.set_axis_off()
            continue
        hits = sorted(hits, key=lambda r: -r["margin"])[:8]
        for h in hits:
            chi = _arr(by_id[h["post_id"]]["chi"])
            if chi.size < 2:
                continue
            x = np.linspace(0, 1, len(chi))
            ax.plot(x, chi, color=COLORS[mot], alpha=0.55, lw=1.2)
        ax.set_ylabel("χ")
        ax.grid(alpha=0.25)
    fig.suptitle("各类里 χ(t) 的样子（每类最多 8 条分差最大的帖）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="chik200")
    ap.add_argument("--jsonl", default=None)
    args = ap.parse_args()

    src = Path(args.jsonl) if args.jsonl else ROOT / "outputs" / "order_params" / f"order_params_{args.tag}.jsonl"
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    feats = [features(r) for r in rows]
    sc = _scales(feats)

    labeled = []
    for r, f in zip(rows, feats):
        lab, scores = classify(f, sc)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        margin = ranked[0][1] - ranked[1][1]
        labeled.append(
            {
                "post_id": r["post_id"],
                "title": r.get("title"),
                "n_windows": r["n_windows"],
                "s_end": r.get("s_end"),
                "motion": lab,
                "margin": round(float(margin), 4),
                "scores": {k: round(float(v), 4) for k, v in scores.items()},
                "features": {k: (None if not np.isfinite(v) else round(float(v), 4)) for k, v in f.items()},
            }
        )

    counts = Counter(x["motion"] for x in labeled)
    fig_dir = ROOT / "outputs" / "figures"
    lab_dir = ROOT / "outputs" / "labels"
    met_dir = ROOT / "outputs" / "metrics"
    for d in (fig_dir, lab_dir, met_dir):
        d.mkdir(parents=True, exist_ok=True)

    draw_counts(counts, len(labeled), fig_dir / f"thread_motions_counts_{args.tag}.png")
    draw_vs_send(labeled, fig_dir / f"thread_motions_send_{args.tag}.png")
    draw_examples(rows, labeled, fig_dir / f"thread_motions_chi_{args.tag}.png")

    out_jsonl = lab_dir / f"thread_motions_{args.tag}.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in labeled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "n_threads": len(labeled),
        "source": str(src),
        "filamentation": "dropped",
        "chi": "polar 0.5(1-cos dtheta) vs first window",
        "k": "range cluster distance 0.55",
        "rule": "whole-thread soft scores, argmax, one label per thread",
        "counts": {k: counts.get(k, 0) for k in MOTIONS},
        "fraction": {k: round(counts.get(k, 0) / len(labeled), 4) for k in MOTIONS},
        "median_margin": round(float(np.median([x["margin"] for x in labeled])), 4),
        "mean_s_end_by_class": {
            k: round(
                float(np.mean([x["s_end"] for x in labeled if x["motion"] == k and x["s_end"] == x["s_end"]])),
                4,
            )
            if any(x["motion"] == k for x in labeled)
            else None
            for k in MOTIONS
        },
    }
    out_stats = met_dir / f"thread_motions_{args.tag}_stats.json"
    out_stats.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {out_jsonl}")
    print(f"wrote {out_stats}")


if __name__ == "__main__":
    main()
