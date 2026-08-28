"""Exclusive per-step motion labels from the five order-parameter trajectories.

Locked decisions (2026-08-26 grilling):
  * label adjacent steps, not whole threads
  * identity-free chi is a Reddit background; Crystallization = relative drop only
  * reduced atlas; Orbit / Vortex / Breathing / Cascade are undetectable here
  * exclusive priority:
      crystallization -> condensation -> fission -> filamentation
      -> diffusion -> translation -> turnover_background

Thresholds are within-sample percentiles of the n=200 step pool, not physical constants.

Example:
    python scripts/classify_step_motions.py --tag pilot200
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans SC", "Segoe UI"],
        "axes.unicode_minus": False,
    }
)

MOTION_ORDER = [
    "crystallization",
    "condensation",
    "fission",
    "filamentation",
    "diffusion",
    "translation",
    "turnover_background",
]
MOTION_ZH = {
    "crystallization": "结晶（χ 相对低）",
    "condensation": "凝聚（σ 下降）",
    "fission": "裂变代理（二分上升）",
    "filamentation": "成丝（d 下降）",
    "diffusion": "扩散（σ 上升）",
    "translation": "平移（m 动、其余稳）",
    "turnover_background": "换位背景",
}
MOTION_COLORS = {
    "crystallization": "#6b4c9a",
    "condensation": "#c44e52",
    "fission": "#dd8452",
    "filamentation": "#55a868",
    "diffusion": "#4c72b0",
    "translation": "#64b5cd",
    "turnover_background": "#bbbbbb",
}
UNDETECTABLE = ["orbit", "vortex", "breathing", "cascade"]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def step_features(row: dict) -> list[dict]:
    t_n = int(row["n_windows"])
    out = []
    for t in range(1, t_n):
        chi = row["churn_config"][t]
        ds = row["sigma"][t] - row["sigma"][t - 1]
        dd = row["eff_dim"][t] - row["eff_dim"][t - 1]
        db = row["bimodality"][t] - row["bimodality"][t - 1]
        dm = 1.0 - float(row["adj_sims"][t])
        if not np.isfinite(chi):
            continue
        out.append(
            {
                "post_id": row["post_id"],
                "t": t,
                "t_frac": t / max(t_n - 1, 1),
                "n_windows": t_n,
                "s_end": row.get("s_end"),
                "delta_m": dm,
                "delta_sigma": ds,
                "delta_d": dd,
                "delta_bimod": db,
                "chi": chi,
            }
        )
    return out


def percentile_thresholds(steps: list[dict]) -> dict[str, float]:
    def col(key: str) -> np.ndarray:
        a = np.asarray([s[key] for s in steps], dtype=np.float64)
        return a[np.isfinite(a)]

    ds = col("delta_sigma")
    dd = col("delta_d")
    db = col("delta_bimod")
    dm = col("delta_m")
    chi = col("chi")
    return {
        "chi_low": float(np.percentile(chi, 20)),
        "sigma_lo": float(np.percentile(ds, 20)),
        "sigma_hi": float(np.percentile(ds, 80)),
        "d_lo": float(np.percentile(dd, 20)),
        "bimod_hi": float(np.percentile(db, 80)),
        "m_move": float(np.percentile(dm, 75)),
        "sigma_stable": float(np.percentile(np.abs(ds), 50)),
        "d_stable": float(np.percentile(np.abs(dd), 50)),
        "bimod_stable": float(np.percentile(np.abs(db), 50)),
    }


def classify_step(s: dict, thr: dict[str, float]) -> str:
    ds, dd, db, dm, chi = s["delta_sigma"], s["delta_d"], s["delta_bimod"], s["delta_m"], s["chi"]
    if (
        chi <= thr["chi_low"]
        and abs(ds) <= thr["sigma_stable"]
        and abs(dd) <= thr["d_stable"]
        and abs(db) <= thr["bimod_stable"]
    ):
        return "crystallization"
    if ds <= thr["sigma_lo"]:
        return "condensation"
    if db >= thr["bimod_hi"]:
        return "fission"
    if dd <= thr["d_lo"]:
        return "filamentation"
    if ds >= thr["sigma_hi"]:
        return "diffusion"
    if (
        dm >= thr["m_move"]
        and abs(ds) <= thr["sigma_stable"]
        and abs(dd) <= thr["d_stable"]
        and abs(db) <= thr["bimod_stable"]
    ):
        return "translation"
    return "turnover_background"


def late_majority(labels: list[str], times: list[int], n_windows: int, late: int = 3) -> str:
    start = max(1, n_windows - late)
    picked = [lab for lab, t in zip(labels, times) if t >= start]
    if not picked:
        return "turnover_background"
    c = Counter(picked)
    return c.most_common(1)[0][0]


def draw_counts(counter: Counter, n_steps: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    xs = np.arange(len(MOTION_ORDER))
    ys = [counter.get(k, 0) / n_steps for k in MOTION_ORDER]
    ax.bar(xs, ys, color=[MOTION_COLORS[k] for k in MOTION_ORDER])
    ax.set_xticks(xs)
    ax.set_xticklabels([MOTION_ZH[k] for k in MOTION_ORDER], rotation=25, ha="right")
    ax.set_ylabel("相邻步占比")
    ax.set_title(f"相邻步运动标签（互斥，n_steps={n_steps}）")
    ax.grid(axis="y", alpha=0.3)
    for i, y in enumerate(ys):
        ax.text(i, y + 0.01, f"{100 * y:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def draw_progress(steps: list[dict], out: Path, bins: int = 10) -> None:
    edges = np.linspace(0.0, 1.0, bins + 1)
    mat = np.zeros((len(MOTION_ORDER), bins))
    idx = {k: i for i, k in enumerate(MOTION_ORDER)}
    for s in steps:
        b = min(bins - 1, int(s["t_frac"] * bins))
        mat[idx[s["motion"]], b] += 1
    col = mat.sum(axis=0)
    col = np.maximum(col, 1.0)
    frac = mat / col
    x = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.stackplot(
        x,
        frac,
        labels=[MOTION_ZH[k] for k in MOTION_ORDER],
        colors=[MOTION_COLORS[k] for k in MOTION_ORDER],
        alpha=0.92,
    )
    ax.set_xlabel("讨论进度（归一化）")
    ax.set_ylabel("该进度上的步占比")
    ax.set_title("相邻步运动随讨论进度的构成（堆叠，每步一类）")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def draw_late_vs_send(thread_rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    rng = np.random.default_rng(0)
    for mot in MOTION_ORDER:
        pts = [r for r in thread_rows if r["late_motion"] == mot]
        if not pts:
            continue
        y = np.asarray([r["s_end"] for r in pts], dtype=float)
        x = np.full(len(y), MOTION_ORDER.index(mot), dtype=float)
        x = x + rng.normal(0, 0.08, size=len(x))
        ax.scatter(x, y, s=18, alpha=0.55, color=MOTION_COLORS[mot], label=MOTION_ZH[mot], edgecolors="none")
    ax.set_xticks(np.arange(len(MOTION_ORDER)))
    ax.set_xticklabels([MOTION_ZH[k] for k in MOTION_ORDER], rotation=25, ha="right")
    ax.set_ylabel("S_end")
    ax.set_title("帖级末期多数运动 vs S_end（每帖末期最多 3 步的众数）")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot200")
    ap.add_argument("--jsonl", default=None)
    args = ap.parse_args()

    src = Path(args.jsonl) if args.jsonl else ROOT / "outputs" / "order_params" / f"order_params_{args.tag}.jsonl"
    rows = load_rows(src)
    steps: list[dict] = []
    for r in rows:
        steps.extend(step_features(r))
    if not steps:
        raise SystemExit("no steps")

    thr = percentile_thresholds(steps)
    for s in steps:
        s["motion"] = classify_step(s, thr)

    by_post: dict[str, list[dict]] = defaultdict(list)
    for s in steps:
        by_post[s["post_id"]].append(s)

    thread_rows = []
    send_map = {r["post_id"]: r.get("s_end") for r in rows}
    nw_map = {r["post_id"]: r["n_windows"] for r in rows}
    for pid, ss in by_post.items():
        ss = sorted(ss, key=lambda x: x["t"])
        labels = [s["motion"] for s in ss]
        late = late_majority(labels, [s["t"] for s in ss], nw_map[pid])
        thread_rows.append(
            {
                "post_id": pid,
                "s_end": send_map[pid],
                "n_windows": nw_map[pid],
                "n_steps": len(ss),
                "late_motion": late,
                "step_counts": dict(Counter(labels)),
            }
        )

    fig_dir = ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    lab_dir = ROOT / "outputs" / "labels"
    lab_dir.mkdir(parents=True, exist_ok=True)
    met_dir = ROOT / "outputs" / "metrics"
    met_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter(s["motion"] for s in steps)
    draw_counts(counts, len(steps), fig_dir / f"step_motions_counts_{args.tag}.png")
    draw_progress(steps, fig_dir / f"step_motions_progress_{args.tag}.png")
    draw_late_vs_send(thread_rows, fig_dir / f"step_motions_late_send_{args.tag}.png")

    step_path = lab_dir / f"step_motions_{args.tag}.jsonl"
    with step_path.open("w", encoding="utf-8") as f:
        for s in steps:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    thr_path = lab_dir / f"thread_late_motions_{args.tag}.jsonl"
    with thr_path.open("w", encoding="utf-8") as f:
        for r in thread_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "n_threads": len(rows),
        "n_steps": len(steps),
        "thresholds": {k: round(v, 4) for k, v in thr.items()},
        "undetectable_at_this_scale": UNDETECTABLE,
        "priority": MOTION_ORDER[:-1],
        "step_fraction": {k: round(counts.get(k, 0) / len(steps), 4) for k in MOTION_ORDER},
        "late_thread_fraction": {
            k: round(sum(1 for r in thread_rows if r["late_motion"] == k) / len(thread_rows), 4)
            for k in MOTION_ORDER
        },
        "note": (
            "chi_low is the 20th percentile of identity-free matching cost (~0.80), "
            "not chi→0. Crystallization here is a relative dip against the Reddit background."
        ),
    }
    out_json = met_dir / f"step_motions_{args.tag}_stats.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {step_path}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
