"""Shape-based motion classes from existing sims_to_first curves.

First cut for Shiyang's trajectory-motion framing. Uses only stored
cosine-to-first curves (no re-embedding). Thresholds are fitted to the
concat-200 scale, not copied from meeting examples (D<=0.1 / 0.8 plateau).

Priority:
  1. orbit               left the first window, then came back
  2. anchored            never went far from the first window
  3. translated_plateau  dropped, then late windows flatten
  4. keep_falling        still declining at the tail
  5. other               residual (noisy / short / mixed)
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
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans SC", "Segoe UI"],
        "axes.unicode_minus": False,
    }
)

# Concat-200 shape rules (see module docstring).
ORBIT_RECOVERY = 0.12
ORBIT_RECOVERY_FRAC = 0.30
ANCHORED_MIN = 0.64
ANCHORED_SEND = 0.70
PLATEAU_LATE_SLOPE = -0.02
PLATEAU_LATE_STD = 0.045
FALLING_LATE_SLOPE = -0.025

MOTION_ORDER = [
    "anchored",
    "translated_plateau",
    "keep_falling",
    "orbit",
    "other",
]
MOTION_LABEL_ZH = {
    "anchored": "锚住",
    "translated_plateau": "搬走后停住",
    "keep_falling": "持续离开",
    "orbit": "离开又回来",
    "other": "未分",
}
MOTION_COLORS = {
    "anchored": "#1f77b4",
    "translated_plateau": "#ff7f0e",
    "keep_falling": "#d62728",
    "orbit": "#9467bd",
    "other": "#bbbbbb",
}
OLD_ORDER = ["converge", "ambiguous", "non_converge"]


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def curve_features(sims: list[float], s_end: float) -> dict | None:
    s = np.asarray(sims, dtype=float)
    if len(s) < 3:
        return None
    min_after = float(s[1:].min())
    depth = 1.0 - min_after
    recovery = float(s_end) - min_after
    late_k = min(3, len(s))
    late = s[-late_k:]
    late_std = float(late.std())
    late_slope = float(np.polyfit(np.arange(len(late)), late, 1)[0])
    t_min = int(np.argmin(s[1:]) + 1)
    return {
        "min_after": min_after,
        "depth": depth,
        "recovery": recovery,
        "late_std": late_std,
        "late_slope": late_slope,
        "t_min": t_min,
        "n_windows": int(len(s)),
    }


def classify_motion(feat: dict) -> str:
    if feat["recovery"] >= ORBIT_RECOVERY and feat["recovery"] >= ORBIT_RECOVERY_FRAC * feat["depth"]:
        return "orbit"
    if feat["min_after"] >= ANCHORED_MIN and feat["s_end"] >= ANCHORED_SEND:
        return "anchored"
    if feat["late_slope"] >= PLATEAU_LATE_SLOPE and feat["late_std"] <= PLATEAU_LATE_STD:
        return "translated_plateau"
    if feat["late_slope"] < FALLING_LATE_SLOPE:
        return "keep_falling"
    return "other"


def annotate(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        feat = curve_features(r["sims_to_first"], r["s_end"])
        if feat is None:
            motion = "other"
            feat = {
                "min_after": float("nan"),
                "depth": float("nan"),
                "recovery": float("nan"),
                "late_std": float("nan"),
                "late_slope": float("nan"),
                "t_min": None,
                "n_windows": int(r.get("n_windows") or len(r.get("sims_to_first") or [])),
            }
        else:
            feat = {**feat, "s_end": float(r["s_end"])}
            motion = classify_motion(feat)
            rec = {**r, **feat, "motion": motion, "old_label": r.get("label_A")}
            out.append(rec)
            continue
        rec = {**r, **feat, "s_end": float(r["s_end"]), "motion": motion, "old_label": r.get("label_A")}
        out.append(rec)
    return out


def _finish(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", path)


def pick_example(xs: list[dict]) -> dict:
    target_send = float(np.mean([r["s_end"] for r in xs]))
    target_rec = float(np.nanmean([r["recovery"] for r in xs]))
    xs = sorted(
        xs,
        key=lambda r: (
            abs(r["s_end"] - target_send) + 0.5 * abs((r["recovery"] or 0) - target_rec),
            abs(r["n_windows"] - 8),
        ),
    )
    return xs[0]


def draw_spaghetti(ax, rows: list[dict], max_t_show: int = 40) -> None:
    by = defaultdict(list)
    for r in rows:
        by[r["motion"]].append(r)
    rng = np.random.default_rng(42)
    for motion in MOTION_ORDER:
        xs = by[motion]
        if not xs:
            continue
        shown = xs
        if len(xs) > 50:
            idx = rng.choice(len(xs), size=50, replace=False)
            shown = [xs[i] for i in idx]
        for r in shown:
            sims = r["sims_to_first"][: max_t_show + 1]
            ax.plot(
                np.arange(len(sims)),
                sims,
                color=MOTION_COLORS[motion],
                alpha=0.18 if motion != "other" else 0.10,
                lw=0.75,
            )
    for motion in MOTION_ORDER:
        xs = [r for r in by[motion] if len(r["sims_to_first"]) >= 3]
        if not xs:
            continue
        max_len = min(max_t_show + 1, max(len(r["sims_to_first"]) for r in xs))
        mat = []
        for r in xs:
            row = np.full(max_len, np.nan)
            s = r["sims_to_first"]
            L = min(len(s), max_len)
            row[:L] = s[:L]
            mat.append(row)
        mean = np.nanmean(np.asarray(mat, dtype=float), axis=0)
        ax.plot(
            np.arange(len(mean)),
            mean,
            color=MOTION_COLORS[motion],
            lw=2.6,
            label=f"{MOTION_LABEL_ZH[motion]} mean (n={len(xs)})",
        )
    ax.set_xlabel("window index t")
    ax.set_ylabel(r"$\cos(1,t)$")
    ax.set_title("Motion shapes from existing curves (concat 200)")
    ax.set_ylim(0.20, 1.05)
    ax.set_xlim(0, max_t_show)
    ax.legend(fontsize=8, loc="lower left")


def draw_examples(axes, rows: list[dict]) -> None:
    by = defaultdict(list)
    for r in rows:
        by[r["motion"]].append(r)
    for ax, motion in zip(axes, MOTION_ORDER[:4]):
        xs = by[motion]
        if not xs:
            ax.set_title(MOTION_LABEL_ZH[motion] + " (empty)")
            ax.axis("off")
            continue
        ex = pick_example(xs)
        sims = ex["sims_to_first"]
        ax.plot(np.arange(len(sims)), sims, color=MOTION_COLORS[motion], lw=2.2)
        ax.axhline(ex["s_end"], color=MOTION_COLORS[motion], ls="--", lw=1.0, alpha=0.8)
        ax.axhline(ex["min_after"], color="0.4", ls=":", lw=1.0)
        ax.set_ylim(0.20, 1.05)
        ax.set_title(f"{MOTION_LABEL_ZH[motion]}  e.g. {ex['post_id']}")
        ax.set_xlabel("t")
        ax.set_ylabel(r"$\cos(1,t)$")
        ax.text(
            0.98,
            0.08,
            f"S_end={ex['s_end']:.2f}\nmin={ex['min_after']:.2f}\nrec={ex['recovery']:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )


def draw_crosstab(ax, rows: list[dict]) -> None:
    mat = np.zeros((len(OLD_ORDER), len(MOTION_ORDER)), dtype=int)
    for i, old in enumerate(OLD_ORDER):
        for j, motion in enumerate(MOTION_ORDER):
            mat[i, j] = sum(1 for r in rows if r["old_label"] == old and r["motion"] == motion)
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(MOTION_ORDER)))
    ax.set_xticklabels([MOTION_LABEL_ZH[m] for m in MOTION_ORDER], rotation=20, ha="right")
    ax.set_yticks(range(len(OLD_ORDER)))
    ax.set_yticklabels(OLD_ORDER)
    ax.set_title("Old q80/q20  ×  motion shape")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", color="black")
    plt.colorbar(im, ax=ax, fraction=0.046)


def draw_scatter(ax, rows: list[dict]) -> None:
    for motion in MOTION_ORDER:
        xs = [r for r in rows if r["motion"] == motion and np.isfinite(r["min_after"])]
        if not xs:
            continue
        ax.scatter(
            [r["min_after"] for r in xs],
            [r["recovery"] for r in xs],
            c=MOTION_COLORS[motion],
            s=22,
            alpha=0.75,
            label=f"{MOTION_LABEL_ZH[motion]} (n={len(xs)})",
            edgecolors="none",
        )
    ax.axvline(ANCHORED_MIN, color=MOTION_COLORS["anchored"], ls="--", lw=1, alpha=0.7)
    ax.axhline(ORBIT_RECOVERY, color=MOTION_COLORS["orbit"], ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("min cos(1,t) after window 1  (higher = never left)")
    ax.set_ylabel("recovery = S_end − min  (higher = came back)")
    ax.set_title("Why S_end mixes motions")
    ax.legend(fontsize=8, loc="upper right")


def summary_dict(rows: list[dict]) -> dict:
    counts = Counter(r["motion"] for r in rows)
    old_counts = Counter(r["old_label"] for r in rows)
    crosstab = {
        old: {m: sum(1 for r in rows if r["old_label"] == old and r["motion"] == m) for m in MOTION_ORDER}
        for old in OLD_ORDER
    }
    by_motion = {}
    for m in MOTION_ORDER:
        xs = [r for r in rows if r["motion"] == m]
        if not xs:
            by_motion[m] = {"n": 0}
            continue
        by_motion[m] = {
            "n": len(xs),
            "s_end_mean": float(np.mean([r["s_end"] for r in xs])),
            "min_after_mean": float(np.nanmean([r["min_after"] for r in xs])),
            "recovery_mean": float(np.nanmean([r["recovery"] for r in xs])),
            "late_slope_mean": float(np.nanmean([r["late_slope"] for r in xs])),
            "late_std_mean": float(np.nanmean([r["late_std"] for r in xs])),
        }
    return {
        "n": len(rows),
        "rules": {
            "orbit": f"recovery>={ORBIT_RECOVERY} and recovery>={ORBIT_RECOVERY_FRAC}*depth",
            "anchored": f"min_after>={ANCHORED_MIN} and S_end>={ANCHORED_SEND}",
            "translated_plateau": f"late_slope>={PLATEAU_LATE_SLOPE} and late_std<={PLATEAU_LATE_STD}",
            "keep_falling": f"late_slope<{FALLING_LATE_SLOPE}",
            "note": "Thresholds are for concat-200 scale only. Do not copy onto mean-pool.",
        },
        "motion_counts": dict(counts),
        "old_label_counts": dict(old_counts),
        "crosstab_old_by_motion": crosstab,
        "by_motion": by_motion,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="outputs/labels/thread_labels_q80q20.jsonl")
    ap.add_argument("--out-dir", default="outputs/figures")
    ap.add_argument("--out-labels", default="outputs/labels/thread_motions_concat200.jsonl")
    ap.add_argument("--out-summary", default="outputs/metrics/motion_shapes_concat200_summary.json")
    ap.add_argument("--prefix", default="concat200")
    args = ap.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.is_absolute():
        labels_path = ROOT / labels_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_labels = Path(args.out_labels)
    if not out_labels.is_absolute():
        out_labels = ROOT / out_labels
    out_summary = Path(args.out_summary)
    if not out_summary.is_absolute():
        out_summary = ROOT / out_summary
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    rows = annotate(load_rows(labels_path))
    with out_labels.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", out_labels)

    summary = summary_dict(rows)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out_summary)
    print(json.dumps(summary["motion_counts"], ensure_ascii=False))
    print(json.dumps(summary["crosstab_old_by_motion"], ensure_ascii=False, indent=2))

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    draw_spaghetti(ax, rows)
    _finish(fig, out_dir / f"motion_shapes_{args.prefix}_spaghetti.png")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
    draw_examples(axes.ravel(), rows)
    fig.suptitle("Example curve per motion class (concat 200)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    combo = out_dir / f"motion_shapes_{args.prefix}_examples.png"
    fig.savefig(combo, dpi=160)
    fig.savefig(combo.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", combo)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    draw_crosstab(ax, rows)
    _finish(fig, out_dir / f"motion_shapes_{args.prefix}_crosstab.png")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    draw_scatter(ax, rows)
    _finish(fig, out_dir / f"motion_shapes_{args.prefix}_scatter.png")

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.0))
    draw_spaghetti(axes[0, 0], rows)
    draw_scatter(axes[0, 1], rows)
    draw_crosstab(axes[1, 0], rows)
    draw_examples([axes[1, 1]], rows)  # unused layout; replace with counts text
    axes[1, 1].clear()
    axes[1, 1].axis("off")
    lines = ["concat 200 · 只用已有 sims_to_first", ""]
    for m in MOTION_ORDER:
        n = summary["motion_counts"].get(m, 0)
        stats = summary["by_motion"][m]
        send = stats.get("s_end_mean")
        send_s = f"{send:.2f}" if send is not None else "—"
        lines.append(f"{MOTION_LABEL_ZH[m]:<6} n={n:<3}  平均 S_end={send_s}")
    lines += [
        "",
        "旧 converge 40 里：锚住 vs 离开又回来 是两类",
        "旧 non_converge 40 里：搬走后停住 vs 持续离开 是两类",
        "切法只适用于 concat 尺度，不要套到 mean-pool",
    ]
    axes[1, 1].text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11, family="sans-serif")
    fig.suptitle("Semantic motion shapes (concat 200, bge-m3)", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    combo = out_dir / f"motion_shapes_{args.prefix}_2x2.png"
    fig.savefig(combo, dpi=160)
    fig.savefig(combo.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", combo)


if __name__ == "__main__":
    main()
