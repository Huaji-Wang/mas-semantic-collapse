"""Human trajectory figures (standalone 1-4 + 2x2)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})

COLORS = {
    "converge": "#1f77b4",
    "non_converge": "#d62728",
    "ambiguous": "#bbbbbb",
}


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def first_drop_below(sims: list[float], thr: float) -> float | None:
    for t, s in enumerate(sims):
        if t == 0:
            continue
        if s < thr:
            return float(t)
    return None


def pick_example(xs: list[dict]) -> dict:
    target = float(np.mean([r["s_end"] for r in xs]))
    xs = sorted(xs, key=lambda r: (abs(r["s_end"] - target), abs(r["n_windows"] - 10)))
    return xs[0]


def _finish(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", path)


def _subsample(rows: list[dict], k: int, rng: np.random.Generator) -> list[dict]:
    if len(rows) <= k:
        return rows
    idx = rng.choice(len(rows), size=k, replace=False)
    return [rows[i] for i in idx]


def draw_spaghetti(ax, rows, by, hi, lo, n, max_t_show=40, seed=42) -> None:
    rng = np.random.default_rng(seed)
    shown = {
        "ambiguous": _subsample(by["ambiguous"], 200, rng),
        "non_converge": _subsample(by["non_converge"], 80, rng),
        "converge": _subsample(by["converge"], 80, rng),
    }
    for lab in ("ambiguous", "non_converge", "converge"):
        for r in shown[lab]:
            sims = r["sims_to_first"][: max_t_show + 1]
            ax.plot(
                np.arange(len(sims)),
                sims,
                color=COLORS[lab],
                alpha=0.10 if lab == "ambiguous" else 0.22,
                lw=0.7,
            )
    for lab in ("converge", "non_converge"):
        max_len = min(max_t_show + 1, max(len(r["sims_to_first"]) for r in by[lab]))
        mat = []
        for r in by[lab]:
            s = r["sims_to_first"]
            if len(s) < 3:
                continue
            row = np.full(max_len, np.nan)
            L = min(len(s), max_len)
            row[:L] = s[:L]
            mat.append(row)
        mean = np.nanmean(np.asarray(mat, dtype=float), axis=0)
        ax.plot(np.arange(len(mean)), mean, color=COLORS[lab], lw=2.6, label=f"{lab} mean")
    ax.axhline(hi, color=COLORS["converge"], ls="--", lw=1, alpha=0.7)
    ax.axhline(lo, color=COLORS["non_converge"], ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("window index t")
    ax.set_ylabel(r"$\cos(1,t)$")
    ax.set_title(f"Figure 1. Human trajectories (n={n})")
    ax.set_ylim(0.40, 1.05)
    ax.set_xlim(0, max_t_show)
    ax.legend(fontsize=8, loc="lower left")


def draw_plateau(ax, ex_c, ex_n) -> None:
    for ex, lab in [(ex_c, "converge"), (ex_n, "non_converge")]:
        sims = ex["sims_to_first"]
        ax.plot(np.arange(len(sims)), sims, color=COLORS[lab], lw=2.2, label=f"{lab} example")
        ax.axhline(ex["s_end"], color=COLORS[lab], ls="--", lw=1.2, alpha=0.85)
    ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("window index t")
    ax.set_ylabel(r"$\cos(1,t)$")
    ax.set_title(r"Figure 2. Plateau levels (examples, $\alpha \approx S_{end}$)")
    ax.set_ylim(0.40, 1.05)
    ax.legend(fontsize=9, loc="lower left")
    ax.text(
        0.98,
        0.08,
        f"α_conv≈{ex_c['s_end']:.2f}\nα_non≈{ex_n['s_end']:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def draw_hist(ax, rows, by, lo, hi, n) -> None:
    s_all = np.array([r["s_end"] for r in rows])
    ax.hist(s_all, bins=24, color="#7f7f7f", alpha=0.35, edgecolor="white", label=f"all (n={n})")
    for lab in ("non_converge", "converge"):
        ax.hist(
            [r["s_end"] for r in by[lab]],
            bins=14,
            color=COLORS[lab],
            alpha=0.55,
            edgecolor="white",
            label=lab,
        )
    ax.axvline(lo, color=COLORS["non_converge"], ls="--", lw=1.5, label=f"q20={lo:.3f}")
    ax.axvline(hi, color=COLORS["converge"], ls="--", lw=1.5, label=f"q80={hi:.3f}")
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel(r"$S_{end}$")
    ax.set_ylabel("count")
    ax.set_title("Figure 3. Distribution of late similarity")
    ax.legend(fontsize=8, loc="upper left")


def draw_depart(ax, by, tau_drop, ex_c, ex_n) -> None:
    all_t = []
    for lab in ("converge", "non_converge"):
        ts = []
        for r in by[lab]:
            td = first_drop_below(r["sims_to_first"], tau_drop)
            if td is not None:
                ts.append(td)
        all_t.extend(ts)
        if not ts:
            continue
        ax.hist(
            ts,
            bins=np.arange(0.5, max(ts) + 1.5, 1),
            color=COLORS[lab],
            alpha=0.55,
            edgecolor="white",
            label=f"{lab} (n={len(ts)})",
        )
        ex = ex_c if lab == "converge" else ex_n
        td = first_drop_below(ex["sims_to_first"], tau_drop)
        if td is not None:
            ax.axvline(td, color=COLORS[lab], ls="--", lw=1.5)
    n_never = sum(
        1
        for lab in ("converge", "non_converge")
        for r in by[lab]
        if first_drop_below(r["sims_to_first"], tau_drop) is None
    )
    ax.set_xlabel(f"first window t with cos(1,t) < {tau_drop:.2f}")
    ax.set_ylabel("count")
    ax.set_title(r"Figure 4. Departure time $t^*$")
    ax.legend(fontsize=8)
    ax.text(
        0.98,
        0.95,
        f"never dropped below thr: {n_never}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", default="outputs/figures")
    ap.add_argument("--prefix", default="mean_full")
    args = ap.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.is_absolute():
        labels_path = ROOT / labels_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(labels_path)
    n = len(rows)
    hi = float(rows[0]["threshold_high"])
    lo = float(rows[0]["threshold_low"])
    tau_drop = float((lo + hi) / 2)
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        by[r["label_A"]].append(r)
    ex_c = pick_example(by["converge"])
    ex_n = pick_example(by["non_converge"])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    draw_spaghetti(ax, rows, by, hi, lo, n)
    _finish(fig, out_dir / f"fig1_human_trajectories_{args.prefix}.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    draw_plateau(ax, ex_c, ex_n)
    _finish(fig, out_dir / f"fig2_plateau_examples_{args.prefix}.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    draw_hist(ax, rows, by, lo, hi, n)
    _finish(fig, out_dir / f"fig3_send_distribution_{args.prefix}.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    draw_depart(ax, by, tau_drop, ex_c, ex_n)
    _finish(fig, out_dir / f"fig4_departure_time_{args.prefix}.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    draw_plateau(axes[0, 0], ex_c, ex_n)
    axes[0, 0].set_title("Plateau levels (examples + α ≈ S_end)")
    draw_hist(axes[0, 1], rows, by, lo, hi, n)
    axes[0, 1].set_title(f"Distribution of late similarity (n={n})")
    draw_depart(axes[1, 0], by, tau_drop, ex_c, ex_n)
    axes[1, 0].set_title(r"Departure time $t^*$ (threshold mid band)")
    draw_spaghetti(axes[1, 1], rows, by, hi, lo, n)
    axes[1, 1].set_title(f"Human trajectories (n={n}), x = window index")
    fig.suptitle(
        f"Human semantic trajectories (n={n}, mean-pool comments, bge-m3, q80/q20)",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    combo = out_dir / f"human_trajectories_{args.prefix}_2x2.png"
    fig.savefig(combo, dpi=160)
    fig.savefig(combo.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", combo)
    print(
        json.dumps(
            {
                "n": n,
                "counts": {k: len(v) for k, v in by.items()},
                "q80": hi,
                "q20": lo,
                "tau_drop": tau_drop,
                "ex_converge": {
                    "post_id": ex_c["post_id"],
                    "s_end": ex_c["s_end"],
                    "n_windows": ex_c["n_windows"],
                    "title": ex_c.get("title", ""),
                },
                "ex_non": {
                    "post_id": ex_n["post_id"],
                    "s_end": ex_n["s_end"],
                    "n_windows": ex_n["n_windows"],
                    "title": ex_n.get("title", ""),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
