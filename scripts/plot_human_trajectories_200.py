"""Human trajectory figures: keep 2x2, also export four standalone panels."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "outputs" / "labels" / "thread_labels_q80q20.jsonl"
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})

COLORS = {
    "converge": "#1f77b4",
    "non_converge": "#d62728",
    "ambiguous": "#bbbbbb",
}


def load_rows() -> list[dict]:
    rows = []
    for line in LABELS.read_text(encoding="utf-8").splitlines():
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


def draw_spaghetti(ax, rows, by, hi, lo, max_t_show=40) -> None:
    for lab in ("ambiguous", "non_converge", "converge"):
        for r in by[lab]:
            sims = r["sims_to_first"][: max_t_show + 1]
            ax.plot(
                np.arange(len(sims)),
                sims,
                color=COLORS[lab],
                alpha=0.12 if lab == "ambiguous" else 0.28,
                lw=0.8,
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
    ax.set_title("Figure 1. Human trajectories (n=200)")
    ax.set_ylim(0.15, 1.05)
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
    ax.set_ylim(0.2, 1.05)
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


def draw_hist(ax, rows, by, lo, hi) -> None:
    s_all = np.array([r["s_end"] for r in rows])
    ax.hist(s_all, bins=20, color="#7f7f7f", alpha=0.35, edgecolor="white", label="all (n=200)")
    for lab in ("non_converge", "converge"):
        ax.hist(
            [r["s_end"] for r in by[lab]],
            bins=12,
            color=COLORS[lab],
            alpha=0.55,
            edgecolor="white",
            label=lab,
        )
    ax.axvline(lo, color=COLORS["non_converge"], ls="--", lw=1.5, label=f"q20={lo:.3f}")
    ax.axvline(hi, color=COLORS["converge"], ls="--", lw=1.5, label=f"q80={hi:.3f}")
    ax.set_xlim(0, 1)
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
    rows = load_rows()
    hi = float(rows[0]["threshold_high"])
    lo = float(rows[0]["threshold_low"])
    tau_drop = float((lo + hi) / 2)
    by = {"converge": [], "non_converge": [], "ambiguous": []}
    for r in rows:
        by[r["label_A"]].append(r)
    ex_c = pick_example(by["converge"])
    ex_n = pick_example(by["non_converge"])

    # four standalone figures (reading order)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    draw_spaghetti(ax, rows, by, hi, lo)
    _finish(fig, OUT_DIR / "fig1_human_trajectories.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    draw_plateau(ax, ex_c, ex_n)
    _finish(fig, OUT_DIR / "fig2_plateau_examples.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    draw_hist(ax, rows, by, lo, hi)
    _finish(fig, OUT_DIR / "fig3_send_distribution.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    draw_depart(ax, by, tau_drop, ex_c, ex_n)
    _finish(fig, OUT_DIR / "fig4_departure_time.png")

    # keep combined 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    draw_plateau(axes[0, 0], ex_c, ex_n)
    axes[0, 0].set_title("Plateau levels (examples + α ≈ S_end)")
    draw_hist(axes[0, 1], rows, by, lo, hi)
    axes[0, 1].set_title("Distribution of late similarity (n=200)")
    draw_depart(axes[1, 0], by, tau_drop, ex_c, ex_n)
    axes[1, 0].set_title(r"Departure time $t_i$ / $t_j$ (threshold mid band)")
    draw_spaghetti(axes[1, 1], rows, by, hi, lo)
    axes[1, 1].set_title(r"Human trajectories (n=200), x = window index")
    fig.suptitle(
        "Human semantic trajectories (200-thread pilot, bge-m3, q80/q20)",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "human_trajectories_200_2x2.png", dpi=160)
    fig.savefig(OUT_DIR / "human_trajectories_200_2x2.pdf")
    plt.close(fig)
    print("wrote", OUT_DIR / "human_trajectories_200_2x2.png")


if __name__ == "__main__":
    main()
