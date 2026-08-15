"""Three separate metric figures: human converge vs non_converge, AI overlay."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 140,
})
# Windows Chinese labels if available
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#1f77b4"
RED = "#d62728"


def load_human() -> tuple[list[dict], list[dict]]:
    path = ROOT / "outputs" / "metrics" / "human_separation_80.jsonl"
    conv, non = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["label_A"] == "converge":
            conv.append(r)
        elif r["label_A"] == "non_converge":
            non.append(r)
    return conv, non


def load_ai() -> list[dict]:
    summary = json.loads((ROOT / "outputs" / "metrics" / "separation_summary.json").read_text(encoding="utf-8"))
    return summary.get("ai_sims", [])


def _jitter(n: int, rng: np.random.Generator, center: float, width: float = 0.08) -> np.ndarray:
    return center + rng.uniform(-width, width, size=n)


def plot_one(
    *,
    key: str,
    ylabel: str,
    title: str,
    outfile: Path,
    conv: list[dict],
    non: list[dict],
    ai: list[dict],
) -> None:
    rng = np.random.default_rng(42)
    yc = np.array([r[key] for r in conv], dtype=float)
    yn = np.array([r[key] for r in non], dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    data = [yc, yn]
    parts = ax.violinplot(data, positions=[1, 2], showmeans=False, showmedians=True, widths=0.7)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(BLUE if i == 0 else RED)
        body.set_alpha(0.35)
        body.set_edgecolor(BLUE if i == 0 else RED)
    for k in ("cbars", "cmins", "cmaxes", "cmedians"):
        if k in parts:
            parts[k].set_color("0.3")

    ax.scatter(_jitter(len(yc), rng, 1.0), yc, s=18, c=BLUE, alpha=0.7, zorder=3, label="Human converge (n=40)")
    ax.scatter(_jitter(len(yn), rng, 2.0), yn, s=18, c=RED, alpha=0.7, zorder=3, label="Human non-converge (n=40)")

    # class means
    ax.scatter([1], [yc.mean()], s=80, c=BLUE, marker="D", zorder=4, edgecolors="white", linewidths=0.6)
    ax.scatter([2], [yn.mean()], s=80, c=RED, marker="D", zorder=4, edgecolors="white", linewidths=0.6)

    markers = {"converge": "P", "non_converge": "X"}
    for row in ai:
        lab = row.get("human_label_A")
        x = 1 if lab == "converge" else 2
        ax.scatter(
            [x],
            [row[key]],
            s=140,
            c="black",
            marker=markers.get(lab, "*"),
            zorder=5,
            label=f"AI on {lab} seed",
        )

    ax.set_xticks([1, 2], ["converge", "non-converge"])
    ax.set_xlabel("Human thread class (q80 / q20 on 200-thread pilot)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # de-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        h2.append(h)
        l2.append(l)
    ax.legend(h2, l2, fontsize=8, loc="best", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(outfile, dpi=160)
    fig.savefig(outfile.with_suffix(".pdf"))
    plt.close(fig)
    print("wrote", outfile)


def main() -> None:
    conv, non = load_human()
    ai = load_ai()
    specs = [
        {
            "key": "avg_converge_distance",
            "ylabel": "Average converge distance  (1 − cos(1,t))",
            "title": "Figure 5. Average converge distance",
            "file": "fig5_avg_converge_distance.png",
        },
        {
            "key": "vendi_late",
            "ylabel": "Late-window Vendi score",
            "title": "Figure 6. Late Vendi score",
            "file": "fig6_vendi_late.png",
        },
        {
            "key": "effective_dim",
            "ylabel": "Effective dimensionality (participation ratio)",
            "title": "Figure 7. Effective dimension",
            "file": "fig7_effective_dim.png",
        },
    ]
    for s in specs:
        plot_one(
            key=s["key"],
            ylabel=s["ylabel"],
            title=s["title"],
            outfile=OUT / s["file"],
            conv=conv,
            non=non,
            ai=ai,
        )


if __name__ == "__main__":
    main()
