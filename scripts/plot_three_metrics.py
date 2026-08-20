"""Three separate metric figures: human converge vs non_converge, optional AI overlay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mas_collapse.metrics.semantic import average_converge_distance

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


def _enrich(row: dict) -> dict:
    r = dict(row)
    if r.get("avg_converge_distance") is None and r.get("sims_to_first"):
        r["avg_converge_distance"] = average_converge_distance(r["sims_to_first"])
    return r


def load_human(path: Path | None = None) -> tuple[list[dict], list[dict]]:
    path = path or (ROOT / "outputs" / "metrics" / "human_separation_80.jsonl")
    conv, non = [], []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        r = _enrich(json.loads(line))
        if r["label_A"] == "converge":
            conv.append(r)
        elif r["label_A"] == "non_converge":
            non.append(r)
    return conv, non


def load_ai() -> list[dict]:
    summary_path = ROOT / "outputs" / "metrics" / "separation_summary.json"
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
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

    ax.scatter(_jitter(len(yc), rng, 1.0), yc, s=18, c=BLUE, alpha=0.7, zorder=3, label=f"Human converge (n={len(yc)})")
    ax.scatter(_jitter(len(yn), rng, 2.0), yn, s=18, c=RED, alpha=0.7, zorder=3, label=f"Human non-converge (n={len(yn)})")

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
    ax.set_xlabel(f"Human thread class (q80 / q20, n_conv={len(yc)}, n_non={len(yn)})")
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


def _has_metric(rows: list[dict], key: str) -> bool:
    vals = [r.get(key) for r in rows]
    return any(v is not None and not (isinstance(v, float) and np.isnan(v)) for v in vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="", help="labeled jsonl; default is the old 80-thread metric file")
    ap.add_argument("--out-dir", default="outputs/figures")
    ap.add_argument("--prefix", default="", help="filename suffix, e.g. mean650")
    ap.add_argument("--no-ai", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.prefix}" if args.prefix else ""

    if args.labels:
        labels_path = Path(args.labels)
        if not labels_path.is_absolute():
            labels_path = ROOT / labels_path
        conv, non = load_human(labels_path)
    else:
        conv, non = load_human()
    ai = [] if args.no_ai else load_ai()
    specs = [
        {
            "key": "avg_converge_distance",
            "ylabel": "Average converge distance  (1 − cos(1,t))",
            "title": "Figure 5. Average converge distance",
            "file": f"fig5_avg_converge_distance{suffix}.png",
        },
        {
            "key": "vendi_late",
            "ylabel": "Late-window Vendi score",
            "title": "Figure 6. Late Vendi score",
            "file": f"fig6_vendi_late{suffix}.png",
        },
        {
            "key": "effective_dim",
            "ylabel": "Effective dimensionality (participation ratio)",
            "title": "Figure 7. Effective dimension",
            "file": f"fig7_effective_dim{suffix}.png",
        },
    ]
    for s in specs:
        if not _has_metric(conv + non, s["key"]):
            print(f"skip {s['file']}: metric {s['key']} not present")
            continue
        plot_one(
            key=s["key"],
            ylabel=s["ylabel"],
            title=s["title"],
            outfile=out_dir / s["file"],
            conv=conv,
            non=non,
            ai=ai,
        )


if __name__ == "__main__":
    main()
