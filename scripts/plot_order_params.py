"""Figures and statistics for the five order parameters.

Reads an ``order_params_*.jsonl`` produced by ``scripts/extract_order_params.py``
and answers the question the S_end labelling could not: do sigma / d / k / chi
separate threads that a single radial score lumps together?

Example:
    python scripts/plot_order_params.py --tag pilot200
"""

from __future__ import annotations

import argparse
import json
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

PANELS = [
    ("sims_to_first", "m: cos(m_t, m_1) 对第一窗", "tab:blue"),
    ("adj_sims", "m 相邻步: cos(m_{t-1}, m_t)", "tab:cyan"),
    ("sigma", "sigma: 窗内散布", "tab:orange"),
    ("eff_dim", "d: 有效维数", "tab:green"),
    ("bimodality", "k 代理: 二分结构强度", "tab:red"),
    ("churn_config", "chi: 配置变动", "tab:purple"),
]

GRID = 20


def resample(vals: list[float], grid: int = GRID) -> np.ndarray:
    v = np.asarray([np.nan if x is None else float(x) for x in vals], dtype=np.float64)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return np.full(grid, np.nan)
    src = np.linspace(0.0, 1.0, len(v))[ok]
    dst = np.linspace(0.0, 1.0, grid)
    return np.interp(dst, src, v[ok])


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def draw_trajectories(rows: list[dict], out: Path, tag: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    x = np.linspace(0.0, 1.0, GRID)
    for ax, (key, title, color) in zip(axes.ravel(), PANELS):
        mat = np.vstack([resample(r[key]) for r in rows])
        for row in mat[:120]:
            ax.plot(x, row, color=color, alpha=0.07, linewidth=0.8)
        med = np.nanmedian(mat, axis=0)
        lo = np.nanpercentile(mat, 25, axis=0)
        hi = np.nanpercentile(mat, 75, axis=0)
        ax.fill_between(x, lo, hi, color=color, alpha=0.25, label="四分位区间")
        ax.plot(x, med, color=color, linewidth=2.4, label="中位数")
        ax.set_title(title)
        ax.set_xlabel("讨论进度（归一化）")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle(
        f"讨论云的五个序参量随时间演化（{tag}，n={len(rows)} 帖，bge-m3，m 窗宽 10 / 支撑 30）",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)


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


def load_motion_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "motion" in r and "post_id" in r:
            out[r["post_id"]] = r["motion"]
    return out


def draw_by_motion(rows: list[dict], motions: dict[str, str], out: Path, tag: str) -> dict:
    by: dict[str, list[dict]] = {k: [] for k in MOTION_ORDER}
    unmatched = 0
    for r in rows:
        m = motions.get(r["post_id"])
        if m in by:
            by[m].append(r)
        else:
            unmatched += 1
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    x = np.linspace(0.0, 1.0, GRID)
    for ax, (key, title, _) in zip(axes.ravel(), PANELS):
        for mot in MOTION_ORDER:
            group = by[mot]
            if not group:
                continue
            mat = np.vstack([resample(r[key]) for r in group])
            ax.plot(
                x,
                np.nanmedian(mat, axis=0),
                color=MOTION_COLORS[mot],
                linewidth=2.2,
                label=f"{MOTION_LABEL_ZH[mot]} n={len(group)}",
            )
        ax.set_title(title)
        ax.set_xlabel("讨论进度（归一化）")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        f"按 concat-200 的 m 轨迹形状分组：另外四个序参量是否分开（{tag}）",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)

    table: dict[str, dict] = {}
    for mot, group in by.items():
        if not group:
            continue
        def mean(k: str) -> float:
            a = np.asarray([g.get(k, np.nan) for g in group], dtype=np.float64)
            a = a[np.isfinite(a)]
            return round(float(a.mean()), 4) if a.size else float("nan")
        table[mot] = {
            "n": len(group),
            "s_end": mean("s_end"),
            "sigma_delta": mean("sigma_delta"),
            "eff_dim_delta": mean("eff_dim_delta"),
            "bimodality_late": mean("bimodality_late"),
            "churn_config_late": mean("churn_config_late"),
            "adj_sim_late": mean("adj_sim_late"),
        }
    table["_unmatched"] = unmatched
    return table


def draw_vs_send(rows: list[dict], out: Path, tag: str) -> None:
    s_end = np.asarray([r["s_end"] for r in rows], dtype=np.float64)
    keys = [
        ("sigma_delta", "sigma 变化（末段-首段）"),
        ("eff_dim_delta", "d 变化"),
        ("bimodality_delta", "二分强度变化"),
        ("adj_sim_late", "末段相邻步相似度"),
        ("churn_config_late", "末段配置变动"),
        ("bimodality_late", "末段二分强度"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    lines = []
    for ax, (key, label) in zip(axes.ravel(), keys):
        y = np.asarray([r.get(key, np.nan) for r in rows], dtype=np.float64)
        ok = np.isfinite(s_end) & np.isfinite(y)
        ax.scatter(s_end[ok], y[ok], s=16, alpha=0.55, color="tab:blue", edgecolors="none")
        if ok.sum() > 3:
            r = float(np.corrcoef(s_end[ok], y[ok])[0, 1])
            lines.append(f"{key}: r={r:+.3f} (n={int(ok.sum())})")
            ax.set_title(f"{label}   r={r:+.2f}")
        else:
            ax.set_title(label)
        ax.set_xlabel("S_end（对第一窗的末段相似度）")
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"S_end 之外的信息量：五个序参量与 S_end 的关系（{tag}，n={len(rows)}）\n"
        "相关系数接近 0 的面板，说明该量携带 S_end 没有的信息",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    for line in lines:
        print("  " + line)


def stats(rows: list[dict]) -> dict:
    def arr(key: str) -> np.ndarray:
        return np.asarray([r.get(key, np.nan) for r in rows], dtype=np.float64)

    def desc(key: str) -> dict:
        a = arr(key)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"n": 0}
        return {
            "n": int(a.size),
            "mean": round(float(a.mean()), 4),
            "sd": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 4),
            "p25": round(float(np.percentile(a, 25)), 4),
            "median": round(float(np.median(a)), 4),
            "p75": round(float(np.percentile(a, 75)), 4),
        }

    keys = [
        "s_end",
        "adj_sim_mean",
        "adj_sim_late",
        "sigma_first",
        "sigma_late",
        "sigma_delta",
        "eff_dim_first",
        "eff_dim_late",
        "eff_dim_delta",
        "k_modes_late",
        "bimodality_first",
        "bimodality_late",
        "bimodality_delta",
        "churn_config_mean",
        "churn_config_late",
    ]
    out: dict = {"n_threads": len(rows), "per_thread": {k: desc(k) for k in keys}}

    s_end = arr("s_end")
    corr = {}
    for k in keys:
        if k == "s_end":
            continue
        y = arr(k)
        ok = np.isfinite(s_end) & np.isfinite(y)
        if ok.sum() > 3 and np.std(y[ok]) > 1e-9:
            corr[k] = round(float(np.corrcoef(s_end[ok], y[ok])[0, 1]), 4)
    out["corr_with_s_end"] = corr

    all_k = [k for r in rows for k in r["k_modes"]]
    out["k_modes_window_level"] = {
        "n_windows": len(all_k),
        "frac_k_eq_1": round(float(np.mean([1.0 if k == 1 else 0.0 for k in all_k])), 4) if all_k else None,
        "max": int(max(all_k)) if all_k else None,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot200")
    ap.add_argument("--jsonl", default=None)
    ap.add_argument(
        "--motions",
        default=str(ROOT / "outputs" / "labels" / "thread_motions_concat200.jsonl"),
        help="optional concat-200 motion labels for overlay",
    )
    args = ap.parse_args()

    path = Path(args.jsonl) if args.jsonl else ROOT / "outputs" / "order_params" / f"order_params_{args.tag}.jsonl"
    rows = load_rows(path)
    if not rows:
        raise SystemExit(f"no rows in {path}")

    fig_dir = ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir = ROOT / "outputs" / "metrics"
    met_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(rows)} threads from {path.name}")
    draw_trajectories(rows, fig_dir / f"order_params_traj_{args.tag}.png", args.tag)
    print("correlations with S_end:")
    draw_vs_send(rows, fig_dir / f"order_params_vs_send_{args.tag}.png", args.tag)

    summary = stats(rows)
    motion_map = load_motion_map(Path(args.motions)) if args.motions else {}
    if motion_map:
        table = draw_by_motion(
            rows,
            motion_map,
            fig_dir / f"order_params_by_motion_{args.tag}.png",
            args.tag,
        )
        summary["by_concat200_motion"] = table
        print(f"wrote {fig_dir / f'order_params_by_motion_{args.tag}.png'}")
        print("by concat-200 motion:", json.dumps(table, ensure_ascii=False, indent=2))
    out_json = met_dir / f"order_params_{args.tag}_stats.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["k_modes_window_level"], ensure_ascii=False))
    print(f"wrote {out_json}")
    print(f"wrote {fig_dir / f'order_params_traj_{args.tag}.png'}")
    print(f"wrote {fig_dir / f'order_params_vs_send_{args.tag}.png'}")


if __name__ == "__main__":
    main()
