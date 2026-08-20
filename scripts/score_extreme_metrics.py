"""Re-embed q80/q20 extremes and compute Vendi / effective dim (mean-pool comments)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from compare_separation_metrics import score_bodies
from mas_collapse.config import load_config
from mas_collapse.data.threads import iter_threads, linearize_comments
from mas_collapse.embed.backend import build_embedder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--only-label", required=True, choices=("converge", "non_converge"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["embedding"]["device"] = args.device
    cfg["embedding"]["batch_size"] = args.batch_size

    labels_path = Path(args.labels)
    if not labels_path.is_absolute():
        labels_path = ROOT / labels_path
    wanted: dict[str, dict] = {}
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("label_A") == args.only_label:
            wanted[str(r["post_id"])] = r
    print(
        f"device={args.device} batch_size={args.batch_size} only={args.only_label} n={len(wanted)}",
        flush=True,
    )

    t0 = time.perf_counter()
    embedder = build_embedder(cfg)
    window_size = int(cfg["windowing"]["window_size"])
    late_windows = int(cfg["windowing"]["late_windows"])
    sample_size = int(cfg["metrics"].get("vendi_sample_size", 30))
    repeats = int(cfg["metrics"].get("vendi_repeats", 20))
    seed = int(cfg["project"].get("seed", 42))

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_suffix(out_path.suffix + ".partial")

    scored: list[dict] = []
    pbar = tqdm(total=len(wanted), desc=args.only_label)
    for th in iter_threads(
        cfg["data"]["threads_path"],
        min_comments=int(cfg["data"]["min_comments"]),
        max_comments=cfg["data"].get("max_comments"),
    ):
        if th.post_id not in wanted:
            continue
        src = wanted[th.post_id]
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
        metrics["label_A"] = args.only_label
        metrics["title"] = th.title
        metrics["subreddit"] = th.subreddit
        metrics["s_end_from_labels"] = src.get("s_end")
        scored.append(metrics)
        pbar.update(1)
        if len(scored) % 25 == 0:
            print(
                f"scored {len(scored)}/{len(wanted)} {args.only_label} elapsed={time.perf_counter()-t0:.0f}s",
                flush=True,
            )
            partial_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in scored) + "\n",
                encoding="utf-8",
            )
        if len(scored) >= len(wanted):
            break
    pbar.close()

    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in scored) + "\n",
        encoding="utf-8",
    )
    if partial_path.exists():
        partial_path.unlink()
    print(json.dumps({"n": len(scored), "label": args.only_label, "elapsed_sec": round(time.perf_counter() - t0, 1)}, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
