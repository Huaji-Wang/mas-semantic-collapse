"""Extract the five discussion-cloud order parameters per thread.

Embeds every comment once (same cost as the existing mean-pool run) and keeps the
point cloud long enough to estimate sigma / d / k / chi before pooling it into m.

Examples
--------
Fast code smoke (hashing backend, not for reported numbers):
    python scripts/extract_order_params.py --backend hashing --limit 15 --tag smoke

Pilot on CPU:
    python scripts/extract_order_params.py --limit 200 --tag pilot200

Resume after a crash (CPU only on the display laptop):
    python scripts/extract_order_params.py --skip N --limit 200 --device cpu --tag chik200

Do not use --device cuda on the RTX 4070 laptop that also drives the screen:
it has hard-powered-off after ~15 minutes. CUDA only on a separate compute box,
and only with --allow-laptop-cuda if you insist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mas_collapse.config import load_config
from mas_collapse.data.threads import iter_threads, linearize_comments
from mas_collapse.embed.backend import build_embedder
from mas_collapse.metrics.order_params import extract_order_params, summarize_order_params


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract m / sigma / d / k / chi trajectories")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=0, help="max threads after --skip")
    ap.add_argument("--skip", type=int, default=0, help="skip this many eligible threads first")
    ap.add_argument("--tag", default="", help="output suffix")
    ap.add_argument("--device", default=None, help="override embedding.device (cpu|cuda)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--backend", default=None, help="override embedding.backend (bge_m3|hashing|openai)")
    ap.add_argument("--window-size", type=int, default=None, help="m grid, default from config")
    ap.add_argument("--support-size", type=int, default=30, help="points supporting sigma/d/k")
    ap.add_argument("--cluster-distance", type=float, default=0.55, help="cosine-distance cutoff for k")
    ap.add_argument("--checkpoint-every", type=int, default=1, help="fsync every N finished threads")
    ap.add_argument(
        "--allow-laptop-cuda",
        action="store_true",
        help="required together with --device cuda; this display GPU has shut the machine down",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["embedding"]["device"] = args.device
    if args.batch_size:
        cfg["embedding"]["batch_size"] = args.batch_size
    if args.backend:
        cfg["embedding"]["backend"] = args.backend
    window_size = args.window_size or int(cfg["windowing"]["window_size"])
    support_size = max(window_size, int(args.support_size))

    if str(cfg["embedding"].get("device", "cpu")).lower() == "cuda" and not args.allow_laptop_cuda:
        raise SystemExit(
            "Refusing --device cuda on this machine. The RTX 4070 also drives the "
            "display and has hard-powered-off after ~15 minutes of embedding. "
            "Run with --device cpu, or pass --allow-laptop-cuda only on a "
            "separate compute GPU that does not drive the screen."
        )

    print(
        f"backend={cfg['embedding'].get('backend')} device={cfg['embedding'].get('device')} "
        f"batch_size={cfg['embedding'].get('batch_size')} window={window_size} support={support_size} "
        f"skip={args.skip} limit={args.limit or 'all'} cluster_distance={args.cluster_distance}",
        flush=True,
    )

    t0 = time.perf_counter()
    embedder = build_embedder(cfg)

    out_dir = Path(cfg["paths"]["outputs"]) / "order_params"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"order_params_{args.tag}" if args.tag else "order_params"
    detail_path = out_dir / f"{stem}.jsonl"
    partial_path = out_dir / f"{stem}.partial.jsonl"
    summary_path = out_dir / f"{stem}_summary.json"

    def append_row(path: Path, record: dict) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def rewrite(path: Path, records: list[dict]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)

    rows: list[dict] = []
    skipped = 0
    thread_iter = iter_threads(
        cfg["data"]["threads_path"],
        cfg["data"]["min_comments"],
        max_comments=cfg["data"].get("max_comments"),
    )
    pbar = tqdm(total=(args.limit or None), desc="threads")
    for th in thread_iter:
        if skipped < args.skip:
            skipped += 1
            continue
        comments = linearize_comments(th)
        bodies = [c.body for c in comments]
        if len(bodies) < window_size:
            continue
        embs = embedder.embed(bodies)
        traj = extract_order_params(
            embs,
            window_size=window_size,
            support_size=support_size,
            cluster_distance=float(args.cluster_distance),
        )
        row = {
            "post_id": th.post_id,
            "subreddit": th.subreddit,
            "title": th.title,
            "n_comments": len(bodies),
            "embedder": embedder.name,
            **traj,
            **summarize_order_params(traj, late_windows=int(cfg["windowing"]["late_windows"])),
        }
        rows.append(row)
        append_row(partial_path, row)
        pbar.update(1)

        if len(rows) % args.checkpoint_every == 0:
            elapsed = time.perf_counter() - t0
            per = elapsed / len(rows)
            remaining = (args.limit - len(rows)) * per if args.limit else 0.0
            print(
                f"{len(rows)}/{args.limit or '?'} elapsed={elapsed:.0f}s "
                f"per_thread={per:.1f}s eta={remaining / 60:.1f}min",
                flush=True,
            )

        if args.limit and len(rows) >= args.limit:
            break
    pbar.close()

    rewrite(detail_path, rows)
    elapsed = time.perf_counter() - t0
    summary = {
        "n_threads": len(rows),
        "skip": args.skip,
        "limit": args.limit or None,
        "embedder": embedder.name,
        "backend": cfg["embedding"].get("backend"),
        "device": cfg["embedding"].get("device"),
        "batch_size": cfg["embedding"].get("batch_size"),
        "window_size": window_size,
        "cluster_distance": float(args.cluster_distance),
        "min_comments": cfg["data"]["min_comments"],
        "max_comments": cfg["data"].get("max_comments"),
        "elapsed_sec": round(elapsed, 1),
        "note": (
            "m on non-overlapping windows; sigma/d/k on centred support windows; "
            "official chi is angular occupancy vs the first window (time vs chi; "
            "wave = loop in high-D). churn_config is the old 1024-D Hungarian cost. "
            "Comment vectors are not persisted."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if partial_path.exists():
        partial_path.unlink()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {detail_path}")


if __name__ == "__main__":
    main()
