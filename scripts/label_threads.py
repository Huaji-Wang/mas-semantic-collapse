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

from mas_collapse.config import load_config
from mas_collapse.data.threads import iter_threads
from mas_collapse.embed.backend import build_embedder
from mas_collapse.labeling import assign_labels, score_thread


def _strip_labels(row: dict) -> dict:
    out = dict(row)
    for k in ("label_A", "threshold_high", "threshold_low"):
        out.pop(k, None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Score / label Reddit threads")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=0, help="max threads to score after --skip")
    ap.add_argument("--skip", type=int, default=0, help="skip this many eligible threads first")
    ap.add_argument("--no-vendi", action="store_true", help="skip utterance-level Vendi (faster)")
    ap.add_argument(
        "--tag",
        default="",
        help="output suffix: thread_scores_<tag>.jsonl or thread_labels_<tag>.jsonl",
    )
    ap.add_argument("--batch-size", type=int, default=None, help="override embedding.batch_size")
    ap.add_argument("--device", default=None, help="override embedding.device (cpu|cuda)")
    ap.add_argument(
        "--scores-only",
        action="store_true",
        help="write per-thread scores without q80/q20 labels (for chunked runs)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.batch_size:
        cfg["embedding"]["batch_size"] = args.batch_size
    if args.device:
        cfg["embedding"]["device"] = args.device
    print(
        f"device={cfg['embedding'].get('device')} batch_size={cfg['embedding'].get('batch_size')} "
        f"skip={args.skip} limit={args.limit or 'all'} scores_only={args.scores_only}",
        flush=True,
    )
    t0 = time.perf_counter()
    embedder = build_embedder(cfg)

    max_comments = cfg["data"].get("max_comments")
    rows = []
    thread_iter = iter_threads(
        cfg["data"]["threads_path"],
        cfg["data"]["min_comments"],
        max_comments=max_comments,
    )
    out_dir = Path(cfg["paths"]["labels"])
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = "thread_scores" if args.scores_only else "thread_labels"
    stem = f"{kind}_{args.tag}" if args.tag else kind
    partial_path = out_dir / f"{stem}.partial.jsonl"

    skipped = 0
    pbar = tqdm(total=(args.limit or None), desc="score")
    for th in thread_iter:
        if skipped < args.skip:
            skipped += 1
            continue
        rows.append(_strip_labels(score_thread(th, embedder, cfg, compute_vendi=not args.no_vendi)))
        pbar.update(1)
        if len(rows) % 25 == 0:
            cap = args.limit or "?"
            print(
                f"scored {len(rows)}/{cap} skip={args.skip} elapsed={time.perf_counter() - t0:.0f}s",
                flush=True,
            )
            with partial_path.open("w", encoding="utf-8") as pf:
                for r in rows:
                    pf.write(json.dumps(r, ensure_ascii=False) + "\n")
        if args.limit and len(rows) >= args.limit:
            break
    pbar.close()

    elapsed_sec = time.perf_counter() - t0
    summary = {
        "n_threads": len(rows),
        "skip": args.skip,
        "limit": args.limit or None,
        "scores_only": bool(args.scores_only),
        "embedder": embedder.name,
        "device": cfg["embedding"].get("device"),
        "batch_size": cfg["embedding"].get("batch_size"),
        "no_vendi": bool(args.no_vendi),
        "elapsed_sec": round(elapsed_sec, 1),
        "window_size": cfg["windowing"]["window_size"],
        "window_pool": cfg.get("windowing", {}).get("window_pool", "mean_comments"),
        "min_comments": cfg["data"]["min_comments"],
        "max_comments": max_comments,
    }

    if args.scores_only:
        detail_path = out_dir / f"{stem}.jsonl"
        summary_path = out_dir / (f"score_summary_{args.tag}.json" if args.tag else "score_summary.json")
        with detail_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary["note"] = "Scores only. Merge chunks then assign q80/q20 on the pooled set."
    else:
        labeled, thresholds = assign_labels(
            rows,
            high_q=float(cfg["labeling"]["high_quantile"]),
            low_q=float(cfg["labeling"]["low_quantile"]),
        )
        detail_path = out_dir / f"{stem}.jsonl"
        summary_path = out_dir / (f"label_summary_{args.tag}.json" if args.tag else "label_summary.json")
        with detail_path.open("w", encoding="utf-8") as f:
            for r in labeled:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts = {"converge": 0, "non_converge": 0, "ambiguous": 0}
        for r in labeled:
            counts[r["label_A"]] = counts.get(r["label_A"], 0) + 1
        summary["counts"] = counts
        summary["thresholds"] = thresholds
        summary["quantile_scheme"] = (
            f"q{int(float(cfg['labeling']['high_quantile'])*100)}"
            f"/q{int(float(cfg['labeling']['low_quantile'])*100)}"
        )
        summary["note"] = "Set metrics.tau to thresholds.tau_suggest for T_tau on later runs"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if partial_path.exists():
        partial_path.unlink()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {detail_path}")
    print(f"elapsed_sec {elapsed_sec:.1f}")


if __name__ == "__main__":
    main()
