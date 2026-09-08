"""Run the three-stage pipeline from a chosen stage.

    python scripts/run_pipeline.py --from 3 --tag chik200
    python scripts/run_pipeline.py --from 3 --tag chik200 --title-backend openai
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.check_call(args, cwd=str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="Comment-cloud pipeline, then title clusters vs motion")
    ap.add_argument("--from", dest="start", choices=("1", "2", "3"), default="3")
    ap.add_argument("--tag", default="chik200")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--title-backend", default=None, help="stage 3 only: bge_m3 | openai | hashing")
    ap.add_argument("--cluster-method", default="kmeans", choices=("kmeans", "average_linkage"))
    ap.add_argument("--min-cluster-size", type=int, default=15)
    ap.add_argument("--limit", type=int, default=200, help="stage 1 thread cap")
    ap.add_argument("--device", default="cpu", help="stage 1 embedding device")
    ap.add_argument("--allow-laptop-cuda", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    start = int(args.start)

    if start <= 1:
        cmd = [
            py,
            str(ROOT / "scripts" / "extract_order_params.py"),
            "--config",
            args.config,
            "--tag",
            args.tag,
            "--limit",
            str(args.limit),
            "--device",
            args.device,
            "--cluster-distance",
            "0.55",
        ]
        if args.allow_laptop_cuda:
            cmd.append("--allow-laptop-cuda")
        _run(cmd)

    if start <= 2:
        _run(
            [
                py,
                str(ROOT / "scripts" / "classify_thread_motions.py"),
                "--tag",
                args.tag,
            ]
        )

    if start <= 3:
        cmd = [
            py,
            str(ROOT / "scripts" / "cluster_titles.py"),
            "--config",
            args.config,
            "--tag",
            args.tag,
        ]
        if args.title_backend:
            cmd.extend(["--title-backend", args.title_backend])
        cmd.extend(
            [
                "--cluster-method",
                args.cluster_method,
                "--min-cluster-size",
                str(args.min_cluster_size),
            ]
        )
        _run(cmd)


if __name__ == "__main__":
    main()
