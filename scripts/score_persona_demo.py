"""Stage C of the persona demo: five order parameters for both lines, then motion labels.

Human and sim are the same cast, same positions, same count, so the two
trajectories are directly comparable window by window. Motion scores are put on
the 200-thread yardstick (``--scale-from``) rather than on six threads, so the
class boundaries are not redefined by the demo itself.

    python scripts/score_persona_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from mas_collapse.config import load_config
from mas_collapse.embed.backend import build_embedder
from mas_collapse.metrics.order_params import extract_order_params, summarize_order_params
from scripts.classify_thread_motions import MOTION_ZH, _scales, classify, features


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the persona demo on the five order parameters")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--window-size", type=int, default=None)
    ap.add_argument("--support-size", type=int, default=30)
    ap.add_argument("--cluster-distance", type=float, default=0.55)
    ap.add_argument(
        "--scale-from",
        default="outputs/order_params/order_params_chik200.jsonl",
        help="population whose feature spread sets the motion-score scales",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["embedding"]["device"] = args.device
    if args.backend:
        cfg["embedding"]["backend"] = args.backend
    window_size = args.window_size or int(cfg["windowing"]["window_size"])
    support_size = max(window_size, int(args.support_size))

    tag = cfg["persona_demo"]["tag"]
    sims_dir = Path(cfg["paths"]["sims"]) / tag
    files = sorted(sims_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no simulations in {sims_dir}; run scripts/run_persona_demo.py first")

    embedder = build_embedder(cfg)
    rows: list[dict] = []
    for path in files:
        blob = json.loads(path.read_text(encoding="utf-8"))
        sides = {
            "human": [t["body"] for t in blob["human"]][: len(blob["generated"])],
            "sim": [g["body"] for g in blob["generated"]],
        }
        for side, bodies in sides.items():
            bodies = [b for b in bodies if b.strip()]
            if len(bodies) < window_size:
                print(f"skip {blob['post_id']}/{side}: only {len(bodies)} comments")
                continue
            embs = embedder.embed(bodies)
            traj = extract_order_params(
                embs,
                window_size=window_size,
                support_size=support_size,
                cluster_distance=float(args.cluster_distance),
            )
            rows.append(
                {
                    "post_id": f"{blob['post_id']}:{side}",
                    "thread_id": blob["post_id"],
                    "side": side,
                    "subreddit": blob["subreddit"],
                    "title": blob["title"],
                    "n_comments": len(bodies),
                    "mean_chars": round(float(np.mean([len(b) for b in bodies])), 1),
                    "embedder": embedder.name,
                    **traj,
                    **summarize_order_params(traj, late_windows=int(cfg["windowing"]["late_windows"])),
                }
            )
            print(f"  {blob['post_id']}/{side}: {len(bodies)} comments, {traj['n_windows']} windows")

    out_dir = Path(cfg["paths"]["outputs"]) / "order_params"
    out_dir.mkdir(parents=True, exist_ok=True)
    op_path = out_dir / f"order_params_{tag}.jsonl"
    with op_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {op_path}")

    scale_path = ROOT / args.scale_from
    if scale_path.exists():
        pop = [json.loads(x) for x in scale_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        sc = _scales([features(r) for r in pop])
        scale_src = str(scale_path.name)
    else:
        sc = _scales([features(r) for r in rows])
        scale_src = "demo rows only (fallback)"
    print(f"motion scales from {scale_src}")

    labeled = []
    for r in rows:
        f = features(r)
        lab, scores = classify(f, sc)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        labeled.append(
            {
                "post_id": r["post_id"],
                "thread_id": r["thread_id"],
                "side": r["side"],
                "subreddit": r["subreddit"],
                "n_comments": r["n_comments"],
                "n_windows": r["n_windows"],
                "s_end": r.get("s_end"),
                "motion": lab,
                "margin": round(float(ranked[0][1] - ranked[1][1]), 4),
                "runner_up": ranked[1][0],
                "scores": {k: round(float(v), 4) for k, v in scores.items()},
            }
        )

    lab_dir = Path(cfg["paths"]["labels"])
    lab_dir.mkdir(parents=True, exist_ok=True)
    lab_path = lab_dir / f"thread_motions_{tag}.jsonl"
    with lab_path.open("w", encoding="utf-8") as f:
        for row in labeled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {lab_path}")

    by_thread: dict[str, dict[str, dict]] = {}
    for x in labeled:
        by_thread.setdefault(x["thread_id"], {})[x["side"]] = x
    print("\nthread              human            sim              same?")
    for tid, pair in by_thread.items():
        h, s = pair.get("human"), pair.get("sim")
        if not h or not s:
            continue
        same = "yes" if h["motion"] == s["motion"] else "no"
        print(
            f"  {tid:<10} {MOTION_ZH[h['motion']]:<6}({h['s_end']:.2f})   "
            f"{MOTION_ZH[s['motion']]:<6}({s['s_end']:.2f})   {same}"
        )


if __name__ == "__main__":
    main()
