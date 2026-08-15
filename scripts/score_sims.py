from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mas_collapse.config import load_config
from mas_collapse.embed.backend import build_embedder
from mas_collapse.labeling import score_comment_sequence


def main() -> None:
    ap = argparse.ArgumentParser(description="Score simulation JSON with Phase-1 metrics")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sims-dir", default=None, help="defaults to outputs/sims")
    ap.add_argument("--no-vendi", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # prefer tau from human labeling summary if present
    summary_path = Path(cfg["paths"]["labels"]) / "label_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        tau = summary.get("thresholds", {}).get("tau_suggest")
        if tau is not None:
            cfg["metrics"]["tau"] = tau

    embedder = build_embedder(cfg)
    sims_root = Path(args.sims_dir) if args.sims_dir else Path(cfg["paths"]["sims"])
    files = sorted(sims_root.rglob("*.json"))
    if not files:
        raise SystemExit(f"No simulation json under {sims_root}")

    out_path = Path(cfg["paths"]["sims"]) / "sim_metrics.jsonl"
    with out_path.open("w", encoding="utf-8") as fout:
        for fp in tqdm(files, desc="score sims"):
            data = json.loads(fp.read_text(encoding="utf-8"))
            bodies = [c["body"] for c in data.get("seed_comments", [])]
            bodies += [c["body"] for c in data.get("comments", [])]
            if len(bodies) < 2:
                continue
            metrics = score_comment_sequence(
                bodies,
                embedder,
                window_size=int(cfg["windowing"]["window_size"]),
                late_windows=int(cfg["windowing"]["late_windows"]),
                tau=cfg["metrics"].get("tau"),
                vendi_sample_size=int(cfg["metrics"].get("vendi_sample_size", 30)),
                vendi_repeats=int(cfg["metrics"].get("vendi_repeats", 20)),
                seed=int(cfg["project"].get("seed", 42)),
                compute_vendi=not args.no_vendi,
            )
            row = {
                "file": str(fp.relative_to(sims_root)),
                "post_id": data.get("post_id"),
                "human_label_A": data.get("human_label_A"),
                "protocol": data.get("protocol"),
                "structure": data.get("structure"),
                "speaking_order": data.get("speaking_order"),
                "model": data.get("model"),
                "n_bodies": len(bodies),
                "embedder": embedder.name,
                **metrics,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
