"""Merge chunked score jsonl files, then optionally assign q80/q20 on the pooled set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mas_collapse.config import load_config
from mas_collapse.labeling import assign_labels

DROP = ("label_A", "threshold_high", "threshold_low")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge thread score chunks")
    ap.add_argument("inputs", nargs="+", help="score/label jsonl files (later files win on post_id)")
    ap.add_argument("--out", required=True, help="merged scores jsonl (no labels unless --assign-labels)")
    ap.add_argument("--assign-labels", action="store_true")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--summary", default="", help="optional summary json path")
    args = ap.parse_args()

    by_id: dict[str, dict] = {}
    order: list[str] = []
    for raw in args.inputs:
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        for r in _load_jsonl(path):
            pid = str(r["post_id"])
            clean = {k: v for k, v in r.items() if k not in DROP}
            if pid not in by_id:
                order.append(pid)
            by_id[pid] = clean
    rows = [by_id[pid] for pid in order]

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "n_threads": len(rows),
        "n_inputs": len(args.inputs),
        "inputs": [str(p) for p in args.inputs],
        "scores_only": not args.assign_labels,
    }

    if args.assign_labels:
        cfg = load_config(args.config)
        labeled, thresholds = assign_labels(
            rows,
            high_q=float(cfg["labeling"]["high_quantile"]),
            low_q=float(cfg["labeling"]["low_quantile"]),
        )
        rows = labeled
        counts = {"converge": 0, "non_converge": 0, "ambiguous": 0}
        for r in labeled:
            counts[r["label_A"]] = counts.get(r["label_A"], 0) + 1
        summary["counts"] = counts
        summary["thresholds"] = thresholds
        summary["scores_only"] = False

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.summary:
        summary_path = Path(args.summary)
        if not summary_path.is_absolute():
            summary_path = ROOT / summary_path
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
