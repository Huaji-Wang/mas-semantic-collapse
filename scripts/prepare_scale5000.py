"""Mark which conditioned-dump threads can take the current cast replay.

Writes a light manifest only. Does not edit configs/default.yaml and does not
build persona cards.

    python scripts/prepare_scale5000.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mas_collapse.config import load_config
from mas_collapse.data.threads import linearize_comments, parse_thread
from scripts.build_persona_demo import BOT_AUTHORS, _cross_thread_counts, _iter_raw, _thread_record


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    scale = yaml.safe_load((ROOT / "configs" / "scale5000.yaml").read_text(encoding="utf-8"))
    pd = {
        "min_thread_comments": int(scale["min_thread_comments"]),
        "max_thread_comments": int(scale["max_thread_comments"]),
        "cast_size": int(scale["cast_size"]),
        "min_in_thread": int(scale["min_in_thread"]),
        "min_cross_threads": int(scale["min_cross_threads"]),
    }
    index_path = Path(cfg["data"]["author_threads_csv"]).parent / "user_index.csv"
    indexed: set[str] = set()
    with index_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            indexed.add(row["author"])
    n_cross = _cross_thread_counts(cfg["data"]["author_threads_csv"])

    out_dir = ROOT / "outputs" / "scale5000"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    n = n_band = n_cast = n_index = 0
    with manifest.open("w", encoding="utf-8") as fout:
        for obj in _iter_raw(cfg["data"]["conditioned_threads_path"]):
            th = parse_thread(obj)
            if th is None:
                continue
            ordered = [c for c in linearize_comments(th) if c.author not in BOT_AUTHORS]
            n_comments = len(ordered)
            n += 1
            in_band = pd["min_thread_comments"] <= n_comments <= pd["max_thread_comments"]
            if in_band:
                n_band += 1
            rec = _thread_record(obj, n_cross, pd) if in_band else None
            cast_ok = rec is not None
            if cast_ok:
                n_cast += 1
            index_ok = bool(cast_ok and all(a in indexed for a in rec["cast"]))
            if index_ok:
                n_index += 1
            fout.write(
                json.dumps(
                    {
                        "post_id": th.post_id,
                        "subreddit": th.subreddit,
                        "n_comments": n_comments,
                        "in_length_band": in_band,
                        "cast_ok": cast_ok,
                        "index_ok": index_ok,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if n % 500 == 0:
                print(f"scanned {n}", flush=True)

    summary = {
        "n_threads": n,
        "in_length_band": n_band,
        "cast_ok": n_cast,
        "index_ok": n_index,
        "rule": pd,
        "manifest": str(manifest),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
