"""Freeze persona-demo posts whose whole cast is in the crawled user index.

Same cast rule as build_persona_demo (length band, 5 people, >=3 comments,
>=2 indexed threads). Threads are kept only when all five have a row in
user_index.csv, so a persona can be written from the 300-record history.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mas_collapse.config import load_config
from scripts.build_persona_demo import (
    _cross_thread_counts,
    _iter_raw,
    _thread_record,
    freeze_posts,
)


def main() -> None:
    cfg_path = ROOT / "configs" / "default.yaml"
    cfg = load_config(cfg_path)
    index_path = Path(cfg["data"]["author_threads_csv"]).parent / "user_index.csv"
    indexed: set[str] = set()
    with index_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            indexed.add(row["author"])
    n_cross = _cross_thread_counts(cfg["data"]["author_threads_csv"])
    preferred = ["1sql6fs", "1rd4ske", "1sp4y8z"]
    found: dict[str, dict] = {}
    for obj in _iter_raw(cfg["data"]["conditioned_threads_path"]):
        rec = _thread_record(obj, n_cross, cfg["persona_demo"])
        if rec is None:
            continue
        if any(a not in indexed for a in rec["cast"]):
            continue
        found[rec["post_id"]] = rec
    ids = [p for p in preferred if p in found]
    ids += [p for p in found if p not in ids]
    subs = {found[p]["subreddit"] for p in ids}
    print(f"eligible with full history: {len(ids)} threads, {len(subs)} subreddits")
    freeze_posts(cfg_path, ids)


if __name__ == "__main__":
    main()
