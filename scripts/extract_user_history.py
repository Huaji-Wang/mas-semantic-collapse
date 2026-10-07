"""Pull compact posting history for 48-thread repliers from user_history.jsonl.zst.

Each line in the zst is one author with up to 300 Reddit records. This script
keeps body/selftext, subreddit, and post_id, and drops comments that belong to
the 48 demo threads so those posts never enter a persona card.

    python scripts/extract_user_history.py
    python scripts/extract_user_history.py --resume
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import sys
from pathlib import Path

import zstandard as zstd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mas_collapse.config import load_config
from mas_collapse.data.threads import linearize_comments, parse_thread

BOT_AUTHORS = {None, "", "[deleted]", "[removed]", "AutoModerator"}


def _iter_zst(path: str):
    with open(path, "rb") as f:
        reader = zstd.ZstdDecompressor().stream_reader(f)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            line = line.strip()
            if line:
                yield json.loads(line)


def _post_id(rec: dict) -> str:
    link = str(rec.get("link_id") or "")
    if link.startswith("t3_"):
        return link[3:]
    name = str(rec.get("name") or rec.get("id") or "")
    if name.startswith("t3_"):
        return name[3:]
    if rec.get("selftext") is not None or rec.get("title") is not None:
        return name
    return link


def _body(rec: dict) -> str:
    return (rec.get("body") or rec.get("selftext") or "").strip()


def collect_repliers(cfg: dict) -> tuple[set[str], dict[str, set[str]], dict[str, str]]:
    want = set(cfg["persona_demo"]["posts"])
    path = cfg["data"]["conditioned_threads_path"]
    repliers: set[str] = set()
    blocked: dict[str, set[str]] = collections.defaultdict(set)
    alias: dict[str, str] = {}
    n = 0
    for obj in _iter_zst(path):
        pid = str(obj.get("post", {}).get("id") or "")
        if pid not in want:
            continue
        th = parse_thread(obj)
        if th is None:
            continue
        n += 1
        ordered = [c for c in linearize_comments(th) if c.author not in BOT_AUTHORS]
        counts = collections.Counter(c.author for c in ordered)
        ranked = [a for a, _ in counts.most_common()]
        for i, a in enumerate(ranked, 1):
            repliers.add(a)
            blocked[a].add(pid)
            if a not in alias:
                alias[a] = f"Agent-{i}" if i <= 5 else a
        if n == len(want):
            break
    return repliers, dict(blocked), alias


def compact_records(records: list[dict], blocked: set[str]) -> list[dict]:
    out = []
    for rec in records:
        body = _body(rec)
        if not body or body in ("[deleted]", "[removed]"):
            continue
        pid = _post_id(rec)
        if pid in blocked:
            continue
        out.append(
            {
                "subreddit": str(rec.get("subreddit") or ""),
                "post_id": pid,
                "body": body,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract compact user-history records for 48-thread repliers")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--resume", action="store_true", help="skip authors already in the jsonl")
    ap.add_argument(
        "--cast-only",
        action="store_true",
        help="only the 5-person cast in cast_and_history.json, not every replier",
    )
    ap.add_argument(
        "--seed-from",
        default="",
        help="existing user_history.jsonl; reuse those authors after dropping newly blocked posts",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    src = cfg["data"].get("user_history_path") or ""
    if not src or not Path(src).exists():
        raise SystemExit(
            "USER_HISTORY_PATH missing or file not found.\n"
            "Set it in .env to user_history.jsonl-001.zst"
        )
    out_dir = Path(cfg["paths"]["personas"]) / cfg["persona_demo"]["tag"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "user_history.jsonl"

    if args.cast_only:
        raw_path = out_dir / "cast_and_history.json"
        if not raw_path.exists():
            raise SystemExit(f"missing {raw_path}; run build_persona_demo.py --no-cards first")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        repliers = set()
        blocked = collections.defaultdict(set)
        alias = {}
        for pid, thread in (raw.get("threads") or {}).items():
            for author in thread.get("cast") or []:
                repliers.add(author)
                blocked[author].add(pid)
                alias.setdefault(author, (thread.get("alias") or {}).get(author) or author)
        for author, posts in (raw.get("cast_threads") or {}).items():
            blocked[author].update(posts)
        blocked = dict(blocked)
        print(f"cast authors {len(repliers)}", flush=True)
    else:
        print("collecting repliers", flush=True)
        repliers, blocked, alias = collect_repliers(cfg)
        print(f"repliers {len(repliers)}", flush=True)

    done: set[str] = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["author"])
        print(f"resume: {len(done)} already extracted", flush=True)

    seed_from = Path(args.seed_from) if args.seed_from else None
    if seed_from and not done:
        seeded = 0
        with out_path.open("w", encoding="utf-8") as out:
            for line in seed_from.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                author = row.get("author")
                if author not in repliers:
                    continue
                blocked_posts = blocked.get(author, set())
                recs = [r for r in row.get("records") or [] if r.get("post_id") not in blocked_posts]
                row["alias"] = alias.get(author) or row.get("alias") or author
                row["blocked_posts"] = sorted(blocked_posts)
                row["records"] = recs
                row["n_kept"] = len(recs)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                done.add(author)
                seeded += 1
        print(f"seeded {seeded} authors from {seed_from}", flush=True)

    mode = "a" if done else "w"
    hit = 0
    with out_path.open(mode, encoding="utf-8") as out:
        for i, obj in enumerate(_iter_zst(src), 1):
            author = obj.get("author")
            if author not in repliers:
                continue
            if author in done:
                continue
            recs = compact_records(obj.get("records") or [], blocked.get(author, set()))
            row = {
                "author": author,
                "alias": alias.get(author) or author,
                "blocked_posts": sorted(blocked.get(author, ())),
                "n_raw": int(obj.get("n_records") or len(obj.get("records") or [])),
                "n_kept": len(recs),
                "records": recs,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            done.add(author)
            hit += 1
            if hit % 50 == 0 or hit <= 3:
                print(f"  extracted {hit}  last={author} kept={len(recs)}/{row['n_raw']}", flush=True)
            if i % 20000 == 0:
                print(f"  scanned {i} authors, extracted {hit}", flush=True)
    still = repliers - done
    print(f"wrote {out_path}")
    print(f"extracted {hit} this run, total on disk {len(done)}, not in zst {len(still)}")


if __name__ == "__main__":
    main()
