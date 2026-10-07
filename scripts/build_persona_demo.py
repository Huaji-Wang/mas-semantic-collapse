"""Stage A of the persona demo: pick each thread's cast and collect history.

Two passes over the conditioned dump. The first finds the demo threads and their
cast. The second collects everything those accounts wrote in other sampled
threads. DeepSeek personas are a separate step: ``scripts/write_personas.py``.

    python scripts/build_persona_demo.py --no-cards   # extract history only
    python scripts/write_personas.py                  # DeepSeek writes cards
    python scripts/build_persona_demo.py              # extract + write (legacy)

Real usernames stay in outputs/ (gitignored). Anything shown to a reader uses the
Agent-N alias.
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import sys
from pathlib import Path

import zstandard as zstd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mas_collapse.config import load_config
from mas_collapse.data.threads import linearize_comments, parse_thread
from mas_collapse.simulate.persona import LLMClient, build_cards, save_persona_outputs

BOT_AUTHORS = {None, "", "[deleted]", "[removed]", "AutoModerator"}


def _iter_raw(path: str):
    with open(path, "rb") as f:
        reader = zstd.ZstdDecompressor().stream_reader(f)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            line = line.strip()
            if line:
                yield json.loads(line)


def _cross_thread_counts(csv_path: str) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            counts[row["author"]] += 1
    return counts


def _thread_record(obj: dict, n_cross: collections.Counter, pd_cfg: dict) -> dict | None:
    th = parse_thread(obj)
    if th is None:
        return None
    ordered = [c for c in linearize_comments(th) if c.author not in BOT_AUTHORS]
    n = len(ordered)
    lo, hi = int(pd_cfg["min_thread_comments"]), int(pd_cfg["max_thread_comments"])
    if n < lo or n > hi:
        return None
    in_thread = collections.Counter(c.author for c in ordered)
    eligible = [
        a
        for a, k in in_thread.most_common()
        if k >= int(pd_cfg["min_in_thread"]) and n_cross.get(a, 0) >= int(pd_cfg["min_cross_threads"])
    ]
    if len(eligible) < int(pd_cfg["cast_size"]):
        return None
    cast = eligible[: int(pd_cfg["cast_size"])]
    alias = {a: f"Agent-{i}" for i, a in enumerate(cast, 1)}
    cast_set = set(cast)
    turns = [
        {
            "position": i,
            "comment_id": c.id,
            "author": c.author,
            "alias": alias[c.author],
            "body": c.body,
            "n_chars": len(c.body),
        }
        for i, c in enumerate(ordered)
        if c.author in cast_set
    ]
    return {
        "post_id": th.post_id,
        "subreddit": th.subreddit,
        "title": th.title,
        "selftext": th.selftext,
        "n_comments_thread": n,
        "cast": cast,
        "alias": alias,
        "in_thread_counts": {a: in_thread[a] for a in cast},
        "cross_thread_counts": {a: n_cross.get(a, 0) for a in cast},
        "turns": turns,
    }


def select_threads(cfg: dict) -> list[str]:
    """Eligible post_ids, original 3 first, then the rest in dump order."""
    pd_cfg = cfg["persona_demo"]
    preferred = ["1sql6fs", "1rd4ske", "1sp4y8z"]
    n_cross = _cross_thread_counts(cfg["data"]["author_threads_csv"])
    found: dict[str, dict] = {}
    for obj in tqdm(_iter_raw(cfg["data"]["conditioned_threads_path"]), desc="select"):
        rec = _thread_record(obj, n_cross, pd_cfg)
        if rec is None:
            continue
        found[rec["post_id"]] = rec
    ids = [p for p in preferred if p in found]
    ids += [p for p in found if p not in ids]
    print(f"eligible threads: {len(ids)} (preferred kept: {sum(p in found for p in preferred)})")
    return ids


def freeze_posts(cfg_path: Path, post_ids: list[str]) -> None:
    text = cfg_path.read_text(encoding="utf-8")
    start = text.find("  posts:\n")
    if start < 0:
        raise SystemExit("configs/default.yaml has no persona_demo posts: block")
    rest = text[start + len("  posts:\n") :]
    end_rel = rest.find("  cast_size:")
    if end_rel < 0:
        raise SystemExit("could not find cast_size after posts:")
    block = "  posts:\n" + "".join(f"    - {p}\n" for p in post_ids)
    cfg_path.write_text(text[:start] + block + rest[end_rel:], encoding="utf-8")
    print(f"froze {len(post_ids)} post_ids in {cfg_path}")


def extract(cfg: dict) -> dict:
    pd_cfg = cfg["persona_demo"]
    want = list(pd_cfg["posts"])
    want_set = set(want)
    path = cfg["data"]["conditioned_threads_path"]

    print(f"pass 1/2: locating {len(want)} locked threads", flush=True)
    n_cross = _cross_thread_counts(cfg["data"]["author_threads_csv"])

    threads: dict[str, dict] = {}
    for obj in tqdm(_iter_raw(path), desc="scan"):
        pid = str(obj.get("post", {}).get("id") or "")
        if pid not in want_set or pid in threads:
            continue
        rec = _thread_record(obj, n_cross, pd_cfg)
        if rec is None:
            raise SystemExit(f"locked thread {pid} no longer matches the filter")
        threads[pid] = rec
        if len(threads) == len(want):
            break

    missing = [p for p in want if p not in threads]
    if missing:
        raise SystemExit(f"not found in dump: {missing}")

    all_cast = {a for t in threads.values() for a in t["cast"]}
    cast_threads: dict[str, set[str]] = collections.defaultdict(set)
    for pid, t in threads.items():
        for a in t["cast"]:
            cast_threads[a].add(pid)

    print(f"pass 2/2: collecting comments for {len(all_cast)} accounts (keep other sample threads)", flush=True)
    history: dict[str, list[dict]] = {a: [] for a in all_cast}
    for obj in tqdm(_iter_raw(path), desc="history"):
        pid = str(obj.get("post", {}).get("id") or "")
        sub = str(obj.get("post", {}).get("subreddit") or "")
        for c in obj.get("comments") or []:
            a = c.get("author")
            if a not in history:
                continue
            body = (c.get("body") or "").strip()
            if not body or body in ("[deleted]", "[removed]"):
                continue
            history[a].append({"subreddit": sub, "post_id": pid, "body": body})

    for pid in want:
        t = threads[pid]
        share = sum(t["in_thread_counts"].values()) / max(1, t["n_comments_thread"])
        print(
            f"  {pid} r/{t['subreddit']}: {t['n_comments_thread']} comments, "
            f"cast writes {len(t['turns'])} ({share:.0%})"
        )
        for a in t["cast"]:
            usable = sum(1 for h in history[a] if h["post_id"] not in cast_threads[a])
            print(
                f"     {t['alias'][a]:<8} in-thread={t['in_thread_counts'][a]:<3} "
                f"history={usable}"
            )

    return {
        "threads": threads,
        "history": history,
        "cast_threads": {a: sorted(s) for a, s in cast_threads.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build persona-demo casts and cards")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--provider", default="deepseek", choices=["deepseek", "openai"])
    ap.add_argument("--model", default=None, help="override the provider's default model")
    ap.add_argument("--no-cards", action="store_true", help="stop after extraction")
    ap.add_argument("--force", action="store_true", help="ignore cached extraction and cards")
    ap.add_argument(
        "--select-and-freeze",
        action="store_true",
        help="scan the dump, write eligible post_ids into configs/default.yaml, then stop",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)

    if args.select_and_freeze:
        ids = select_threads(cfg)
        freeze_posts(cfg_path, ids)
        return

    tag = cfg["persona_demo"]["tag"]
    out_dir = Path(cfg["paths"]["personas"]) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "cast_and_history.json"
    cards_path = out_dir / "cards.json"

    if raw_path.exists() and not args.force:
        print(f"reusing {raw_path}")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        raw = extract(cfg)
        raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {raw_path}")

    if args.no_cards:
        print("history extracted. next: python scripts/write_personas.py")
        return
    if cards_path.exists() and not args.force:
        print(f"reusing {cards_path}  (python scripts/write_personas.py --force to rewrite)")
        return

    reused = {}
    reuse_from = cfg["persona_demo"].get("reuse_cards_from") or ""
    reuse_path = Path(reuse_from)
    if reuse_from and not reuse_path.is_absolute():
        reuse_path = ROOT / reuse_path
    if reuse_path.exists():
        reused = json.loads(reuse_path.read_text(encoding="utf-8")).get("cards") or {}
        print(f"will reuse cards from {reuse_path} ({len(reused)} authors)")

    client = LLMClient.from_env(args.provider, args.model)
    print(f"writing cards with {client.model}")
    cards = build_cards(raw, cfg["persona_demo"], client, reused=reused)
    payload = {"model": client.model, "provider": args.provider, "cards": cards}
    save_persona_outputs(out_dir, payload)


if __name__ == "__main__":
    main()
