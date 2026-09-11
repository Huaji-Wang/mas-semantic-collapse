"""Stage A of the persona demo: pick each thread's cast and write their persona cards.

Two passes over the conditioned dump. The first finds the demo threads and their
cast (people who speak often enough in the thread *and* show up in other sampled
threads). The second collects everything those accounts wrote in the other
threads -- that is the only history text available, since the dump stores history
as a count. The demo thread is excluded from its own cast's history.

    python scripts/build_persona_demo.py                 # extract + write cards
    python scripts/build_persona_demo.py --no-cards      # extraction only
    python scripts/build_persona_demo.py --force         # ignore caches

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
from mas_collapse.simulate.persona import (
    CARD_SYSTEM,
    HistoryStats,
    LLMClient,
    PersonaCard,
    build_card_prompt,
    parse_card,
)

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


def extract(cfg: dict) -> dict:
    pd_cfg = cfg["persona_demo"]
    want = list(pd_cfg["posts"])
    path = cfg["data"]["conditioned_threads_path"]

    print(f"pass 1/2: locating {len(want)} demo threads", flush=True)
    n_cross = _cross_thread_counts(cfg["data"]["author_threads_csv"])

    threads: dict[str, dict] = {}
    for obj in tqdm(_iter_raw(path), desc="scan"):
        pid = str(obj.get("post", {}).get("id") or "")
        if pid not in want or pid in threads:
            continue
        th = parse_thread(obj)
        if th is None:
            raise SystemExit(f"thread {pid} has no usable comments")
        ordered = [c for c in linearize_comments(th) if c.author not in BOT_AUTHORS]
        in_thread = collections.Counter(c.author for c in ordered)
        eligible = [
            a
            for a, k in in_thread.most_common()
            if k >= int(pd_cfg["min_in_thread"]) and n_cross.get(a, 0) >= int(pd_cfg["min_cross_threads"])
        ]
        cast = eligible[: int(pd_cfg["cast_size"])]
        if len(cast) < int(pd_cfg["cast_size"]):
            raise SystemExit(f"thread {pid} only has {len(cast)} eligible cast members")
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
        threads[pid] = {
            "post_id": pid,
            "subreddit": th.subreddit,
            "title": th.title,
            "selftext": th.selftext,
            "n_comments_thread": len(ordered),
            "cast": cast,
            "alias": alias,
            "in_thread_counts": {a: in_thread[a] for a in cast},
            "cross_thread_counts": {a: n_cross.get(a, 0) for a in cast},
            "turns": turns,
        }
        if len(threads) == len(want):
            break

    missing = [p for p in want if p not in threads]
    if missing:
        raise SystemExit(f"not found in dump: {missing}")

    all_cast = {a for t in threads.values() for a in t["cast"]}
    print(f"pass 2/2: collecting cross-thread comments for {len(all_cast)} accounts", flush=True)
    history: dict[str, list[dict]] = {a: [] for a in all_cast}
    demo_ids = set(want)
    for obj in tqdm(_iter_raw(path), desc="history"):
        pid = str(obj.get("post", {}).get("id") or "")
        if pid in demo_ids:
            continue
        sub = str(obj.get("post", {}).get("subreddit") or "")
        for c in obj.get("comments") or []:
            a = c.get("author")
            if a not in history:
                continue
            body = (c.get("body") or "").strip()
            if not body or body in ("[deleted]", "[removed]"):
                continue
            history[a].append({"subreddit": sub, "post_id": pid, "body": body})

    for pid, t in threads.items():
        share = sum(t["in_thread_counts"].values()) / max(1, t["n_comments_thread"])
        print(
            f"  {pid} r/{t['subreddit']}: {t['n_comments_thread']} comments, "
            f"cast writes {len(t['turns'])} ({share:.0%})"
        )
        for a in t["cast"]:
            print(f"     {t['alias'][a]:<8} in-thread={t['in_thread_counts'][a]:<3} history={len(history[a])}")

    return {"threads": threads, "history": history}


def _spread(bodies: list[str], n: int) -> list[str]:
    """Evenly spaced by length, so the card writer sees short and long alike."""
    if len(bodies) <= n:
        return list(bodies)
    order = sorted(bodies, key=len)
    idx = [round(i * (len(order) - 1) / (n - 1)) for i in range(n)]
    return [order[i] for i in sorted(set(idx))]


def _verbatim(bodies: list[str], n: int, max_chars: int, median: int) -> list[str]:
    ranked = sorted(bodies, key=lambda b: abs(len(b) - median))
    return [b[:max_chars] for b in ranked[:n]]


def write_cards(cfg: dict, raw: dict, client: LLMClient) -> dict:
    pd_cfg = cfg["persona_demo"]
    cards: dict[str, dict] = {}
    for pid, t in raw["threads"].items():
        for author in t["cast"]:
            if author in cards:
                continue
            hist = raw["history"].get(author, [])
            bodies = [h["body"] for h in hist]
            subs = [h["subreddit"] for h in hist]
            stats = HistoryStats.from_bodies(bodies, subs)
            samples = _verbatim(
                bodies, int(pd_cfg["card_samples"]), int(pd_cfg["sample_max_chars"]), stats.len_median
            )
            card = PersonaCard(author=author, alias=t["alias"][author], stats=stats, samples=samples)
            if not bodies:
                card.error = "no cross-thread history"
                cards[author] = card.to_dict()
                continue
            prompt = build_card_prompt(
                stats, _spread(bodies, int(pd_cfg["history_max_samples"])), int(pd_cfg["history_max_chars"])
            )
            try:
                text = client.chat(
                    CARD_SYSTEM, prompt, max_tokens=400, temperature=float(pd_cfg["card_temperature"])
                )
            except Exception as exc:  # noqa: BLE001 - surface API failures in the artifact
                card.error = f"{type(exc).__name__}: {exc}"
                cards[author] = card.to_dict()
                print(f"  card FAILED {card.alias} ({author}): {card.error}")
                continue
            fields = parse_card(text)
            card.raw_card = text
            card.stance = fields.get("stance", "")
            card.tone = fields.get("tone", "")
            card.quirks = fields.get("quirks", "")
            if not fields:
                card.error = "card was not valid JSON"
            cards[author] = card.to_dict()
            flag = " [thin history]" if stats.thin else ""
            print(f"  card {card.alias:<8} {author:<28} n={stats.n_comments:<4}{flag}")
    return cards


def main() -> None:
    ap = argparse.ArgumentParser(description="Build persona-demo casts and cards")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--provider", default="deepseek", choices=["deepseek", "openai"])
    ap.add_argument("--model", default=None, help="override the provider's default model")
    ap.add_argument("--no-cards", action="store_true", help="stop after extraction")
    ap.add_argument("--force", action="store_true", help="ignore cached extraction and cards")
    args = ap.parse_args()

    cfg = load_config(args.config)
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
        return
    if cards_path.exists() and not args.force:
        print(f"reusing {cards_path}")
        return

    client = LLMClient.from_env(args.provider, args.model)
    print(f"writing cards with {client.model}")
    cards = write_cards(cfg, raw, client)
    payload = {"model": client.model, "provider": args.provider, "cards": cards}
    cards_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {cards_path}")


if __name__ == "__main__":
    main()
