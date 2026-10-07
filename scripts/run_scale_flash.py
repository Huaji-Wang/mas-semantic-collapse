"""Full-dump replay for one DeepSeek V4.1 Flash arm.

Protocol: post title and body, a short persona, the latest 3 generated
comments plus 2 earlier ones drawn from a fixed seed. Reply cap is 128
tokens. Thinking is off. Every linearized comment position is kept.

The account on this machine can call the official DeepSeek API. There is
no OpenRouter key, so this arm is the official endpoint, not InferenceNet.
A 402 or a balance under 1 yuan stops the run. Completed threads stay put.

    python scripts/run_scale_flash.py prepare
    python scripts/run_scale_flash.py run --threads 1 --limit 2
    python scripts/run_scale_flash.py run --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import APIConnectionError, RateLimitError

from mas_collapse.config import load_config
from mas_collapse.data.threads import linearize_comments, parse_thread
from mas_collapse.simulate.persona import BillingError, LLMClient, agent_user_prompt, visible_history_indices
from scripts.build_persona_demo import BOT_AUTHORS, _iter_raw

RECENT = 3
EARLIER = 2
SEED = 42
MAX_TOKENS = 128
TARGET_CHARS_CAP = 200
TEMPERATURE = 0.9
ARM = "low_deepseek"


def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0].strip()
    return cut or text[:n].strip()


def _median(vals: list[int]) -> int:
    if not vals:
        return 0
    ordered = sorted(vals)
    return int(ordered[len(ordered) // 2])


def prepare(cfg: dict) -> None:
    out = ROOT / "outputs" / "scale5000"
    out.mkdir(parents=True, exist_ok=True)
    replay_path = out / "replay.jsonl"
    pack_path = out / "author_pack.jsonl"
    tmp = out / "replay.jsonl.tmp"
    authors: dict[str, dict] = {}
    n_threads = n_comments = 0
    with tmp.open("w", encoding="utf-8") as fout:
        for obj in _iter_raw(cfg["data"]["conditioned_threads_path"]):
            th = parse_thread(obj)
            if th is None:
                continue
            ordered = [c for c in linearize_comments(th) if c.author not in BOT_AUTHORS]
            if not ordered:
                continue
            alias: dict[str, str] = {}
            turns = []
            for i, c in enumerate(ordered):
                if c.author not in alias:
                    alias[c.author] = f"Agent-{len(alias) + 1}"
                pack = authors.get(c.author)
                if pack is None:
                    pack = {"lengths": [], "samples": []}
                    authors[c.author] = pack
                pack["lengths"].append(len(c.body))
                posts = {s["post_id"] for s in pack["samples"]}
                if th.post_id not in posts and len(pack["samples"]) < 6:
                    pack["samples"].append({"post_id": th.post_id, "body": _clip(c.body, 160)})
                turns.append(
                    {
                        "position": i,
                        "comment_id": c.id,
                        "author": c.author,
                        "alias": alias[c.author],
                        "body": c.body,
                        "n_chars": len(c.body),
                    }
                )
            rec = {
                "post_id": th.post_id,
                "subreddit": th.subreddit,
                "title": th.title,
                "selftext": th.selftext,
                "cast": list(alias),
                "alias": alias,
                "turns": turns,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_threads += 1
            n_comments += len(turns)
            if n_threads % 500 == 0:
                print(f"prepared {n_threads} threads, {n_comments} comments", flush=True)
    tmp.replace(replay_path)
    with pack_path.open("w", encoding="utf-8") as fout:
        for author, pack in authors.items():
            fout.write(
                json.dumps({"author": author, "lengths": pack["lengths"], "samples": pack["samples"]}, ensure_ascii=False)
                + "\n"
            )
    summary = {
        "threads": n_threads,
        "comments": n_comments,
        "authors": len(authors),
        "replay": str(replay_path),
        "author_pack": str(pack_path),
    }
    (out / "replay_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def _load_llm_cards() -> dict[str, dict]:
    cards: dict[str, dict] = {}
    for tag in ("persona_148", "persona_lenfree"):
        path = ROOT / "outputs" / "personas" / tag / "cards.json"
        if not path.exists():
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        for author, card in (blob.get("cards") or {}).items():
            if card.get("error") and not (card.get("stance") or card.get("tone") or card.get("quirks")):
                continue
            cards[author] = card
    return cards


def _load_packs(path: Path) -> dict[str, dict]:
    packs = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                packs[row["author"]] = row
    return packs


def _persona(author: str, post_id: str, packs: dict, cards: dict) -> tuple[str, str, int]:
    card = cards.get(author)
    pack = packs.get(author) or {"lengths": [], "samples": []}
    lines = ["Who you are, based on your own posting history:"]
    source = "none"
    median = _median(pack.get("lengths") or [])
    if card and (card.get("stance") or card.get("tone") or card.get("quirks")):
        source = "llm_card"
        if card.get("stance"):
            lines.append("- Positions you keep taking: " + _clip(card["stance"], 220))
        if card.get("tone"):
            lines.append("- How you sound: " + _clip(card["tone"], 160))
        if card.get("quirks"):
            lines.append("- Verbal habits: " + _clip(card["quirks"], 160))
        median = int((card.get("stats") or {}).get("len_median") or median)
        samples = card.get("samples") or []
        if samples:
            lines.append(f'- One thing you have written: "{_clip(samples[0], 120)}"')
    else:
        samples = [s for s in pack.get("samples") or [] if s.get("post_id") != post_id and s.get("body")]
        if samples:
            source = "dump_quote"
            lines.append(f'- One thing you have written: "{_clip(samples[0]["body"], 120)}"')
        elif median:
            source = "length_only"
    if median:
        aim = min(median, TARGET_CHARS_CAP)
        lines.append(f"- Length: typically about {median} characters. Aim for {aim}, and finish the sentence.")
    else:
        aim = TARGET_CHARS_CAP
        lines.append(f"- Length: aim for about {aim} characters, and finish the sentence.")
    lines += ["", "Write one comment in this thread, in your own voice.", "Output only the comment text."]
    return "\n".join(lines), source, aim


def _balance_cny() -> float | None:
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        return None
    req = urllib.request.Request(
        "https://api.deepseek.com/user/balance",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    infos = data.get("balance_infos") or []
    if not infos:
        return None
    try:
        return float(infos[0].get("total_balance") or 0)
    except (TypeError, ValueError):
        return None


def _generated_rows(path: Path) -> tuple[dict | None, list[dict]]:
    if not path.exists():
        return None, []
    meta = None
    rows = []
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text = text.rsplit("\n", 1)[0] + "\n"
        path.write_text(text, encoding="utf-8")
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") == "meta":
            meta = row
        else:
            rows.append(row)
    return meta, rows


def _run_thread(
    thread: dict,
    out_path: Path,
    client: LLMClient,
    packs: dict,
    cards: dict,
    stop: threading.Event,
    limit: int,
    retries: int,
) -> str:
    if stop.is_set():
        return "stopped"
    try:
        return _run_thread_body(thread, out_path, client, packs, cards, stop, limit, retries)
    except BillingError:
        stop.set()
        return "billing"
    except Exception as exc:  # noqa: BLE001 - one thread must not kill the arm
        print(f"[{thread.get('post_id')}] failed {type(exc).__name__}: {exc}", flush=True)
        return "ok"


def _run_thread_body(
    thread: dict,
    out_path: Path,
    client: LLMClient,
    packs: dict,
    cards: dict,
    stop: threading.Event,
    limit: int,
    retries: int,
) -> str:
    turns = thread["turns"]
    if limit:
        turns = turns[:limit]
    meta, done_rows = _generated_rows(out_path)
    if meta is None or meta.get("n_turns") != len(turns) or meta.get("endpoint") != client.base_url:
        meta = {
            "type": "meta",
            "post_id": thread["post_id"],
            "subreddit": thread["subreddit"],
            "arm": ARM,
            "endpoint": client.base_url,
            "model": client.model,
            "temperature": TEMPERATURE,
            "history_view": {"recent": RECENT, "earlier": EARLIER, "seed": SEED},
            "reply_max_tokens": MAX_TOKENS,
            "n_turns": len(turns),
            "protocol": "limited history 3+2; official DeepSeek; full comment positions",
        }
        done_rows = []
        out_path.write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")
    if len(done_rows) >= len(turns):
        return "skip"
    prompts = {}
    targets = {}
    sources = {}
    for author in thread["cast"]:
        if limit and author not in {t["author"] for t in turns}:
            continue
        prompt, source, aim = _persona(author, thread["post_id"], packs, cards)
        prompts[author] = prompt
        sources[author] = source
        targets[author] = aim
    generated = list(done_rows)
    with out_path.open("a", encoding="utf-8") as fout:
        for i in range(len(generated), len(turns)):
            if stop.is_set():
                return "stopped"
            turn = turns[i]
            author = turn["author"]
            shown = visible_history_indices(
                len(generated),
                recent=RECENT,
                earlier=EARLIER,
                seed=SEED,
                post_id=thread["post_id"],
                step=i,
            )
            user = agent_user_prompt(
                thread["title"], thread["selftext"], generated, targets.get(author), indices=shown
            )
            err = ""
            comp = None
            for attempt in range(retries + 1):
                if stop.is_set():
                    return "stopped"
                try:
                    comp = client.complete(
                        prompts[author],
                        user,
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                        disable_thinking=True,
                    )
                    if comp.text:
                        err = ""
                        break
                    err = "empty completion"
                except BillingError:
                    stop.set()
                    return "billing"
                except (RateLimitError, APIConnectionError) as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    time.sleep(2.0 * (attempt + 1))
                except Exception as exc:  # noqa: BLE001
                    err = f"{type(exc).__name__}: {exc}"
                    if "402" in err or "Insufficient" in err or "insufficient" in err.lower():
                        stop.set()
                        return "billing"
                    time.sleep(1.5 * (attempt + 1))
            row = {
                "position": turn["position"],
                "comment_id": turn["comment_id"],
                "alias": turn["alias"],
                "author": author,
                "body": "" if comp is None else comp.text,
                "n_chars": 0 if comp is None else len(comp.text),
                "human_n_chars": turn["n_chars"],
                "shown_indices": shown,
                "persona_source": sources.get(author, "none"),
                "finish_reason": "" if comp is None else comp.finish_reason,
                "truncated": bool(comp and comp.finish_reason == "length"),
                "prompt_tokens": 0 if comp is None else comp.prompt_tokens,
                "completion_tokens": 0 if comp is None else comp.completion_tokens,
                "response_model": "" if comp is None else comp.model,
                "error": err,
            }
            generated.append(row)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
    print(f"[{thread['post_id']}] {len(generated)}/{len(turns)}", flush=True)
    return "ok"


def run(cfg: dict, *, workers: int, threads: int, limit: int, retries: int) -> None:
    del cfg
    replay = ROOT / "outputs" / "scale5000" / "replay.jsonl"
    pack_path = ROOT / "outputs" / "scale5000" / "author_pack.jsonl"
    if not replay.exists() or not pack_path.exists():
        raise SystemExit("missing replay. Run: python scripts/run_scale_flash.py prepare")
    balance = _balance_cny()
    print(f"balance_cny={balance}", flush=True)
    if balance is not None and balance < 1:
        _write_stop("balance under 1 CNY before start", balance)
        raise SystemExit("balance under 1 CNY")
    print("loading personas", flush=True)
    packs = _load_packs(pack_path)
    cards = _load_llm_cards()
    print(f"authors={len(packs)} llm_cards={len(cards)}", flush=True)
    client = LLMClient.from_env("deepseek")
    out_dir = ROOT / "outputs" / "sims" / "scale5000" / ARM
    out_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    t0 = time.perf_counter()
    n_ok = n_skip = n_stop = 0
    reason = ""

    def _jobs():
        sent = 0
        with replay.open(encoding="utf-8") as f:
            for line in f:
                if threads and sent >= threads:
                    break
                if stop.is_set():
                    break
                thread = json.loads(line)
                sent += 1
                yield thread

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        inflight: set = set()
        for thread in _jobs():
            fut = pool.submit(
                _run_thread,
                thread,
                out_dir / f"{thread['post_id']}.jsonl",
                client,
                packs,
                cards,
                stop,
                limit,
                retries,
            )
            inflight.add(fut)
            if len(inflight) >= max(1, workers):
                done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for fut_done in done:
                    status = fut_done.result()
                    n_ok, n_skip, n_stop, reason = _count(status, n_ok, n_skip, n_stop, reason)
                    if status == "billing":
                        stop.set()
        while inflight:
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for fut_done in done:
                status = fut_done.result()
                n_ok, n_skip, n_stop, reason = _count(status, n_ok, n_skip, n_stop, reason)
    balance_end = _balance_cny()
    elapsed = (time.perf_counter() - t0) / 60
    print(
        f"finished status={reason or 'ok'} new_or_resumed={n_ok} skip={n_skip} "
        f"stopped={n_stop} minutes={elapsed:.1f} balance_cny={balance_end}",
        flush=True,
    )
    if reason == "billing" or (balance_end is not None and balance_end < 1):
        _write_stop(reason or "balance under 1 CNY", balance_end)


def _count(status: str, n_ok: int, n_skip: int, n_stop: int, reason: str) -> tuple[int, int, int, str]:
    if status == "skip":
        n_skip += 1
    elif status in {"stopped", "billing"}:
        n_stop += 1
        if status == "billing":
            reason = "billing"
    else:
        n_ok += 1
    if n_ok and n_ok % 20 == 0:
        print(f"threads_written={n_ok} skipped={n_skip}", flush=True)
    return n_ok, n_skip, n_stop, reason


def _write_stop(reason: str, balance: float | None) -> None:
    out = ROOT / "outputs" / "sims" / "scale5000" / ARM
    out.mkdir(parents=True, exist_ok=True)
    payload = {"reason": reason, "balance_cny": balance, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "STOP.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"STOP {payload}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare or run the official Flash full-dump arm")
    ap.add_argument("command", choices=["prepare", "run"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=0, help="cap number of posts")
    ap.add_argument("--limit", type=int, default=0, help="cap turns per post")
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()
    cfg = load_config(ROOT / "configs" / "default.yaml")
    if args.command == "prepare":
        prepare(cfg)
    else:
        run(cfg, workers=args.workers, threads=args.threads, limit=args.limit, retries=args.retries)


if __name__ == "__main__":
    main()
