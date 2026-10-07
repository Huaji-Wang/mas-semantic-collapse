"""History → DeepSeek persona → agent system prompt.

The conditioned dump ships history *counts* only (``participants[a].h``), so the
text a card is written from comes from the same dump: every comment the author
left in sampled threads other than the demo thread. The demo thread itself is
never shown to the card writer or to the agent.

Pipeline: filter history → code stats + sampled comments → DeepSeek writes
stance/tone/quirks → pin 2–3 verbatim samples → ``agent_system_prompt``.
Countable fields are computed here so the model cannot invent numbers.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from openai import APIStatusError, OpenAI, RateLimitError


class BillingError(RuntimeError):
    """The provider refused the call because the account cannot pay."""


@dataclass
class Completion:
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    model: str


@dataclass
class HistoryStats:
    n_comments: int
    n_subreddits: int
    top_subreddits: list[tuple[str, int]]
    len_p10: int
    len_median: int
    len_p90: int
    thin: bool  # too little history for a trustworthy card

    @staticmethod
    def from_bodies(bodies: list[str], subs: list[str], thin_below: int = 5) -> "HistoryStats":
        lens = sorted(len(b) for b in bodies) or [0]
        counts: dict[str, int] = {}
        for s in subs:
            counts[s] = counts.get(s, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
        q = np.percentile(lens, [10, 50, 90]) if bodies else np.zeros(3)
        return HistoryStats(
            n_comments=len(bodies),
            n_subreddits=len(counts),
            top_subreddits=top,
            len_p10=int(q[0]),
            len_median=int(q[1]),
            len_p90=int(q[2]),
            thin=len(bodies) < thin_below,
        )


@dataclass
class PersonaCard:
    author: str
    alias: str
    stats: HistoryStats
    samples: list[str]
    stance: str = ""
    tone: str = ""
    quirks: str = ""
    raw_card: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stats"] = asdict(self.stats)
        return d


@dataclass
class LLMClient:
    """OpenAI-compatible wrapper for DeepSeek, OpenAI, and Qwen."""

    api_key: str
    base_url: str
    model: str
    provider: str = "deepseek"
    _client: Any = field(default=None, repr=False)

    @staticmethod
    def from_env(provider: str = "deepseek", model: str | None = None) -> "LLMClient":
        if provider == "deepseek":
            key = os.getenv("DEEPSEEK_API_KEY", "").strip()
            base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
            name = model or os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip()
            if not key:
                raise SystemExit("DEEPSEEK_API_KEY missing. Put it in .env")
        elif provider == "openai":
            key = os.getenv("OPENAI_API_KEY", "").strip()
            base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
            name = model or os.getenv("OPENAI_MODEL", "gpt-5.6-luna").strip()
            if not key:
                raise SystemExit("OPENAI_API_KEY is empty in .env, so this provider cannot run yet")
        elif provider == "qwen":
            key = os.getenv("QWEN_API_KEY", "").strip() or os.getenv("DASHSCOPE_API_KEY", "").strip()
            base = os.getenv(
                "QWEN_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ).strip()
            name = model or os.getenv("QWEN_MODEL", "").strip()
            if not key:
                raise SystemExit("QWEN_API_KEY missing. Put it in .env")
            if not name:
                raise SystemExit("Qwen model id missing. Pass --model or set QWEN_MODEL")
        else:
            raise SystemExit(f"unknown provider {provider!r}")
        return LLMClient(api_key=key, base_url=base, model=name, provider=provider)

    def __post_init__(self) -> None:
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=180.0)

    def chat(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        temperature: float,
        disable_thinking: bool | None = None,
    ) -> str:
        return self.complete(
            system,
            user,
            max_tokens=max_tokens,
            temperature=temperature,
            disable_thinking=disable_thinking,
        ).text

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        temperature: float,
        disable_thinking: bool | None = None,
        route: dict[str, Any] | None = None,
    ) -> Completion:
        # Reasoning models spend the token budget before any comment text.
        # Comment generation needs the completion itself.
        want_off = disable_thinking if disable_thinking is not None else ("flash" in (self.model or ""))
        extra: dict[str, Any] = {}
        if want_off and self.provider == "qwen":
            extra["enable_thinking"] = False
        elif want_off and self.provider in {"deepseek", "openrouter"}:
            extra["thinking"] = {"type": "disabled"}
        if route:
            extra["provider"] = {
                "only": [route["provider_only"]],
                "allow_fallbacks": bool(route.get("allow_fallbacks", False)),
            }
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **({"extra_body": extra} if extra else {}),
            )
        except RateLimitError:
            raise
        except APIStatusError as exc:
            blob = f"{exc.status_code} {exc}"
            if exc.status_code in {401, 402} or "Insufficient Balance" in blob or "insufficient" in blob.lower():
                raise BillingError(blob) from exc
            raise
        if resp is None or not getattr(resp, "choices", None):
            raise RuntimeError(f"empty completion from {self.model}: {resp!r}")
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return Completion(
            text=(choice.message.content or "").strip(),
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            model=str(getattr(resp, "model", "") or self.model),
        )


CARD_SYSTEM = (
    "You write a persona from this account's past comments. "
    "Describe only what the comments show: positions, tone, and verbal habits. "
    "Counts and lengths are already supplied; leave them out of the JSON. "
    "Reply with JSON only, keys: stance, tone, quirks. Each value one or two sentences."
)


def build_card_prompt(stats: HistoryStats, bodies: list[str], max_chars: int) -> str:
    header = (
        f"This account left {stats.n_comments} comments across {stats.n_subreddits} "
        f"subreddit(s): {', '.join(f'r/{s} x{n}' for s, n in stats.top_subreddits)}. "
        f"Typical comment length {stats.len_p10}-{stats.len_p90} characters "
        f"(median {stats.len_median})."
    )
    parts = [header, "", "Past comments:"]
    used = 0
    for i, b in enumerate(bodies, 1):
        snippet = b.strip()
        if used + len(snippet) > max_chars:
            snippet = snippet[: max(0, max_chars - used)]
        if not snippet:
            break
        parts.append(f"[{i}] {snippet}")
        used += len(snippet)
        if used >= max_chars:
            break
    parts += [
        "",
        "JSON keys:",
        '  "stance"  - what positions or angles this account keeps taking',
        '  "tone"    - how it sounds: blunt, hedging, joking, lecturing, ...',
        '  "quirks"  - recurring verbal or formatting habits',
    ]
    return "\n".join(parts)


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_card(text: str) -> dict[str, str]:
    text = (text or "").strip()
    if not text:
        return {}
    start = text.find("{")
    obj: Any = None
    if start >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            m = _JSON_RE.search(text)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except json.JSONDecodeError:
                    obj = None
    if not isinstance(obj, dict):
        return {}
    fields = {k: str(obj.get(k, "")).strip() for k in ("stance", "tone", "quirks")}
    if not any(fields.values()):
        return {}
    return fields


def agent_system_prompt(card: PersonaCard, *, limit_length: bool = True) -> str:
    """What the agent is told about itself. Never mentions the demo thread."""
    s = card.stats
    lines = [
        "Who you are, based on your own posting history:",
    ]
    if card.stance:
        lines.append(f"- Positions you keep taking: {card.stance}")
    if card.tone:
        lines.append(f"- How you sound: {card.tone}")
    if card.quirks:
        lines.append(f"- Verbal habits: {card.quirks}")
    if s.top_subreddits:
        lines.append(f"- You mostly post in: {', '.join('r/' + x for x, _ in s.top_subreddits)}")
    if limit_length:
        lines.append(
            f"- Length: your comments run {s.len_p10}-{s.len_p90} characters, typically about "
            f"{s.len_median}. Aim for {s.len_median}; stay within {s.len_p90}."
        )
    if card.samples:
        lines += ["", "Things you have actually written before:"]
        lines += [f'  "{x}"' for x in card.samples]
    lines += [
        "",
        "Write one comment in this thread, in your own voice.",
        "Output only the comment text.",
    ]
    return "\n".join(lines)


def visible_history_indices(
    n_prev: int,
    *,
    recent: int = 3,
    earlier: int = 2,
    seed: int = 42,
    post_id: str = "",
    step: int = 0,
) -> list[int]:
    """Indices into already generated comments, shared by every arm.

    The draw depends only on the post id and the step, not on comment text,
    so six generators see the same positions and different bodies.
    """
    if n_prev <= 0:
        return []
    if n_prev <= recent + earlier:
        return list(range(n_prev))
    recent_idx = list(range(n_prev - recent, n_prev))
    pool = list(range(0, n_prev - recent))
    rng = random.Random(f"{seed}:{post_id}:{step}")
    picked = rng.sample(pool, earlier)
    return sorted(picked + recent_idx)


def agent_user_prompt(
    title: str,
    selftext: str,
    history: list[dict[str, str]],
    target_chars: int | None = None,
    indices: list[int] | None = None,
) -> str:
    """Post plus a view of generated comments. No real human comment is ever shown."""
    parts = ["Post title:", title.strip()]
    body = (selftext or "").strip()
    if body:
        parts += ["", "Post body:", body]
    shown = history if indices is None else [history[i] for i in indices]
    labels = list(range(1, len(shown) + 1)) if indices is None else [i + 1 for i in indices]
    if shown:
        parts += ["", "Comments so far:"]
        for n, h in zip(labels, shown):
            parts.append(f"[{n}] {h['alias']}: {h['body']}")
    else:
        parts += ["", "No one has commented yet. Yours is the first comment."]
    tail = "Write your comment now."
    if target_chars:
        tail += f" Around {target_chars} characters, and finish your last sentence."
    parts += ["", tail]
    return "\n".join(parts)


def max_tokens_for(stats: HistoryStats, chars_per_token: float, floor: int, ceil: int) -> int:
    """Cap from the author's own p90, with headroom so a reply ends mid-sentence rarely."""
    est = int(stats.len_p90 / max(chars_per_token, 1e-6)) + 60
    return int(min(max(est, floor), ceil))


def spread_bodies(bodies: list[str], n: int) -> list[str]:
    """Evenly spaced by length, so the card writer sees short and long alike."""
    if len(bodies) <= n:
        return list(bodies)
    order = sorted(bodies, key=len)
    idx = [round(i * (len(order) - 1) / (n - 1)) for i in range(n)]
    return [order[i] for i in sorted(set(idx))]


def verbatim_samples(bodies: list[str], n: int, max_chars: int, median: int) -> list[str]:
    ranked = sorted(bodies, key=lambda b: abs(len(b) - median))
    return [b[:max_chars] for b in ranked[:n]]


def card_from_history(
    author: str,
    alias: str,
    records: list[dict[str, Any]],
    client: LLMClient,
    pd_cfg: dict[str, Any],
    *,
    blocked: set[str] | None = None,
) -> PersonaCard:
    """One author: history comments in, DeepSeek persona out."""
    blocked = blocked or set()
    hist = [h for h in records if h.get("post_id") not in blocked]
    bodies = [(h.get("body") or "").strip() for h in hist]
    bodies = [b for b in bodies if b]
    subs = [str(h.get("subreddit") or "") for h in hist if (h.get("body") or "").strip()]
    stats = HistoryStats.from_bodies(bodies, subs)
    samples = verbatim_samples(
        bodies, int(pd_cfg["card_samples"]), int(pd_cfg["sample_max_chars"]), stats.len_median
    )
    card = PersonaCard(author=author, alias=alias, stats=stats, samples=samples)
    if not bodies:
        card.error = "no history"
        return card
    prompt = build_card_prompt(
        stats, spread_bodies(bodies, int(pd_cfg["history_max_samples"])), int(pd_cfg["history_max_chars"])
    )
    text = ""
    fields: dict[str, str] = {}
    last_err = ""
    for _attempt in range(3):
        try:
            text = client.chat(
                CARD_SYSTEM, prompt, max_tokens=2000, temperature=float(pd_cfg["card_temperature"])
            )
        except Exception as exc:  # noqa: BLE001 - surface API failures in the artifact
            last_err = f"{type(exc).__name__}: {exc}"
            continue
        fields = parse_card(text)
        if fields:
            last_err = ""
            break
        last_err = "empty completion" if not text else "card was not valid JSON"
    card.raw_card = text
    card.stance = fields.get("stance", "")
    card.tone = fields.get("tone", "")
    card.quirks = fields.get("quirks", "")
    if last_err:
        card.error = last_err
    return card


def card_to_saved(
    card: PersonaCard,
    model: str | None = None,
    history_source: str | None = None,
) -> dict[str, Any]:
    d = card.to_dict()
    if card.stance or card.tone or card.quirks or card.samples:
        d["system_prompt"] = agent_system_prompt(card)
    if model:
        d["writer_model"] = model
    if history_source:
        d["history_source"] = history_source
    return d


def iter_cast_authors(raw: dict) -> list[tuple[str, str, str]]:
    """Unique (author, alias, first_post_id) in dump order."""
    seen: set[str] = set()
    rows: list[tuple[str, str, str]] = []
    for pid, t in (raw.get("threads") or {}).items():
        alias_map = t.get("alias") or {}
        for author in t.get("cast") or []:
            if author in seen:
                continue
            seen.add(author)
            rows.append((author, alias_map.get(author) or author, pid))
    return rows


def load_user_history_jsonl(path: Path) -> dict[str, Any]:
    """Turn extract_user_history.py output into the same shape as cast_and_history.json."""
    history: dict[str, list] = {}
    cast_threads: dict[str, list] = {}
    alias: dict[str, str] = {}
    authors: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        author = row["author"]
        authors.append(author)
        history[author] = row.get("records") or []
        cast_threads[author] = list(row.get("blocked_posts") or [])
        alias[author] = row.get("alias") or author
    return {
        "history": history,
        "cast_threads": cast_threads,
        "history_source": "user_history_zst",
        "threads": {"_user_history": {"cast": authors, "alias": alias}},
    }


def build_cards(
    raw: dict,
    pd_cfg: dict[str, Any],
    client: LLMClient | None,
    *,
    reused: dict[str, dict] | None = None,
    existing: dict[str, dict] | None = None,
    force: bool = False,
    limit: int = 0,
    authors: set[str] | None = None,
    on_update: Any = None,
    history_source: str | None = None,
) -> dict[str, dict]:
    reused = reused or {}
    existing = dict(existing or {})
    current_model = getattr(client, "model", None) if client is not None else None
    source = history_source or raw.get("history_source") or "dump_cross_thread"
    cast_threads = {a: set(s) for a, s in (raw.get("cast_threads") or {}).items()}
    history = raw.get("history") or {}
    cards: dict[str, dict] = {}
    n_reuse = n_skip = n_new = 0

    def _keep_existing(author: str, alias: str, src: dict) -> dict:
        d = dict(src)
        d["alias"] = alias
        if "system_prompt" not in d:
            try:
                tmp = PersonaCard(
                    author=d["author"],
                    alias=alias,
                    stats=HistoryStats(**d["stats"]),
                    samples=d.get("samples") or [],
                    stance=d.get("stance", ""),
                    tone=d.get("tone", ""),
                    quirks=d.get("quirks", ""),
                )
                d["system_prompt"] = agent_system_prompt(tmp)
            except Exception:  # noqa: BLE001
                pass
        return d

    for author, alias, pid in iter_cast_authors(raw):
        if authors and author not in authors:
            continue
        if limit and (n_reuse + n_skip + n_new) >= limit:
            break
        have = existing.get(author)
        valid = bool(have and not have.get("error"))
        already_this_model = bool(valid and current_model and have.get("writer_model") == current_model)
        already_this_source = bool(valid and have.get("history_source") == source)
        if not force and valid and already_this_model and already_this_source:
            cards[author] = _keep_existing(author, alias, have)
            n_skip += 1
            continue
        if not force and valid and already_this_source:
            cards[author] = _keep_existing(author, alias, have)
            n_skip += 1
            continue
        if not force and author in reused and not reused[author].get("error"):
            card = dict(reused[author])
            card["alias"] = alias
            card["reused"] = True
            tmp = PersonaCard(
                author=card["author"],
                alias=alias,
                stats=HistoryStats(**card["stats"]),
                samples=card.get("samples") or [],
                stance=card.get("stance", ""),
                tone=card.get("tone", ""),
                quirks=card.get("quirks", ""),
            )
            card["system_prompt"] = agent_system_prompt(tmp)
            cards[author] = card
            n_reuse += 1
            print(f"  reuse {alias:<8} {author}")
            continue
        blocked = cast_threads.get(author, {pid})
        if client is None:
            raise SystemExit(f"need an LLM client to write a new persona for {author}")
        card = card_from_history(
            author, alias, history.get(author, []), client, pd_cfg, blocked=blocked
        )
        saved = card_to_saved(card, model=client.model, history_source=source)
        cards[author] = saved
        n_new += 1
        if on_update:
            on_update({author: saved})
        flag = " [thin history]" if card.stats.thin else ""
        if card.error:
            print(f"  card FAILED {alias} ({author}): {card.error}", flush=True)
        else:
            print(f"  card {alias:<8} {author:<28} n={card.stats.n_comments:<4}{flag}", flush=True)
    print(f"cards reused={n_reuse} skipped={n_skip} new={n_new} total={len(cards)}")
    return cards


def _safe_author_name(author: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", author)[:80] or "author"


def save_persona_outputs(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cards_path = out_dir / "cards.json"
    cards_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    pdir = out_dir / "prompts"
    pdir.mkdir(parents=True, exist_ok=True)
    n_prompt = 0
    for author, d in (payload.get("cards") or {}).items():
        text = (d.get("system_prompt") or "").strip()
        if not text:
            continue
        alias = d.get("alias") or "Agent"
        (pdir / f"{alias}_{_safe_author_name(author)}.txt").write_text(text + "\n", encoding="utf-8")
        n_prompt += 1
    print(f"wrote {cards_path}")
    print(f"wrote {n_prompt} prompts under {pdir}")
