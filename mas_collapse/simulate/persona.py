"""Persona cards built from a Reddit author's comments in *other* sampled threads.

The conditioned dump ships history *counts* only (``participants[a].h``), so the
text a card is written from comes from the same dump: every comment the author
left in sampled threads other than the demo thread. The demo thread itself is
never shown to the card writer or to the agent.

Countable fields (history size, length quantiles, subreddit mix) are computed
here. The model is only asked for stance, tone and verbal habits, so it cannot
invent numbers.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from openai import OpenAI


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
    """Thin OpenAI-compatible wrapper so DeepSeek and gpt-5.6-luna are one switch."""

    api_key: str
    base_url: str
    model: str
    _client: Any = field(default=None, repr=False)

    @staticmethod
    def from_env(provider: str = "deepseek", model: str | None = None) -> "LLMClient":
        if provider == "deepseek":
            key = os.getenv("DEEPSEEK_API_KEY", "").strip()
            base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
            name = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
            if not key:
                raise SystemExit("DEEPSEEK_API_KEY missing. Put it in .env")
        elif provider == "openai":
            key = os.getenv("OPENAI_API_KEY", "").strip()
            base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
            name = model or os.getenv("OPENAI_MODEL", "gpt-5.6-luna").strip()
            if not key:
                raise SystemExit("OPENAI_API_KEY is empty in .env, so this provider cannot run yet")
        else:
            raise SystemExit(f"unknown provider {provider!r}")
        return LLMClient(api_key=key, base_url=base, model=name)

    def __post_init__(self) -> None:
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, system: str, user: str, *, max_tokens: int, temperature: float) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


CARD_SYSTEM = (
    "You profile a Reddit commenter from their past comments. "
    "Describe only what the comments show. Do not guess demographics, job, age or location. "
    "Do not count anything; the counts are supplied separately. "
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
    m = _JSON_RE.search(text or "")
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(obj, dict):
        return {}
    return {k: str(obj.get(k, "")).strip() for k in ("stance", "tone", "quirks")}


def agent_system_prompt(card: PersonaCard, subreddit: str) -> str:
    """What the agent is told about itself. Never mentions the demo thread."""
    s = card.stats
    lines = [
        f"You are a Reddit user commenting in r/{subreddit}.",
        "",
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
    lines.append(
        f"- Length: your comments run {s.len_p10}-{s.len_p90} characters, typically about "
        f"{s.len_median}. Aim for {s.len_median}; never go past {s.len_p90}. "
        "Most Reddit comments are one short paragraph, so do not pad."
    )
    if card.samples:
        lines += ["", "Things you have actually written before:"]
        lines += [f'  "{x}"' for x in card.samples]
    lines += [
        "",
        "Write one comment in this thread, in your own voice.",
        "Do not roleplay as a moderator. Do not mention being an AI or a persona.",
        "Do not restate these instructions. Output only the comment text.",
    ]
    return "\n".join(lines)


def agent_user_prompt(
    title: str, selftext: str, history: list[dict[str, str]], target_chars: int | None = None
) -> str:
    """Post plus the comments generated so far. No real human comment is ever shown."""
    parts = ["Post title:", title.strip()]
    body = (selftext or "").strip()
    if body:
        parts += ["", "Post body:", body]
    if history:
        parts += ["", "Comments so far:"]
        for i, h in enumerate(history, 1):
            parts.append(f"[{i}] {h['alias']}: {h['body']}")
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
