from __future__ import annotations

import asyncio
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from openai import OpenAI

from mas_collapse.data.threads import Comment, Thread, linearize_comments, post_text


StructureMode = Literal["flat", "weak_tree"]
OrderMode = Literal["round_robin", "random"]
ProtocolMode = Literal["post_only", "prefix10"]


@dataclass
class SimComment:
    agent: str
    body: str
    parent_id: str | None
    created_order: int


@dataclass
class SimulationConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.9
    max_tokens: int = 200
    n_agents: int = 3
    speaking_order: OrderMode = "round_robin"
    structure: StructureMode = "flat"
    weak_tree_reply_prob: float = 0.25
    protocol: ProtocolMode = "post_only"
    prefix_k: int = 10
    max_comments: int = 100
    seed: int = 42


@dataclass
class SimulationResult:
    post_id: str
    protocol: str
    structure: str
    speaking_order: str
    model: str
    comments: list[SimComment] = field(default_factory=list)
    seed_comments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


AGENT_NAMES = ["Commenter-A", "Commenter-B", "Commenter-C"]


def _system_prompt(name: str, subreddit: str) -> str:
    return (
        f"You are {name}, a Reddit user commenting in r/{subreddit}. "
        "Write one short natural comment (1-3 sentences). "
        "Do not roleplay as a moderator. Do not mention being an AI. "
        "Respond to the post and prior comments."
    )


def _build_user_prompt(
    thread: Thread,
    history: list[SimComment],
    seed_bodies: list[str],
    parent_hint: str | None,
) -> str:
    parts = [
        f"Subreddit: r/{thread.subreddit}",
        "Post:",
        post_text(thread),
    ]
    if seed_bodies:
        parts.append("\nEarlier human comments:")
        for i, b in enumerate(seed_bodies, 1):
            parts.append(f"[H{i}] {b}")
    if history:
        parts.append("\nDiscussion so far:")
        for c in history:
            pid = f" (reply to {c.parent_id})" if c.parent_id else ""
            parts.append(f"[{c.agent}{pid}] {c.body}")
    if parent_hint:
        parts.append(f"\nYour comment should reply to: {parent_hint}")
    else:
        parts.append("\nYour comment should be a top-level reply to the post.")
    parts.append("\nWrite only the comment text.")
    return "\n".join(parts)


def _choose_parent(
    structure: StructureMode,
    reply_prob: float,
    history: list[SimComment],
    rng: random.Random,
) -> str | None:
    if structure == "flat" or not history:
        return None
    if rng.random() > reply_prob:
        return None
    recent = history[-3:]
    target = rng.choice(recent)
    return f"{target.agent}#{target.created_order}"


def _agent_cycle(n_agents: int, n_comments: int, order: OrderMode, rng: random.Random) -> list[str]:
    names = AGENT_NAMES[:n_agents]
    out: list[str] = []
    if order == "round_robin":
        for i in range(n_comments):
            out.append(names[i % n_agents])
        return out
    # random: reshuffle each round of n_agents
    while len(out) < n_comments:
        batch = names[:]
        rng.shuffle(batch)
        out.extend(batch)
    return out[:n_comments]


class DeepSeekCommenter:
    """OpenAI-compatible client used inside an AutoGen-style turn loop."""

    def __init__(self, cfg: SimulationConfig):
        if not cfg.api_key:
            raise ValueError("DEEPSEEK_API_KEY missing. Set it in .env")
        self.cfg = cfg
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    def generate(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


async def _run_autogen_round(
    commenter: DeepSeekCommenter,
    name: str,
    thread: Thread,
    history: list[SimComment],
    seed_bodies: list[str],
    parent_hint: str | None,
) -> str:
    """Async wrapper so the runner can sit behind AutoGen's async API later."""
    system = _system_prompt(name, thread.subreddit or "reddit")
    user = _build_user_prompt(thread, history, seed_bodies, parent_hint)
    # Offload sync HTTP to thread to keep async surface
    return await asyncio.to_thread(commenter.generate, system, user)


def run_thread_simulation(thread: Thread, cfg: SimulationConfig) -> SimulationResult:
    rng = random.Random(cfg.seed + hash(thread.post_id) % 10_000)
    linearized = linearize_comments(thread)
    target_n = min(cfg.max_comments, len(linearized))
    if target_n < 1:
        target_n = min(cfg.max_comments, 50)

    seed_bodies: list[str] = []
    seed_meta: list[dict[str, Any]] = []
    if cfg.protocol == "prefix10":
        for c in linearized[: cfg.prefix_k]:
            seed_bodies.append(c.body)
            seed_meta.append(
                {"id": c.id, "author": c.author, "body": c.body, "created_utc": c.created_utc}
            )
        # generate remaining comments after prefix
        gen_n = max(0, target_n - len(seed_bodies))
    else:
        gen_n = target_n

    commenter = DeepSeekCommenter(cfg)
    speakers = _agent_cycle(cfg.n_agents, gen_n, cfg.speaking_order, rng)
    history: list[SimComment] = []

    async def _loop() -> list[SimComment]:
        local: list[SimComment] = []
        for i, name in enumerate(speakers):
            parent = _choose_parent(cfg.structure, cfg.weak_tree_reply_prob, local, rng)
            body = await _run_autogen_round(
                commenter, name, thread, local, seed_bodies, parent
            )
            local.append(
                SimComment(agent=name, body=body, parent_id=parent, created_order=i)
            )
        return local

    history = asyncio.run(_loop())

    return SimulationResult(
        post_id=thread.post_id,
        protocol=cfg.protocol,
        structure=cfg.structure,
        speaking_order=cfg.speaking_order,
        model=cfg.model,
        comments=history,
        seed_comments=seed_meta,
    )


# Optional AutoGen GroupChat facade (kept thin; generation uses DeepSeek client above)
def try_build_autogen_agents(cfg: SimulationConfig):
    """Best-effort AutoGen agent construction for framework completeness."""
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_ext.models.openai import OpenAIChatCompletionClient
    except ImportError:
        return None

    model_client = OpenAIChatCompletionClient(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        },
    )
    agents = []
    for name in AGENT_NAMES[: cfg.n_agents]:
        agents.append(
            AssistantAgent(
                name=name.replace("-", "_"),
                model_client=model_client,
                system_message=_system_prompt(name, "reddit"),
            )
        )
    return agents
