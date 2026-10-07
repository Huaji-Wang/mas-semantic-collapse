"""Replay one scale arm into its own directory.

Same cards and the same cast file for every arm. Existing persona_148 and
persona_lenfree simulations are not read or written.

    python scripts/run_scale_arms.py --list
    python scripts/run_scale_arms.py --arm low_deepseek --limit 1 --workers 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mas_collapse.config import load_config
from mas_collapse.simulate.persona import LLMClient
from scripts.run_persona_demo import _card_from_dict, run_one_thread

FORBIDDEN = {"persona_148", "persona_lenfree", "persona_48"}


def _arms(scale: dict) -> list[dict]:
    arms = scale.get("arms") or []
    if len(arms) != 6:
        raise SystemExit(f"scale5000.yaml has {len(arms)} arms; expected 6")
    tiers: dict[str, int] = {}
    for arm in arms:
        tiers[arm["tier"]] = tiers.get(arm["tier"], 0) + 1
    if tiers != {"high": 2, "mid": 2, "low": 2}:
        raise SystemExit(f"tier counts must be 2/2/2, got {tiers}")
    return arms


def _resolve_model(arm: dict) -> str:
    env_name = arm.get("model_env") or ""
    from_env = os.getenv(env_name, "").strip() if env_name else ""
    name = from_env or str(arm.get("model") or "").strip()
    if not name:
        raise SystemExit(
            f"{arm['id']} has no model id. Set {env_name} in .env or model: in configs/scale5000.yaml"
        )
    return name


def _list(arms: list[dict]) -> None:
    print(f"{'arm':<16} {'tier':<6} {'provider':<10} model")
    for arm in arms:
        env_name = arm.get("model_env") or ""
        from_env = os.getenv(env_name, "").strip() if env_name else ""
        name = from_env or str(arm.get("model") or "").strip() or "(unset)"
        print(f"{arm['id']:<16} {arm['tier']:<6} {arm['provider']:<10} {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one six-arm scale replay")
    ap.add_argument("--arm", default="", help="arm id in configs/scale5000.yaml")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap turns per thread")
    ap.add_argument("--threads", type=int, default=0, help="cap number of posts")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    load_config(ROOT / "configs" / "default.yaml")
    scale = yaml.safe_load((ROOT / "configs" / "scale5000.yaml").read_text(encoding="utf-8"))
    arms = _arms(scale)
    if args.list or not args.arm:
        _list(arms)
        return

    arm = next((a for a in arms if a["id"] == args.arm), None)
    if arm is None:
        raise SystemExit(f"unknown arm {args.arm!r}")
    tag = str(scale["tag"])
    if tag in FORBIDDEN or arm["id"] in FORBIDDEN:
        raise SystemExit("refusing to write into an existing persona run")

    persona_dir = ROOT / "outputs" / "personas" / tag
    cast_path = persona_dir / "cast_and_history.json"
    cards_path = persona_dir / "cards.json"
    if not cast_path.exists() or not cards_path.exists():
        raise SystemExit(
            f"missing {cast_path.name} or {cards_path.name} under {persona_dir}. "
            "The manifest is outputs/scale5000/summary.json. Cards are shared across arms and are not built per model."
        )

    model = _resolve_model(arm)
    client = LLMClient.from_env(arm["provider"], model)
    raw = json.loads(cast_path.read_text(encoding="utf-8"))
    cards_blob = json.loads(cards_path.read_text(encoding="utf-8"))
    cards = {a: _card_from_dict(d) for a, d in cards_blob["cards"].items()}
    history_source = cards_blob.get("history_source") or "unspecified"
    threads = list(raw["threads"].values())
    if args.threads:
        threads = threads[: args.threads]

    out_dir = ROOT / "outputs" / "sims" / tag / arm["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers or int(scale.get("workers") or 4)
    disable = bool(scale.get("disable_thinking", True))
    view = scale.get("history_view") or {}
    history_view = {
        "recent": int(view.get("recent", 3)),
        "earlier": int(view.get("earlier", 2)),
        "seed": int(view.get("seed", 42)),
    }
    reply_max_tokens = int(scale.get("reply_max_tokens", 128))
    reply_target_chars = int(scale.get("reply_target_chars", 200))
    print(
        f"{arm['id']} tier={arm['tier']} provider={client.provider} model={client.model} "
        f"posts={len(threads)} history={history_view['recent']}+{history_view['earlier']} "
        f"max_tokens={reply_max_tokens} out={out_dir}"
    )

    t0 = time.perf_counter()
    jobs = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for thread in threads:
            jobs.append(
                pool.submit(
                    run_one_thread,
                    thread,
                    cards,
                    client,
                    out_dir / f"{thread['post_id']}.json",
                    temperature=float(scale.get("temperature", 0.9)),
                    chars_per_token=3.6,
                    floor=80,
                    ceil=900,
                    limit=args.limit,
                    retries=args.retries,
                    history_source=history_source,
                    limit_length=True,
                    free_max_tokens=int(scale.get("max_tokens_free", 8192)),
                    disable_thinking=disable,
                    arm=arm["id"],
                    tier=arm["tier"],
                    history_view=history_view,
                    reply_max_tokens=reply_max_tokens,
                    reply_target_chars=reply_target_chars,
                )
            )
        results = [j.result() for j in jobs]
    failed = sum(sum(1 for g in r["generated"] if g.get("error")) for r in results)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min, failed comments={failed}, wrote {out_dir}")


if __name__ == "__main__":
    main()
