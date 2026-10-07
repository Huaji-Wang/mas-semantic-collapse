"""Read posting history and have DeepSeek write each agent's persona.

Default input is ``cast_and_history.json`` (dump-proxy comments). With
``--from-user-history`` it reads ``user_history.jsonl`` instead: up to 300
crawled records per author, demo posts already stripped.

    python scripts/write_personas.py
    python scripts/write_personas.py --from-user-history
    python scripts/write_personas.py --from-user-history --limit 3
    python scripts/write_personas.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mas_collapse.config import load_config
from mas_collapse.simulate.persona import (
    LLMClient,
    build_cards,
    iter_cast_authors,
    load_user_history_jsonl,
    save_persona_outputs,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="History comments → DeepSeek persona → agent system prompt")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--provider", default="deepseek", choices=["deepseek", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--force", action="store_true", help="rewrite personas even if cards.json already has them")
    ap.add_argument("--limit", type=int, default=0, help="cap authors this run (resume the rest later)")
    ap.add_argument("--authors", default="", help="comma-separated Reddit usernames; default is the whole cast")
    ap.add_argument(
        "--from-user-history",
        action="store_true",
        help="use outputs/personas/{tag}/user_history.jsonl (crawled 300-record packs)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    pd_cfg = cfg["persona_demo"]
    out_dir = Path(cfg["paths"]["personas"]) / pd_cfg["tag"]
    cards_path = out_dir / "cards.json"
    if args.from_user_history:
        raw_path = out_dir / "user_history.jsonl"
        if not raw_path.exists():
            raise SystemExit(
                f"missing {raw_path}\n"
                "run: python scripts/extract_user_history.py"
            )
        raw = load_user_history_jsonl(raw_path)
        source = "user_history_zst"
        print(f"loaded {len(raw['history'])} authors from {raw_path}")
    else:
        raw_path = out_dir / "cast_and_history.json"
        if not raw_path.exists():
            raise SystemExit(
                f"missing {raw_path}\n"
                "run: python scripts/build_persona_demo.py --no-cards"
            )
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        source = raw.get("history_source") or "dump_cross_thread"

    existing: dict = {}
    prev_model = ""
    prev_provider = args.provider
    if cards_path.exists():
        prev = json.loads(cards_path.read_text(encoding="utf-8"))
        existing = prev.get("cards") or {}
        prev_model = prev.get("model") or ""
        prev_provider = prev.get("provider") or args.provider
        print(f"loaded {len(existing)} existing cards from {cards_path}")

    reused = {}
    if not args.from_user_history:
        reuse_from = pd_cfg.get("reuse_cards_from") or ""
        reuse_path = Path(reuse_from) if reuse_from else None
        if reuse_path and not reuse_path.is_absolute():
            reuse_path = ROOT / reuse_path
        if reuse_path and reuse_path.exists():
            reused = json.loads(reuse_path.read_text(encoding="utf-8")).get("cards") or {}
            print(f"will reuse cards from {reuse_path} ({len(reused)} authors)")

    want = {a.strip() for a in args.authors.split(",") if a.strip()} or None
    target = (args.model or cfg["simulation"].get("model") or "deepseek-flash").strip()
    need_llm: list[str] = []
    for author, _alias, _pid in iter_cast_authors(raw):
        if want and author not in want:
            continue
        have = existing.get(author)
        valid = bool(have and not have.get("error"))
        same_source = bool(valid and have.get("history_source") == source)
        if valid and have.get("writer_model") == target and same_source and not args.force:
            continue
        if valid and same_source and not args.force:
            continue
        if not args.force and author in reused and not reused[author].get("error"):
            continue
        need_llm.append(author)
        if args.limit and len(need_llm) >= args.limit:
            break
    client = None
    if need_llm:
        client = LLMClient.from_env(args.provider, args.model)
        print(f"writing {len(need_llm)} personas with {client.model} from {source}", flush=True)
    else:
        print("no new DeepSeek calls; assembling system prompts from existing cards")

    acc = dict(existing)

    def on_update(partial: dict) -> None:
        acc.update(partial)
        cards_path.write_text(
            json.dumps(
                {
                    "model": client.model if client else prev_model,
                    "provider": args.provider if client else prev_provider,
                    "history_source": source,
                    "cards": acc,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )

    built = build_cards(
        raw,
        pd_cfg,
        client,
        reused=reused,
        existing=existing,
        force=args.force,
        limit=0,
        authors=set(need_llm) if need_llm else want,
        on_update=on_update if client else None,
        history_source=source,
    )
    cards = dict(existing)
    cards.update(built)
    payload = {
        "model": client.model if client else prev_model,
        "provider": args.provider if client else prev_provider,
        "history_source": source,
        "cards": cards,
    }
    save_persona_outputs(out_dir, payload)


if __name__ == "__main__":
    main()
