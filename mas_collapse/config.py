from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    cfg_path = Path(path) if path else ROOT / "configs" / "default.yaml"
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    threads = Path(cfg["data"]["threads_path"])
    if not threads.is_absolute():
        threads = (cfg_path.parent.parent / threads).resolve()
    cfg["data"]["threads_path"] = str(threads)

    for key in ("outputs", "labels", "sims"):
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        cfg["paths"][key] = str(p)
        p.mkdir(parents=True, exist_ok=True)

    # env overrides for generation
    cfg["simulation"]["api_key"] = os.getenv("DEEPSEEK_API_KEY", "")
    cfg["simulation"]["base_url"] = os.getenv(
        "DEEPSEEK_BASE_URL", cfg["simulation"].get("base_url", "https://api.deepseek.com")
    )
    cfg["simulation"]["model"] = os.getenv(
        "DEEPSEEK_MODEL", cfg["simulation"].get("model", "deepseek-chat")
    )
    cfg["embedding"]["openai_api_key"] = os.getenv("OPENAI_API_KEY", "")
    cfg["embedding"]["openai_model"] = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        cfg["embedding"].get("openai_model", "text-embedding-3-large"),
    )
    return cfg
