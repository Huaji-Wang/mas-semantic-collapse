"""Smoke: 1 converge + 1 non_converge, post_only, flat, ~20 comments each."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mas_collapse.config import load_config
from mas_collapse.data.threads import iter_threads
from mas_collapse.simulate.autogen_runner import SimulationConfig, run_thread_simulation


def _pick_ids(labels_path: Path) -> dict[str, str]:
    buckets: dict[str, list[dict]] = {"converge": [], "non_converge": []}
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            lab = r.get("label_A")
            if lab in buckets:
                buckets[lab].append(r)
    picked = {}
    for lab, rows in buckets.items():
        if not rows:
            raise SystemExit(f"No {lab} rows in {labels_path}")
        # prefer mid-length threads for readable smoke
        rows = sorted(rows, key=lambda r: abs(int(r.get("n_comments", 0)) - 80))
        picked[lab] = rows[0]["post_id"]
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-comments", type=int, default=20)
    ap.add_argument(
        "--labels",
        default="outputs/labels/thread_labels_q80q20.jsonl",
        help="prefer q80/q20 labels",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not cfg["simulation"].get("api_key"):
        raise SystemExit(
            "DEEPSEEK_API_KEY missing.\n"
            "Put it in .env (copy from .env.example). Do NOT paste the key into chat."
        )

    labels_path = Path(args.labels)
    if not labels_path.is_absolute():
        labels_path = ROOT / labels_path
    picked = _pick_ids(labels_path)
    print("picked:", picked)

    want = set(picked.values())
    threads = {}
    for th in iter_threads(
        cfg["data"]["threads_path"],
        min_comments=int(cfg["data"]["min_comments"]),
        max_comments=cfg["data"].get("max_comments"),
    ):
        if th.post_id in want:
            threads[th.post_id] = th
        if len(threads) >= len(want):
            break

    out_dir = Path(cfg["paths"]["sims"]) / "smoke_flat_post_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    for lab, pid in picked.items():
        th = threads.get(pid)
        if th is None:
            print(f"WARN: thread {pid} ({lab}) not found under filters; retry without max_comments")
            for th2 in iter_threads(cfg["data"]["threads_path"], min_comments=50, max_comments=None):
                if th2.post_id == pid:
                    th = th2
                    break
        if th is None:
            raise SystemExit(f"Could not load thread {pid}")

        sc = SimulationConfig(
            api_key=cfg["simulation"]["api_key"],
            base_url=cfg["simulation"]["base_url"],
            model=cfg["simulation"]["model"],
            temperature=float(cfg["simulation"]["temperature"]),
            max_tokens=int(cfg["simulation"]["max_tokens"]),
            n_agents=3,
            speaking_order="round_robin",
            structure="flat",
            protocol="post_only",
            max_comments=args.max_comments,
            seed=int(cfg["project"]["seed"]),
        )
        title_safe = (th.title or "").encode("ascii", "replace").decode("ascii")[:80]
        print(f"\n=== simulating {lab} | r/{th.subreddit} | {title_safe} ===")
        result = run_thread_simulation(th, sc)
        payload = result.to_dict()
        payload["human_label_A"] = lab
        payload["title"] = th.title
        payload["subreddit"] = th.subreddit
        out_path = out_dir / f"{lab}_{pid}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # human-readable transcript
        txt_path = out_dir / f"{lab}_{pid}.txt"
        lines = [
            f"label: {lab}",
            f"post_id: {pid}",
            f"subreddit: r/{th.subreddit}",
            f"title: {th.title}",
            f"selftext: {(th.selftext or '')[:500]}",
            "",
            "--- simulated comments ---",
        ]
        for c in result.comments:
            lines.append(f"[{c.agent}] {c.body}")
            lines.append("")
        txt_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote {out_path}")
        print(f"wrote {txt_path}")
        for c in result.comments[:6]:
            body_safe = (c.body or "").encode("ascii", "replace").decode("ascii")[:120]
            print(f"  [{c.agent}] {body_safe}")
        if len(result.comments) > 6:
            print(f"  ... ({len(result.comments)} comments total)")


if __name__ == "__main__":
    main()
