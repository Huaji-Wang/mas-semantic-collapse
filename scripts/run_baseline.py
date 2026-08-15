from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mas_collapse.config import load_config
from mas_collapse.data.threads import iter_threads
from mas_collapse.simulate.autogen_runner import SimulationConfig, run_thread_simulation


def _load_labeled_ids(labels_path: Path) -> dict[str, list[str]]:
    buckets = {"converge": [], "non_converge": []}
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            lab = r.get("label_A")
            if lab in buckets:
                buckets[lab].append(r["post_id"])
    return buckets


def main() -> None:
    ap = argparse.ArgumentParser(description="Run AutoGen/DeepSeek multi-agent baseline")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--structure", choices=["flat", "weak_tree"], default=None)
    ap.add_argument("--speaking-order", choices=["round_robin", "random"], default=None)
    ap.add_argument("--protocol", choices=["post_only", "prefix10", "both"], default=None)
    ap.add_argument("--n-per-class", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="hard cap total sims (smoke)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sim_cfg = cfg["simulation"]
    structure = args.structure or sim_cfg["structure"]
    speaking_order = args.speaking_order or sim_cfg["speaking_order"]
    protocol_mode = args.protocol or sim_cfg["protocol"]
    n_per_class = args.n_per_class or int(sim_cfg["n_per_class"])

    labels_path = Path(cfg["paths"]["labels"]) / "thread_labels.jsonl"
    if not labels_path.exists():
        raise SystemExit(f"Missing labels: {labels_path}. Run scripts.label_threads first.")

    buckets = _load_labeled_ids(labels_path)
    rng = random.Random(int(cfg["project"]["seed"]))
    selected: list[tuple[str, str]] = []
    for lab in ("converge", "non_converge"):
        ids = buckets[lab][:]
        rng.shuffle(ids)
        for pid in ids[:n_per_class]:
            selected.append((lab, pid))

    want = {pid for _, pid in selected}
    threads = {}
    for th in tqdm(iter_threads(cfg["data"]["threads_path"], cfg["data"]["min_comments"]), desc="load"):
        if th.post_id in want:
            threads[th.post_id] = th
        if len(threads) >= len(want):
            break

    protocols = ["post_only", "prefix10"] if protocol_mode == "both" else [protocol_mode]
    out_dir = Path(cfg["paths"]["sims"]) / f"{structure}_{speaking_order}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done = 0
    for lab, pid in tqdm(selected, desc="simulate"):
        th = threads.get(pid)
        if th is None:
            continue
        for protocol in protocols:
            sc = SimulationConfig(
                api_key=sim_cfg["api_key"],
                base_url=sim_cfg["base_url"],
                model=sim_cfg["model"],
                temperature=float(sim_cfg["temperature"]),
                max_tokens=int(sim_cfg["max_tokens"]),
                n_agents=int(sim_cfg["n_agents"]),
                speaking_order=speaking_order,
                structure=structure,
                weak_tree_reply_prob=float(sim_cfg["weak_tree_reply_prob"]),
                protocol=protocol,
                prefix_k=int(sim_cfg["prefix_k"]),
                max_comments=int(sim_cfg["max_comments"]),
                seed=int(cfg["project"]["seed"]),
            )
            result = run_thread_simulation(th, sc)
            payload = result.to_dict()
            payload["human_label_A"] = lab
            out_path = out_dir / f"{pid}_{protocol}.json"
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            n_done += 1
            if args.limit and n_done >= args.limit:
                print(f"stopped at limit={args.limit}")
                return

    print(f"wrote {n_done} simulations to {out_dir}")


if __name__ == "__main__":
    main()
