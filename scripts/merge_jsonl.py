"""Concatenate chunked jsonl runs (order-params or thread scores) in skip order.

Example:
    python scripts/merge_jsonl.py outputs/order_params/order_params_gpu_000_500.jsonl \\
        outputs/order_params/order_params_gpu_500_1000.jsonl \\
        -o outputs/order_params/order_params_full.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge jsonl chunks; drop duplicate post_id (first wins)")
    ap.add_argument("inputs", nargs="+", help="jsonl files in skip order")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    seen: set[str] = set()
    n_in = n_out = n_dup = 0
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as w:
        for path in args.inputs:
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                n_in += 1
                obj = json.loads(line)
                pid = str(obj.get("post_id") or "")
                if pid in seen:
                    n_dup += 1
                    continue
                if pid:
                    seen.add(pid)
                w.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_out += 1
    print(f"read={n_in} wrote={n_out} duplicates_dropped={n_dup} -> {out}")


if __name__ == "__main__":
    main()
