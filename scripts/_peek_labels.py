import json
import statistics
from pathlib import Path

p = Path(r"D:\自学\iclr\mas-semantic-collapse\outputs\labels\thread_labels.jsonl")
rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
s = [r["s_end"] for r in rows]
ss = sorted(s)
print(
    "s_end min/p25/p50/p75/max",
    round(min(s), 3),
    round(ss[49], 3),
    round(ss[99], 3),
    round(ss[149], 3),
    round(max(s), 3),
)
print("mean/std", round(statistics.mean(s), 3), round(statistics.pstdev(s), 3))
for lab in ["converge", "non_converge"]:
    xs = [r for r in rows if r["label_A"] == lab]
    xs = sorted(xs, key=lambda r: -r["s_end"] if lab == "converge" else r["s_end"])
    print(f"\n=== {lab} n={len(xs)} ===")
    for r in xs[:5]:
        title = (r.get("title") or "")[:70]
        print(
            f"{r['s_end']:.3f} | slope={r['slope']:+.4f} | n={r['n_comments']} | "
            f"r/{r['subreddit']} | {title}"
        )
