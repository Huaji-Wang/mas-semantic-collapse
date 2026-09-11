# Persisted embeddings and shared-mean centring

Two flag-gated additions to `scripts/extract_order_params.py`. With no flags the
script behaves exactly as before.

## Why

The five coordinates (m, σ, d, k, χ) are geometric quantities on the comment
cloud, but bge-m3's raw space is a cone: on the 4,837-thread Reddit run,
random comment pairs from **different** threads have cosine median **0.397**
(92% above 0.30), while pairs from the **same** thread sit at only **0.479**.
The whole "same discussion?" signal is a gap of ~0.08 on a floor of ~0.40.

That floor is inside the formal k rule: the 0.55 cosine-distance cutoff
(cosine ≥ 0.45) already merges **23%** of *unrelated* pairs, and the formal k is
pinned at 1 in 27% of windows in raw space. See
`outputs/embeddings/full/anisotropy_null.json` and `scripts/anisotropy_null.py`.

Subtracting the corpus mean (the cone axis) and re-normalising fixes the floor:
cross-thread cosine → −0.005, k-rule false merges → 0.1%, within-vs-cross gap
×1.55. On the paired 4,837-thread run (medians): σ 0.28 → 0.56, effective
dimension 3.4 → 11.7, formal k pinned-at-1 27% → 8%, S_end 0.86 → 0.49. It does
**not** change the eigengap k (99% pinned either way — dense-affinity
saturation), the legacy Hungarian χ (0.83 either way — it already centres per
window), the polar χ, or the bimodality proxy; those need a different fix. The
persisted vectors make the recomputation a seconds-long job instead of a
re-embed.

## Flags

| flag | effect |
|---|---|
| `--save-embeddings` | write each thread's **raw** vectors to `--emb-dir/<post_id>.npz` (`embs`, `comment_ids`, `embedder`) |
| `--reuse-embeddings` | load a thread's `.npz` instead of embedding it; the model is built lazily, so a run where every thread is persisted needs no model and no GPU |
| `--emb-dir DIR` | where the `.npz` live (default `outputs/embeddings/<tag>`) |
| `--center-mean FILE.npy` | subtract this reference vector from every comment vector and re-normalise **before** computing the coordinates; persisted vectors stay raw |

`--reuse-embeddings` also makes a run resumable: resubmit the same command and
only the missing threads are embedded.

## The one rule for centring

Use **one shared reference mean on both sides** — the human Reddit corpus mean
(`outputs/embeddings/full/corpus_mean.npy`, 628,753 comments) applied unchanged
to the AutoGen threads too. Centring each side on its own mean would remove the
human-vs-agent difference the comparison is meant to measure.

Thresholds set on raw geometry (formal-k cutoff 0.55, S_end q80/q20 label cuts)
do not transfer to the centred space and need re-deriving there; χ is
unaffected by centring, so its 0.92 gate can stay.

## Reproduce

```bash
# embed once, persist raw vectors (GPU)
python scripts/extract_order_params.py --limit 0 --device cuda --allow-laptop-cuda \
    --batch-size 64 --save-embeddings --reuse-embeddings --tag full

# cone measurement + reference mean
python scripts/anisotropy_null.py --emb-dir outputs/embeddings/full --n-pairs 5000

# centred coordinates from the persisted vectors (CPU, minutes)
python scripts/extract_order_params.py --limit 0 --reuse-embeddings \
    --emb-dir outputs/embeddings/full \
    --center-mean outputs/embeddings/full/corpus_mean.npy --tag full_centered
```

Outputs: `outputs/order_params/order_params_full.jsonl` (raw geometry, comparable
to the earlier pilot) and `order_params_full_centered.jsonl` (centred). Each row
records `emb_path` and `center_mean` so the two are never confused.
