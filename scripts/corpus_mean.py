"""Compute the corpus mean of persisted comment embeddings (the cone axis).

The result is the shared reference vector for ``--center-mean``. Compute it ONCE
on the human Reddit corpus and apply the same file, unchanged, to the simulated
(AutoGen) side -- centring each side on its own mean would remove the very
human-vs-agent difference the comparison measures.

Mean is accumulated in float64 over every row of every ``<post_id>.npz`` written
by ``extract_order_params.py --save-embeddings``, then stored as float32.

Usage
-----
    python scripts/corpus_mean.py --emb-dir outputs/embeddings/full \
        --out reference/corpus_mean.npy
"""
import argparse
import glob
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True, help="dir of <post_id>.npz from --save-embeddings")
    ap.add_argument("--out", required=True, help="output .npy (float32, shape (dim,))")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.emb_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz in {args.emb_dir}")
    total = None
    n = 0
    for f in files:
        e = np.load(f, allow_pickle=False)["embs"].astype(np.float64)
        total = e.sum(axis=0) if total is None else total + e.sum(axis=0)
        n += len(e)
    mu = (total / n).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, mu)
    print(f"{args.out}: mean of {n:,} comment vectors from {len(files):,} threads; "
          f"dim={mu.shape[0]}; cone-axis strength ||mu||={float(np.linalg.norm(mu)):.3f}")


if __name__ == "__main__":
    main()
