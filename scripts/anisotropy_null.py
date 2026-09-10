"""
anisotropy_null.py  (added with the embedding-persistence patch)
==================
Null check for embedding-space anisotropy, run on embeddings persisted by the
patched extract_order_params.py (--save-embeddings).

Why: the five discussion-cloud coordinates (sigma, d, k, chi) are geometric.
If the embedder puts all comments in a narrow cone, unrelated comments still
look moderately similar: sigma/d get squeezed, k collapses to 1, chi inflates.
Huaji's pilot shows k==1 in 99.2% of windows and chi median ~0.83 everywhere --
exactly what a cone would produce. This measures the cone directly.

PREDICTION (written 2026-09-10, BEFORE seeing any result):
  cosine similarity of random comment pairs drawn from DIFFERENT threads under
  bge-m3 will center around 0.3-0.5, not near 0.
  - If median >= ~0.3 : cone confirmed -> chi/k need a null-corrected reading
    (compare per-thread values to this baseline, or centre/whiten the space).
  - If median near 0 (say |median| < 0.1): geometry is fine, drop the concern.
  What would change the prediction: nothing observed yet; this is the planned
  measurement. Within-thread pairs are reported alongside as the contrast.

Usage
-----
    python anisotropy_null.py --emb-dir outputs/embeddings/smoke50 --n-pairs 2000
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--n-pairs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="json output (default <emb-dir>/anisotropy_null.json)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.emb_dir, "*.npz")))
    if len(files) < 2:
        raise SystemExit(f"need >=2 persisted threads in {args.emb_dir}, found {len(files)}")
    clouds = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        e = np.asarray(z["embs"], dtype=np.float64)
        # vectors are L2-normalised by the embedder; re-normalise defensively
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        clouds.append(e)
    rng = np.random.default_rng(args.seed)
    nt = len(clouds)

    # cross-thread random pairs: one random comment from each of two different threads
    cross = np.empty(args.n_pairs)
    for i in range(args.n_pairs):
        a, b = rng.choice(nt, size=2, replace=False)
        cross[i] = clouds[a][rng.integers(len(clouds[a]))] @ clouds[b][rng.integers(len(clouds[b]))]
    # within-thread random pairs (contrast): two different comments from the same thread
    within = np.empty(args.n_pairs)
    for i in range(args.n_pairs):
        a = rng.integers(nt)
        n = len(clouds[a])
        j, k = rng.choice(n, size=2, replace=False)
        within[i] = clouds[a][j] @ clouds[a][k]

    def q(x):
        return {k_: float(np.quantile(x, v)) for k_, v in
                (("p05", .05), ("p25", .25), ("median", .5), ("p75", .75), ("p95", .95))}

    res = {
        "n_threads": nt,
        "n_comments_total": int(sum(len(c) for c in clouds)),
        "n_pairs": args.n_pairs,
        "cross_thread_cosine": {**q(cross), "mean": float(cross.mean()),
                                "frac_gt_0.3": float((cross > 0.3).mean())},
        "within_thread_cosine": {**q(within), "mean": float(within.mean())},
        "prediction": "cross-thread median in 0.3-0.5 (cone); near 0 would falsify",
        "verdict": ("CONE CONFIRMED: unrelated comments look similar; read chi/k against this baseline"
                    if np.median(cross) >= 0.3 else
                    "GEOMETRY OK: cross-thread similarity near 0; anisotropy concern dropped"
                    if abs(np.median(cross)) < 0.1 else
                    "INTERMEDIATE: mild cone; report baseline alongside chi/k"),
    }
    out = args.out or os.path.join(args.emb_dir, "anisotropy_null.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
