# reference/corpus_mean.npy

Shared reference mean for `extract_order_params.py --center-mean`. float32, shape (1024,).

Provenance: mean of all **628,753** bge-m3 (`BAAI/bge-m3`, fp32, max_seq 8192)
comment vectors from the **4,837** eligible conditioned Reddit threads
(2026-01..2026-05; 50-1000 non-deleted comments), full run 2026-09-10.
Cone-axis strength ‖μ‖ = 0.631. Regenerate with
`python scripts/corpus_mean.py --emb-dir outputs/embeddings/full --out reference/corpus_mean.npy`
(bit-identical to this file).

Apply this SAME file to both the human and the simulated side.
