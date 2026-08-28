# mas-semantic-collapse

Phase-1 research framework: label Reddit threads as semantic **converge / non-converge**, then run an **AutoGen** multi-agent comment-section baseline (DeepSeek). Human threads are also described as a discussion cloud with five order parameters \((m,\sigma,d,k,\chi)\).

Reference measurement family: Kong et al., *Multi-LLM Systems Exhibit Robust Semantic Collapse* (arXiv:2605.17193). This repo does **not** copy a human binary label from that paper (they only provide a continuous human reference); we define q80/q20 labels on our corpus.

**Private repository.** To let someone run the GPU jobs, add them as a collaborator (Settings → Collaborators). They do not need the data file to be in git.

**Run on a GPU machine (advisor):** see [`docs/RUN_ON_GPU.md`](docs/RUN_ON_GPU.md).

## Setup

```bash
cd mas-semantic-collapse
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # then set keys / THREADS_PATH if needed
```

Data file (not in git): `../threads_2026-01_to_2026-05.jsonl.zst`, or set `THREADS_PATH`.

## Pipeline

1. **Label humans** (embedding: interim `bge-m3`; swap later to `text-embedding-3-large`):

```bash
python -m scripts.label_threads --config configs/default.yaml
# faster smoke (NOT for paper numbers): set embedding.backend: hashing in config
python -m scripts.label_threads --config configs/default.yaml --limit 50 --no-vendi
```

2. **Sample pilot set** (50/class) and simulate:

```bash
python -m scripts.run_baseline --config configs/default.yaml --structure flat
python -m scripts.run_baseline --config configs/default.yaml --structure weak_tree
```

3. **Score simulations** with the same metrics:

```bash
python -m scripts.score_sims --config configs/default.yaml
```

## Order parameters of the discussion cloud (m, sigma, d, k, chi)

Beyond the single radial score `S_end`, each window is treated as a cloud of
comment vectors and described by five order parameters. Embedding cost is the
same as the mean-pool labelling run (every comment is embedded once); the extra
quantities are arithmetic on vectors that are already in memory.

GPU chunked run: [`docs/RUN_ON_GPU.md`](docs/RUN_ON_GPU.md).

```bash
# code smoke, seconds, hashing backend (NOT for reported numbers)
python scripts/extract_order_params.py --backend hashing --limit 15 --tag smoke

# CPU pilot
python scripts/extract_order_params.py --limit 200 --tag pilot200

# GPU, resumable in chunks
python scripts/extract_order_params.py --skip 0    --limit 500 --device cuda --batch-size 32 --tag gpu_000_500
python scripts/extract_order_params.py --skip 500  --limit 500 --device cuda --batch-size 32 --tag gpu_500_1000

# figures + correlations with S_end
python scripts/plot_order_params.py --tag pilot200
python scripts/classify_step_motions.py --tag pilot200
```

Definitions and the reasons behind them:

| Symbol | Quantity | How it is computed |
| --- | --- | --- |
| `m` | cloud center | L2-normalized mean of the window's comment vectors; reported as `cos(m_1, m_t)` and as the adjacent step `cos(m_{t-1}, m_t)` |
| `sigma` | dispersion | mean cosine distance from each comment to `m` |
| `d` | effective dimension | eigenvalue participation ratio of the cosine kernel |
| `k` | modality | eigengap on a self-tuning (Zelnik-Manor) affinity, **plus** a continuous `bimodality` = silhouette of the Fiedler-vector bipartition |
| `chi` | turnover | identity-free: adjacent centroid step, plus a Hungarian matching cost between the two centred clouds |

- The time grid and the estimation support are decoupled: `m` uses
  non-overlapping windows of 10 comments (comparable with the existing
  `sims_to_first` curves), while `sigma / d / k` use a 30-comment support window
  centred on the same block, because ten points cannot support a dispersion,
  dimension or mode-count estimate.
- `chi` cannot use author identity: adjacent windows share a median of 12.5% of
  their authors, and 37.7% of adjacent pairs share none.
- Integer `k` is 1 in essentially every window, so `bimodality` is the reported
  fission signal; the integer is kept for reference only.
- Comment vectors are not persisted. Only the per-window scalar trajectories are written.

Method notes (HTML is self-contained):

- `docs/protocol_for_shiyang.md` — three decisions pending advisor sign-off
- `docs/order_params_pilot200.md` — five coordinates, \(n=200\)
- `docs/step_motions_pilot200.md` — adjacent-step motion labels
- `docs/motion_shapes_concat200.md` — \(m\)-trajectory shapes on concat 200

## Locked Phase-1 decisions

- Linearize: top-level first, children by time
- Windows: 10 comments
- Labels: A = S_end q80/q20 (middle dropped); B = slope auxiliary
- Protocols: (1) post-only and (2) post + first 10 human comments
- Agents: 3 homogeneous DeepSeek, minimal personas Commenter-A/B/C
- Context: full history
- Structure: flat and weak_tree (switchable)
- Order: round_robin main; random optional
- Pilot: 50 threads/class, max 100 comments/thread
- Metrics (all computed): S_end, T_tau, Vendi, slope, lexical
- Embedding: bge-m3 now (CPU default in config; override `--device cuda` on a GPU machine); interface reserved for text-embedding-3-large
- Labeling rollout: pilot 200 with bge_m3 → full corpus with `--no-vendi` → Vendi later

## Security

Never commit `.env` or API keys. Rotate any key that was pasted into chat.
