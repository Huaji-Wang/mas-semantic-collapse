# mas-semantic-collapse

Phase-1 research framework: label Reddit threads as semantic **converge / non-converge**, then run an **AutoGen** multi-agent comment-section baseline (DeepSeek).

Reference measurement family: Kong et al., *Multi-LLM Systems Exhibit Robust Semantic Collapse* (arXiv:2605.17193). This repo does **not** copy a human binary label from that paper (they only provide a continuous human reference); we define q70/q30 labels on our corpus.

## Setup

```bash
cd mas-semantic-collapse
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # then set DEEPSEEK_API_KEY
```

Data file (not in git): `../threads_2026-01_to_2026-05.jsonl.zst`

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

## Locked Phase-1 decisions

- Linearize: top-level first, children by time
- Windows: 10 comments
- Labels: A = S_end q70/q30 (middle dropped); B = slope auxiliary
- Protocols: (1) post-only and (2) post + first 10 human comments
- Agents: 3 homogeneous DeepSeek, minimal personas Commenter-A/B/C
- Context: full history
- Structure: flat and weak_tree (switchable)
- Order: round_robin main; random optional
- Pilot: 50 threads/class, max 100 comments/thread
- Metrics (all computed): S_end, T_tau, Vendi, slope, lexical
- Embedding: bge-m3 now (CUDA on RTX 4070); interface reserved for text-embedding-3-large
- Labeling rollout: pilot 200 with bge_m3 → full corpus with `--no-vendi` → Vendi later

## Security

Never commit `.env` or API keys. Rotate any key that was pasted into chat.
