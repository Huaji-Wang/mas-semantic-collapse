# Pipeline: continuous semantic-motion mixtures

The default pipeline measures five semantic trajectories, assigns ten continuous motion weights to each trajectory, and compares human and simulated discussions within shared title topics. The versioned standard is [motion-mixture-v1](motion_weights.md).

```text
Human corpus OR paired replay + saved simulation comments
    Stage 1: embeddings and five trajectories
outputs/order_params/order_params_{tag}.jsonl
    Stage 2: ten weights per trajectory, sum = 1
outputs/labels/thread_motion_weights_{tag}.jsonl
outputs/metrics/thread_motion_weights_{tag}.json
    Stage 3: one title-topic assignment per original thread
outputs/title_clusters/thread_topics_{tag}.jsonl
outputs/title_clusters/motion_weights_{tag}.json
outputs/title_clusters/motion_weights_{tag}.html
```

## Existing measurements

```bash
python scripts/run_pipeline.py --from 2 --tag chik200
```

The default now starts at stage 2, because legacy stage-2 hard labels cannot feed the weighted stage 3. `--to 2` stops after scoring. `--from 3` requires the new weight file. No generation API is called by this pipeline; selecting OpenAI embeddings does incur embedding requests.

For multiple compatible measurement files or custom paths, run the scorer directly:

```bash
python scripts/classify_motion_weights.py --tag combined --input path/to/first.jsonl path/to/second.jsonl
```

Thread IDs must be unique within each side across those files. Human-only trajectories are supported; if simulation rows are present, complete human/simulation pairs are required.

## Full-position scale replay

The existing official-DeepSeek run stores one simulated thread per file in `outputs/sims/scale5000/low_deepseek/`. Its corresponding human comments are in `outputs/scale5000/replay.jsonl`. The first simulation line is metadata, followed by comment records. These files remain local and are excluded from Git.

Audit these inputs without loading an embedding model:

```bash
python scripts/extract_scale_order_params.py --audit-only
```

Measure and compare the saved comments (potentially a long CPU job):

```bash
python scripts/run_pipeline.py --from 1 --source scale-replay --tag scale5000_low_deepseek_centered --device cpu
```

Prerequisites: dependencies installed, the two local input locations present, access to the configured embedding model, and the matching reference mean at `reference/corpus_mean.npy`. The mean is a local derived artifact: transfer the matching mean or regenerate it with `scripts/corpus_mean.py`; see [centring](persisted_embeddings_and_centring.md). A mean from a different embedding model must not be substituted. Paths can be overridden with `--replay`, `--sims-dir`, and `--center-mean`.

Stage 1 checks paired thread IDs, comment count, speaking positions, author/alias/comment IDs, nonempty output, generation errors, and valid past-history indices. It records truncation counts. Invalid pairs abort measurement by default; `--skip-invalid` explicitly excludes them and records each reason in `outputs/metrics/replay_audit_{tag}.json`. This structural audit does not establish semantic quality or verify adherence to persona instructions. At least two measurement windows are required.

Per-thread measurement checkpoints resume after interruption. They are reused only when source text, embedding settings, centring, and measurement-code fingerprints match. Model weights must also be kept fixed; the fingerprint does not download or hash model checkpoints. Avoid concurrent writers using the same tag. `--limit N` uses the first N replay records for a pilot; use a separate tag to preserve a full-run output.

The full-scale input count is not a completed measurement result. The earlier 311-pair report and this scale run are distinct cohorts. By default, feature scales are fitted only on the current cohort's human trajectories. To compare multiple arms, reuse one saved summary with `--scales-file outputs/metrics/thread_motion_weights_REFERENCE.json`; keep embedding, centring and window settings identical. Do not interpret absolute weight differences across independently fitted scales as a controlled comparison.

## Outputs and interpretation

Each trajectory retains all ten weights. `dominant_motion` is a convenience label only; all default summaries use complete vectors. Overall and per-topic outputs include mean weights, fractional thread-equivalent mass, and paired simulation-minus-human differences in percentage points. They are thread-level descriptions, not counts of individual people.

Titles are embedded once per original thread, so the two sides receive exactly the same topic. Automatic clusters and keyword labels are exploratory topics, not the manually annotated topic codebook in the earlier report. Empty/removed titles form an explicit `unusable` group. Fractional weights are not passed to the legacy chi-square/Fisher hard-count tests. Inferential uncertainty and topic validation remain separate research steps.

The default title backend is BGE-M3; `--title-backend openai` changes only stage 3. `--title-backend hashing` is for smoke tests, not research. Topic summaries and generated HTML stay under ignored `outputs/`.

## Legacy reproduction and generation

The old nine-class argmax pipeline is available only through `--motion-method discrete` with corpus input, or its original scripts. Its files and count-based tests are legacy results, not ten-weight results. The historical demo instructions below describe that separate generation workflow.

The scale source files have distinct roles: `prepare_scale5000.py` prepares a six-arm eligibility manifest; `run_scale_arms.py` uses `configs/scale5000.yaml`; `run_scale_flash.py` prepares/runs the full-position official-DeepSeek protocol. For the existing run, saved generation metadata is authoritative. None needs to be rerun merely to change classification.

---

# 旁支：人设 LLM 模拟 demo

和上面三级平行，不共用 tag。做的是**同一个帖、同一批人、同样的发言位置**，把真人换成按其历史写成的 LLM 角色，再用同一套五个序参量量两条线。

数据来自另一份 dump（`conditioned_threads.jsonl.zst`，5000 帖、104 个版块各 48 帖、2026-01 到 05）。它的结构和主 dump 一样是 `{post, comments}`，额外多了 `participants`。注意 `participants[author].h` 是历史**条数**不是正文，`cond` 就是 `h >= 10`。每人最多 300 条评论正文在独立的 `user_history.jsonl-001.zst` 里（`USER_HISTORY_PATH`）。48 帖回复者里约 862 人有这份记录；人设优先用它，并排除这些人在 48 个 demo 帖上的发言。

路径用环境变量给：

```bash
CONDITIONED_THREADS_PATH=.../conditioned_threads_2026/conditioned_threads.jsonl.zst
AUTHOR_THREADS_CSV=.../conditioned_threads_2026/user_history/author_threads.csv
USER_HISTORY_PATH=.../user_history.jsonl-001.zst
```

逐步，每步只认上一步的文件：

| 步 | 脚本 | 输入 | 输出 |
|---|---|---|---|
| A | `scripts/build_persona_demo.py --no-cards` | 上面 dump 路径 | `outputs/personas/{tag}/cast_and_history.json`（dump 跨帖评论，班底用） |
| A1 | `scripts/extract_user_history.py` | `USER_HISTORY_PATH` + 48 帖名单 | `outputs/personas/{tag}/user_history.jsonl`（精简记录，demo 帖已剔除） |
| A2 | `scripts/write_personas.py --from-user-history` | A1 | `cards.json`，以及 `prompts/{alias}_{author}.txt` |
| B | `scripts/run_persona_demo.py` | A2 | `outputs/sims/{tag}/{post_id}.json` |
| C | `scripts/score_persona_demo.py` | B | `outputs/order_params/order_params_{tag}.jsonl` |
| D | `scripts/export_persona_demo_html.py` | B、C | `docs/persona_demo.html` |

```bash
python scripts/build_persona_demo.py --no-cards
python scripts/extract_user_history.py            # 扫 10GB zst，抽出 48 帖回复者
python scripts/write_personas.py --from-user-history
python scripts/write_personas.py --from-user-history --limit 3   # 先写 3 个看卡
python scripts/run_persona_demo.py --limit 4
python scripts/run_persona_demo.py
python scripts/score_persona_demo.py
python scripts/export_persona_demo_html.py
```

协议里几条定死的口径：

- **班底**：本帖发言 ≥3 条、且在其它采样帖里出现过 ≥2 次的人，按发言量取前 5。两条线都只保留这 5 个人的评论，所以位置、发言人、条数完全一致。
- **人设料**：目标帖本身从不进入人设，也从不给 agent 看。优先用 `user_history.jsonl`（每人最多约 300 条爬取记录）。`write_personas.py --from-user-history` 读这些记录，代码统计条数、长度、版块，DeepSeek 只写立场、语气、习惯三项，再钉 2–3 条原文，拼成 agent 的 system prompt。没有这份记录的人仍可用 dump 跨帖评论（`cast_and_history.json`）。
- **生成**：给原帖标题正文，给已生成的前 k−1 条，不给任何真人评论，不给回复树提示（扁平）。长度按各人自己的历史中位数下指示，token 上限按其 p90 换算。
- **模型**：`--provider deepseek`（默认）或 `--provider openai --model gpt-5.6-luna`。后者要 `OPENAI_API_KEY`。
- **运动打分尺度**取自 200 帖那次运行（`--scale-from`），不由这 3 个帖自己定，否则类边界会被 demo 自己改掉。
- 真名只留在 `outputs/`（已 gitignore），`docs/persona_demo.html` 里只有 `Agent-N`。

`configs/default.yaml` 的 `persona_demo:` 段是全部旋钮（选哪几个帖、班底大小、取料上限、长度换算）。

已知局限写在页面顶部，最要紧的一条是**长度混淆**：模拟评论平均只有真人一半长，所以 σ 和 d 偏低有可能只是文本短，要把真人截到同长再嵌一遍才能排除。
