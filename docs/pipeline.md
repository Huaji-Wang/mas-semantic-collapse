# 三级 pipeline：输入和输出

一条命令按级跑。每一级只认上一级写好的文件，不把标题向量写进评论云。

```text
Reddit 帖（zst）
    │  第 1 级  嵌评论，算五条曲线
    ▼
outputs/order_params/order_params_{tag}.jsonl
    │  第 2 级  整帖打九类运动标签
    ▼
outputs/labels/thread_motions_{tag}.jsonl
    │  第 3 级  只嵌标题，聚类，再对上九类
    ▼
outputs/title_clusters/  和  outputs/metrics/title_cluster_vs_motion_{tag}.json
```

本机默认从第 3 级起：chik200 的第 1、2 级已经有了，不要重嵌评论。

```bash
python scripts/run_pipeline.py --from 3 --tag chik200
```

换标题编码器（不重跑第 1、2 级）：

```bash
python scripts/run_pipeline.py --from 3 --tag chik200 --title-backend openai
```

---

## 第 1 级　评论 → 五条曲线

| | 路径 |
|---|---|
| **输入** | 环境变量 `THREADS_PATH`，或 `configs/default.yaml` 里的 `data.threads_path`（zst 评论区） |
| **输出** | `outputs/order_params/order_params_{tag}.jsonl` 每帖一行：`post_id`、`title`、`sigma`、`eff_dim`、`k_modes`、`chi`、`sims_to_first` 等 |
| **脚本** | `scripts/extract_order_params.py` |
| **本机** | `--device cpu`。不要在这台显示用的 RTX 4070 上开 CUDA。 |

chik200 已经跑完，本轮不要再跑这一级。

---

## 第 2 级　曲线 → 九类运动

| | 路径 |
|---|---|
| **输入** | `outputs/order_params/order_params_{tag}.jsonl` |
| **输出** | `outputs/labels/thread_motions_{tag}.jsonl` 每帖一行：`post_id`、`title`、`motion`、九行 `scores` |
| | `outputs/metrics/thread_motions_{tag}_stats.json` 各类帖数 |
| **脚本** | `scripts/classify_thread_motions.py` |

一条帖一个类。成丝不打分。

---

## 第 3 级　标题 → 题目簇 × 运动类

标题和评论**不是**同一团点。这里只嵌标题。

| | 路径 |
|---|---|
| **输入** | `outputs/labels/thread_motions_{tag}.jsonl`（用里面的 `title` 和 `motion`） |
| **输出（每帖）** | `outputs/title_clusters/title_clusters_{tag}.jsonl`：`post_id`、标题、`cluster_id`、`motion` |
| **输出（每簇）** | `outputs/title_clusters/clusters_{tag}.json`：簇大小、示例标题、突出词、`label_auto`、`definition_auto` |
| **输出（对照）** | `outputs/metrics/title_cluster_vs_motion_{tag}.json`：计数表、独立性检验、校正后的格子偏离 |
| **输出（给人看）** | `docs/title_clusters_{tag}.html` |
| **脚本** | `scripts/cluster_titles.py` |
| **编码器** | `configs/default.yaml` 的 `title_embedding`（默认 `bge_m3`）。`--title-backend` 可换成 `openai` / `hashing`。 |

空标题和 `[ Removed by moderator ]` 的 `cluster_id` 是 `unusable`，不参加聚类和检验。

默认切法是球形 k-means，再把小于 `--min-cluster-size`（默认 15）的簇并进最近簇。旧的平均连接层次聚类：`--cluster-method average_linkage`。

---

## 从哪一级接着跑

| `--from` | 做什么 |
|---|---|
| `3` | 只跑标题聚类（默认，本轮用这个） |
| `2` | 第 2 级 + 第 3 级（曲线已有、要重打运动标签时） |
| `1` | 三级全跑。会重新嵌全部评论，本机 CPU 要数小时。 |

入口：`scripts/run_pipeline.py`。

---

# 旁支：人设 LLM 模拟 demo

和上面三级平行，不共用 tag。做的是**同一个帖、同一批人、同样的发言位置**，把真人换成按其历史写成的 LLM 角色，再用同一套五个序参量量两条线。

数据来自另一份 dump（`conditioned_threads.jsonl.zst`，5000 帖、104 个版块各 48 帖、2026-01 到 05）。它的结构和主 dump 一样是 `{post, comments}`，额外多了 `participants`。注意 `participants[author].h` 是历史**条数**不是正文，`cond` 就是 `h >= 10`。人设的正文只能来自该账号在**其它采样帖**里的真实评论。

两个路径用环境变量给：

```bash
CONDITIONED_THREADS_PATH=.../conditioned_threads_2026/conditioned_threads.jsonl.zst
AUTHOR_THREADS_CSV=.../conditioned_threads_2026/user_history/author_threads.csv
```

四步，每步只认上一步的文件：

| 步 | 脚本 | 输入 | 输出 |
|---|---|---|---|
| A | `scripts/build_persona_demo.py` | 上面两个路径 | `outputs/personas/persona_demo/cast_and_history.json`、`cards.json` |
| B | `scripts/run_persona_demo.py` | A | `outputs/sims/persona_demo/{post_id}.json` |
| C | `scripts/score_persona_demo.py` | B | `outputs/order_params/order_params_persona_demo.jsonl`、`outputs/labels/thread_motions_persona_demo.jsonl` |
| D | `scripts/export_persona_demo_html.py` | B、C | `docs/persona_demo.html` |

```bash
python scripts/build_persona_demo.py                 # 扫两遍 dump，约 40 秒；卡有缓存
python scripts/run_persona_demo.py --limit 4         # 先小跑看看
python scripts/run_persona_demo.py                   # 全量 164 条，约 2 分钟，可断点续跑
python scripts/score_persona_demo.py                 # CPU 嵌入，约 10 分钟
python scripts/export_persona_demo_html.py
```

协议里几条定死的口径：

- **班底**：本帖发言 ≥3 条、且在其它采样帖里出现过 ≥2 次的人，按发言量取前 5。两条线都只保留这 5 个人的评论，所以位置、发言人、条数完全一致。
- **人设料**：目标帖本身从不进入人设，也从不给 agent 看。历史条数、长度分位、版块分布由代码统计，模型只写立场、语气、习惯三项，免得它编数字。
- **生成**：给原帖标题正文，给已生成的前 k−1 条，不给任何真人评论，不给回复树提示（扁平）。长度按各人自己的历史中位数下指示，token 上限按其 p90 换算。
- **模型**：`--provider deepseek`（默认）或 `--provider openai --model gpt-5.6-luna`。后者要 `OPENAI_API_KEY`。
- **运动打分尺度**取自 200 帖那次运行（`--scale-from`），不由这 3 个帖自己定，否则类边界会被 demo 自己改掉。
- 真名只留在 `outputs/`（已 gitignore），`docs/persona_demo.html` 里只有 `Agent-N`。

`configs/default.yaml` 的 `persona_demo:` 段是全部旋钮（选哪几个帖、班底大小、取料上限、长度换算）。

已知局限写在页面顶部，最要紧的一条是**长度混淆**：模拟评论平均只有真人一半长，所以 σ 和 d 偏低有可能只是文本短，要把真人截到同长再嵌一遍才能排除。
