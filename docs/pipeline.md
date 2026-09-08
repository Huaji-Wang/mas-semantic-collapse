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
