# mas-semantic-collapse

衡量一场讨论在语义空间里如何演化：人类 Reddit 评论区是对照，多智能体互相接话是实验对象。本仓库提供嵌入、打分、讨论云五个序参量，以及（可选的）AutoGen 模拟评论区。

测量族参考 Kong et al., [*Multi-LLM Systems Exhibit Robust Semantic Collapse*](https://arxiv.org/abs/2605.17193)。本文不沿用该文的人类二分类标签；人类侧在本语料内按 \(S_{end}\) 的 q80/q20 相对切开，并用 \((m,\sigma,d,k,\chi)\) 描述讨论云，而不是只用一个终点分数。

私有仓库。需要别人跑数时，在 GitHub **Settings → Collaborators** 添加协作者。原始 Reddit dump 与 `outputs/*.jsonl` **不在 git 里**。

---

## 这是什么

一条讨论被看成语义空间中的一团点（每条评论一个 embedding）。短时窗上的点构成讨论云。云的状态用五个序参量描述：

| 符号 | 含义 | 本仓库中的估计 |
| --- | --- | --- |
| \(m\) | 云心：这批发言的平均语义位置 | 非重叠 10 条评论的归一化均值 |
| \(\sigma\) | 散布 | 点到云心的平均余弦距离 |
| \(d\) | 有效维数 | 余弦核特征值的参与比 |
| \(k\) | 模态（几坨） | 整数特征间隙几乎恒为 1；可操作信号是二分强度 |
| \(\chi\) | 换位 | 身份无关：去中心后两点云的匹配代价 |

\(S_{end}\) 只是 \(m(t)\) 相对第一窗的径向终点。它不能单独回答云是否散开、降维、裂成阵营，或内部构型是否更换。十条运动（Translation、Condensation、Orbit 等）是这些坐标的**相邻步差分签名**，不是十种整帖类别。

当前阶段（Phase-1）做两件事的准备：（1）在人类帖上把上述量算出来；（2）搭好多智能体评论区基线，便于之后用**同一套估计**对照，而不是只比两个 \(S_{end}\)。

---

## 作用是什么

- **人类侧打分**：评论级 `bge-m3` 嵌入（接口预留 `text-embedding-3-large`），得到 \(S_{end}\) 等曲线，并可按 q80/q20 打相对标签。
- **讨论云**：同一批评论向量上估计五个序参量；可再做相邻步运动标签（启发式，口径待确认）。
- **多智能体基线**：AutoGen + DeepSeek，flat / weak_tree，用同一套指标给模拟评论区打分。
- **给算力机器跑**：分块 `--skip` / `--limit` / `--device cuda`，合并 jsonl 后在本地作图。

口径尚未全部锁定（窗宽与支撑拆开、身份无关 \(\chi\)、步级互斥标签）。全量约 1900 条嵌入建议在 [`docs/protocol_for_shiyang.md`](docs/protocol_for_shiyang.md) 确认后再开。小样本 `--limit` 随时可用来验证环境。

---

## 怎么跑

### 1. 安装

```bash
git clone git@github.com:Huaji-Wang/mas-semantic-collapse.git
cd mas-semantic-collapse
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

仅跑嵌入时不必填 DeepSeek。可选：`HF_TOKEN`（下载 `BAAI/bge-m3`）、`THREADS_PATH`（数据不在默认位置时）。

数据文件（不进 git）：默认 `../threads_2026-01_to_2026-05.jsonl.zst`。别处则：

```bash
export THREADS_PATH=/absolute/path/threads_2026-01_to_2026-05.jsonl.zst
```

合格线程：评论数 ≥ 50 且 ≤ 1000。

检查 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### 2. 先确认代码能跑（秒级，结果不可写进报告）

```bash
python scripts/extract_order_params.py --backend hashing --limit 15 --tag smoke
```

### 3. 主任务：五个序参量（算力机）

每个合格线程只嵌入一次；\(\sigma,d,k,\chi\) 在内存点云上计算，评论向量不落盘。详细分块、断点与合并见 [`docs/RUN_ON_GPU.md`](docs/RUN_ON_GPU.md)。

```bash
python scripts/extract_order_params.py --skip 0 --limit 500 --device cuda --batch-size 32 --tag gpu_000_500
# 下一块：--skip 500 --limit 500 --tag gpu_500_1000
# --limit 0 表示 skip 之后全部跑完

python scripts/merge_jsonl.py \
  outputs/order_params/order_params_gpu_000_500.jsonl \
  outputs/order_params/order_params_gpu_500_1000.jsonl \
  -o outputs/order_params/order_params_full.jsonl
```

本地作图（不必 GPU）：

```bash
python scripts/plot_order_params.py --jsonl outputs/order_params/order_params_full.jsonl --tag full
python scripts/classify_step_motions.py --jsonl outputs/order_params/order_params_full.jsonl --tag full
```

请回传 `outputs/order_params/` 下的 jsonl 与 `*_summary.json`，不要回传 Hugging Face 模型缓存。

### 4. 可选：人类 \(S_{end}\) 打分、多智能体模拟

```bash
python -m scripts.label_threads --config configs/default.yaml --device cuda --no-vendi --limit 50
python -m scripts.run_baseline --config configs/default.yaml --structure flat
python -m scripts.score_sims --config configs/default.yaml
```

模拟需要 `.env` 里的 `DEEPSEEK_API_KEY`。不要把 concat 200 的形状阈值套到 mean-pool 分数上。

---

## 文档（HTML 已嵌图，浏览器打开即可）

| 文件 | 内容 |
| --- | --- |
| [`docs/RUN_ON_GPU.md`](docs/RUN_ON_GPU.md) | 算力机逐步命令 |
| [`docs/protocol_for_shiyang.md`](docs/protocol_for_shiyang.md) | 待确认的三项口径 |
| [`docs/order_params_pilot200.md`](docs/order_params_pilot200.md) | 五个坐标，\(n=200\) |
| [`docs/step_motions_pilot200.md`](docs/step_motions_pilot200.md) | 相邻步运动标签（第一版） |
| [`docs/motion_shapes_concat200.md`](docs/motion_shapes_concat200.md) | 仅 \(m\) 的四种径向形状（concat，另一套窗口） |
| [`docs/phase1_status_report.md`](docs/phase1_status_report.md) | Phase-1 前置报告 |

---

## English

Research code for **semantic evolution of comment-section discussions**: Reddit threads as the human reference, multi-agent LLM threads as the experimental condition. Phase-1 (i) scores human threads with windowed embeddings and five discussion-cloud order parameters \((m,\sigma,d,k,\chi)\), and (ii) runs an AutoGen DeepSeek baseline. \(S_{end}\) is only the radial endpoint of \(m\); it is not a path and not internal geometry.

Private repo: add collaborators on GitHub. Data dumps and `outputs/*.jsonl` are not in git. Set `THREADS_PATH` if the zstd dump is not at `../threads_2026-01_to_2026-05.jsonl.zst`. GPU chunk recipe: [`docs/RUN_ON_GPU.md`](docs/RUN_ON_GPU.md).

```bash
pip install -r requirements.txt
python scripts/extract_order_params.py --backend hashing --limit 15 --tag smoke
python scripts/extract_order_params.py --skip 0 --limit 500 --device cuda --batch-size 32 --tag gpu_000_500
```

---

## Locked Phase-1 knobs

Linearize top-level first, children by time. Window size 10; late windows 3. Human labels: \(S_{end}\) q80/q20 (middle dropped). Simulation: 3 homogeneous DeepSeek agents, full history, round-robin, flat or weak_tree, 50 threads/class, max 100 comments. Embedding: `bge-m3` now (`device: cpu` in config; pass `--device cuda` on GPU); `text-embedding-3-large` reserved.

Never commit `.env` or API keys.
