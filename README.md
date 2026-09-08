# mas-semantic-collapse

多人（或多个模型）接着聊时，意思会不会越来越窄、越来越像开场那几句？本项目把一场评论区讨论写成语义空间里的一团点，跟踪这团点怎么动；人类 Reddit 帖是对照，多智能体互相接话是实验对象。

只看「聊到最后还像不像开头」会把不同过程标成同一类。有的帖几乎没离开开场，有的绕一圈又回来，两者期末都可以很像开头；有的搬走后停住，有的还在往下掉，两者期末都可以不像开头。要区分这些过程，需要五个互相不蕴含的量。

测量方式参考 Kong et al., [*Multi-LLM Systems Exhibit Robust Semantic Collapse*](https://arxiv.org/abs/2605.17193)。本仓库不照搬该文的人类标签，而在本语料上估计下面五个量，并（可选地）跑 AutoGen 模拟评论区。

私有仓库。让别人跑数：GitHub **Settings → Collaborators**。原始数据和 `outputs/*.jsonl` 不在 git 里。

---

## 五个量分别是什么、有什么意义

把每条评论做成 embedding 里的一个点。相邻一段时间里的点构成**讨论云**。五个量描述这团云**此刻**长什么样；它们随时间的变化，才对应老师表里的运动名称（平移、凝聚、裂变等）。

**\(m\)（位置 / 云心）**  
这批发言的平均语义位置，也就是「这会儿大家总体在说什么」。\(m\) 搬走，表示话题中心从开场挪到了别处（例如从「该不该买」转到「买哪个型号」）。常用的 \(S_{end}\) 只是末期 \(m\) 相对第一窗 \(m\) 还有多像，是这一根轴的终点，不是整场讨论的全部。\(m\) 搬走本身**不等于**讨论死掉：整团可以换地方，内部仍可以很丰富。

**\(\sigma\)（散布）**  
点离云心有多远，也就是「这会儿大家说得有多散」。\(\sigma\) 变大：各说各话、例子越甩越开（扩散）。\(\sigma\) 变小：说法收成同一句口号（凝聚）。只看 \(m\) 看不见这件事：中心可以停在原地，云自己胀大或收紧。

**\(d\)（有效维数）**  
这团点占了几个独立方向，而不是离中心有多远。\(d\) 高：好几件不相关的意思同时在场。\(d\) 下降：被压成一条线，例如只剩赞成–反对（成丝）。\(\sigma\) 大只说明离中心远；\(d\) 回答的是「远在一个方向上，还是摊在许多方向上」。

**\(k\)（模态 / 有几坨）**  
点是连续的一团，还是中间空着、裂成几派。正式 \(k\) 用余弦距离的层次聚类：距离低于 0.55（大约相似度 ≥ 0.45）收成一簇，再数人数不少于 2 的簇。\(k=1\)：意见是过渡的。\(k\) 上升：出现分开的阵营（裂变）。旧的特征间隙整数 \(k\) 几乎总是 1，只留作对照；二分强度仍用来看「硬切成两坨干不干净」。

**\(\chi\)（换位）**  
只看语义，不看是谁在说。\(\chi\) 是一个随时间变的数，图的横轴仍是时间。在本帖内部的二维语义平面上，取每窗云心相对第一窗的方位角，\(\chi_t=0.5(1-\cos\Delta\theta)\)。高维里绕一圈，时间图上是波浪；冻住则贴着 0。旧的 1024 维匈牙利匹配（`churn_config`）在真实句子上会饱和，不再当作正式 \(\chi\)。

这五个量不能互相替代。\(S_{end}\) 高，可能是锚在开场，也可能是绕一圈回来（轨道）。\(\sigma\) 上升，可能是均匀扩散，也可能是裂成两派，只有 \(k\) 能分开。相邻 \(m\) 很像，只说明质心没怎么挪，不说明 \(\chi\) 低。人机对照若只用 \(S_{end}\) 切两类，等于把上述混合物当成同一种处理条件。

本仓库现阶段：（1）在人类帖上把五个量算出来；（2）搭好多智能体评论区，以后用**同一套量**对照，而不是只比两个终点分数。

---

## 这个仓库能做什么

- 把 Reddit 线程按窗嵌入，得到 \(S_{end}\) 等曲线，并按样本内 q80/q20 做相对标签。
- 在同一批评论向量上估计 \((m,\sigma,d,k,\chi)\)，再按相邻步差分贴运动名（第一版启发式，口径待确认）。
- 用 AutoGen + DeepSeek 生成模拟评论区，并用同一套指标打分。
- 在 GPU 上分块跑嵌入（`--skip` / `--limit` / `--device cuda`），合并后在本地作图。

全量约 1900 条建议在 [`docs/protocol_for_shiyang.md`](docs/protocol_for_shiyang.md) 确认后再跑。验证环境随时可以用 `--limit`。

---

## 怎么跑

没接触过本仓库：先读 **[操作手册](docs/HANDBOOK.md)**（文件放哪、模型怎么设、结果在哪）。三级输入输出： [pipeline.md](docs/pipeline.md)。

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
| [`docs/HANDBOOK.md`](docs/HANDBOOK.md) | 新手操作：文件位置、模型、命令、结果路径 |
| [`docs/pipeline.md`](docs/pipeline.md) | 三级输入输出（评论云 → 运动标签 → 标题簇） |
| [`docs/RUN_ON_GPU.md`](docs/RUN_ON_GPU.md) | 算力机逐步命令 |
| [`docs/protocol_for_shiyang.md`](docs/protocol_for_shiyang.md) | 待确认的三项口径 |
| [`docs/order_params_pilot200.md`](docs/order_params_pilot200.md) | 五个坐标，\(n=200\) |
| [`docs/step_motions_pilot200.md`](docs/step_motions_pilot200.md) | 相邻步运动标签（第一版） |
| [`docs/motion_shapes_concat200.md`](docs/motion_shapes_concat200.md) | 仅 \(m\) 的四种径向形状（concat，另一套窗口） |
| [`docs/phase1_status_report.md`](docs/phase1_status_report.md) | Phase-1 前置报告 |

---

## English

A comment thread is a **cloud of embedding points**. Five coordinates describe the cloud at a time; \(S_{end}\) is only how close the late center \(m\) remains to the first window.

- \(m\): where the mean meaning sits. Moving \(m\) is a topic shift, not by itself “collapse”.
- \(\sigma\): how far comments sit from that mean (spread vs slogan-like condensation).
- \(d\): how many independent directions the cloud occupies (a blob vs a single agree–disagree line).
- \(k\): one lump vs separated camps. Integer \(k\) saturates at 1 on short windows; we also report a continuous split score.
- \(\chi\): whether the internal arrangement turns over after the center is removed. The center can stay put while \(\chi\) stays high.

Human Reddit is the reference; multi-agent reply threads are the experimental condition. Same estimator on both sides, not two \(S_{end}\) numbers alone.

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
