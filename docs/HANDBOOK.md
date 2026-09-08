# 操作手册（没接触过本仓库也可以按这个做）

这份只讲：**文件放哪、跑前怎么设模型、命令怎么敲、结果出现在哪**。  
三级分别在算什么，见 [pipeline.md](pipeline.md)。打开就能看的样例页：

- [运动分类审计（chik200）](thread_motions_chik200.html)
- [标题簇 × 运动（chik200）](title_clusters_chik200.html)

---

## 1. 电脑上要有什么

- Python 3.10 或更新
- 大约 5GB 磁盘（第一次会下载 `bge-m3` 模型）
- Reddit 数据压缩包：`threads_2026-01_to_2026-05.jsonl.zst`（**不在 GitHub 里**，要自己拷）

Windows 用 PowerShell。下面命令里的 `python` 若不行，改成 `py`。

---

## 2. 文件放哪里

建议目录长这样（数据在仓库**外面**一层）：

```text
某文件夹/
  threads_2026-01_to_2026-05.jsonl.zst    ← 原始评论区，放这里
  mas-semantic-collapse/                  ← git clone 出来的仓库
    .env                                  ← 你自己复制出来的密钥，不要提交
    configs/default.yaml                  ← 改模型主要改这里
    scripts/run_pipeline.py               ← 入口
    outputs/                              ← 跑完自动出现，不进 git
    docs/                                 ← 说明书和 HTML 样例
```

数据不在默认位置时，在 `.env` 里写绝对路径：

```text
THREADS_PATH=D:\数据\threads_2026-01_to_2026-05.jsonl.zst
```

Windows 也可以写成 `D:/数据/threads_2026-01_to_2026-05.jsonl.zst`。

---

## 3. 第一次安装

```bash
git clone git@github.com:Huaji-Wang/mas-semantic-collapse.git
cd mas-semantic-collapse
python -m venv .venv
```

Windows 激活：

```text
.\.venv\Scripts\activate
```

macOS / Linux：

```bash
source .venv/bin/activate
```

然后：

```bash
pip install -r requirements.txt
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux
```

`.env` 可以先空着。只跑本地 `bge-m3` 时不必填任何 key。

可选：

| 变量 | 什么时候要填 |
|---|---|
| `THREADS_PATH` | 数据文件不在仓库上一级目录 |
| `HF_TOKEN` | 下载 `BAAI/bge-m3` 很慢或被限流 |
| `DEEPSEEK_API_KEY` | 要给标题簇自动起中文名字 |
| `OPENAI_API_KEY` | 第 3 级标题改用 OpenAI 向量 |

---

## 4. 跑之前：怎么设定模型

打开 `configs/default.yaml`。

**评论怎么嵌（第 1 级，又慢又重）**

```yaml
embedding:
  backend: bge_m3          # 改成 openai 则用 OpenAI
  model_name: BAAI/bge-m3
  device: cpu              # 本机请保持 cpu
```

**标题怎么嵌（第 3 级，只有标题，很快）**

```yaml
title_embedding:
  backend: bge_m3          # 可改 openai 或 hashing
  model_name: BAAI/bge-m3
  device: cpu
```

不改 yaml、只临时换标题模型：

```bash
python scripts/run_pipeline.py --from 3 --tag chik200 --title-backend openai
```

三种 `backend`：

| 值 | 用什么 | 要什么 |
|---|---|---|
| `bge_m3` | 本地 `BAAI/bge-m3` | 第一次会下载模型 |
| `openai` | `text-embedding-3-large` | `.env` 里 `OPENAI_API_KEY` |
| `hashing` | 假向量，只测通路 | 结果不能写进报告 |

**不要在这台也用来看屏幕的 RTX 4070 上开 CUDA。** 曾经嵌十几分钟整机断电。算力机另说，见 [RUN_ON_GPU.md](RUN_ON_GPU.md)。

---

## 5. 跑哪条命令

先激活 `.venv`，再在仓库根目录执行。

`{tag}` 只是这批结果的名字，例如 `chik200`。输入输出文件名都带这个后缀。

**已经有第 2 级标签、只想重跑标题聚类（默认、最快）**

```bash
python scripts/run_pipeline.py --from 3 --tag chik200
```

**曲线已经有了，要重新打九类运动，再跑标题**

```bash
python scripts/run_pipeline.py --from 2 --tag chik200
```

**从原始 zst 三级全跑（CPU 嵌评论要数小时）**

```bash
python scripts/run_pipeline.py --from 1 --tag chik200 --limit 200 --device cpu
```

第一次只想确认代码没坏（数字无意义）：

```bash
python scripts/extract_order_params.py --backend hashing --limit 15 --tag smoke
```

---

## 6. 泡出来的结果在哪里

全部在仓库里的 `outputs/`（本机才有）和 `docs/`（HTML 可进 git）。

| 你想看什么 | 打开哪个 |
|---|---|
| 每帖五条曲线 | `outputs/order_params/order_params_{tag}.jsonl` |
| 每帖运动类（平移、裂变…） | `outputs/labels/thread_motions_{tag}.jsonl` |
| 九类各有多少帖 | `outputs/metrics/thread_motions_{tag}_stats.json` |
| 为什么判成这一类（给人看） | `docs/thread_motions_{tag}.html` |
| 每帖标题属于哪一簇 | `outputs/title_clusters/title_clusters_{tag}.jsonl` |
| 每簇的名字、例子、定义 | `outputs/title_clusters/clusters_{tag}.json` |
| 标题簇 × 运动对照 | `outputs/metrics/title_cluster_vs_motion_{tag}.json` |
| 标题簇给人看 | `docs/title_clusters_{tag}.html` |

JSONL 是一行一篇帖，用文本编辑器或 VS Code 打开即可。HTML 用浏览器打开。

GitHub **没有** 这些 jsonl。克隆仓库后如果没有自己跑过、也没有人拷给你 `outputs/`，第 3 级会因为找不到 `outputs/labels/thread_motions_chik200.jsonl` 而失败。

---

## 7. 常见卡住

| 现象 | 怎么办 |
|---|---|
| 找不到 zst | 把数据放到仓库上一级，或设 `THREADS_PATH` |
| `--from 3` 报没有 labels | 先跑 `--from 2` 或 `--from 1`，或拷一份 `outputs/labels/` |
| 下载 Hugging Face 失败 | `.env` 填 `HF_TOKEN` |
| 标题簇没有中文名 | 正常：没填 `DEEPSEEK_API_KEY` 就会跳过起名 |
| 想开 CUDA | 不要在显示用的 GPU 上开。算力机才用 `--allow-laptop-cuda` |

---

## 8. 和「移动这份说明」

把本文件 `docs/HANDBOOK.md` 单独拷走也可以看。仓库根目录始终是「敲命令的地方」；数据在仓库外；结果在 `outputs/` 和 `docs/*.html`。
