# 在算力机器上跑数据

仓库不含 Reddit 原始数据、不含 `outputs/` 下的 jsonl。嵌入模型首次运行时会从 Hugging Face 下载 `BAAI/bge-m3`。

当前口径（窗宽 10 / 支撑 30、身份无关 \(\chi\)、二分强度代理 \(k\)）仍待确认。确认前若只跑通流水线，可用 `--limit` 做小样本；全量约 1900 条请在口径确认后启动。

## 1. 环境

```bash
git clone git@github.com:Huaji-Wang/mas-semantic-collapse.git
cd mas-semantic-collapse
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # 仅跑嵌入时可不填 DeepSeek
```

可选：`HF_TOKEN` 写入 `.env`，避免 Hub 限速。

检查 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## 2. 数据位置

默认读取相对仓库的 `../threads_2026-01_to_2026-05.jsonl.zst`（与 `configs/default.yaml` 中 `data.threads_path` 一致）。

若文件在别处：

```bash
export THREADS_PATH=/absolute/path/threads_2026-01_to_2026-05.jsonl.zst
```

合格线程：评论数 \(\ge 50\) 且 \(\le 1000\)。

## 3. 五个序参量（老师机器上的主任务）

每个合格线程只嵌入一次评论向量；\(\sigma,d,k,\chi\) 在内存中的点云上计算，向量不落盘。输出为 `outputs/order_params/order_params_<tag>.jsonl`。

代码自检（秒级，不可写进报告）：

```bash
python scripts/extract_order_params.py --backend hashing --limit 15 --tag smoke
```

GPU 分块（可中断；`--skip` 按合格线程计数，不是按文件行号）：

```bash
python scripts/extract_order_params.py --skip 0    --limit 500 --device cuda --batch-size 32 --tag gpu_000_500
python scripts/extract_order_params.py --skip 500  --limit 500 --device cuda --batch-size 32 --tag gpu_500_1000
python scripts/extract_order_params.py --skip 1000 --limit 500 --device cuda --batch-size 32 --tag gpu_1000_1500
python scripts/extract_order_params.py --skip 1500 --limit 0   --device cuda --batch-size 32 --tag gpu_1500_end
```

`--limit 0` 表示 skip 之后全部跑完。若某块中断，同 `--tag` 会覆盖该块；换新 tag 或从该块的 skip 重跑。检查点为 `*.partial.jsonl`，正常结束会写成正式 jsonl 并删除 partial。

合并：

```bash
python scripts/merge_jsonl.py \
  outputs/order_params/order_params_gpu_000_500.jsonl \
  outputs/order_params/order_params_gpu_500_1000.jsonl \
  outputs/order_params/order_params_gpu_1000_1500.jsonl \
  outputs/order_params/order_params_gpu_1500_end.jsonl \
  -o outputs/order_params/order_params_full.jsonl
```

作图（本地即可，不必 GPU）：

```bash
python scripts/plot_order_params.py --jsonl outputs/order_params/order_params_full.jsonl --tag full
python scripts/classify_step_motions.py --jsonl outputs/order_params/order_params_full.jsonl --tag full
```

## 4. 人类侧 \(S_{end}\) 打分（可选，与序参量嵌入成本同量级）

```bash
python -m scripts.label_threads --config configs/default.yaml \
  --device cuda --batch-size 32 --no-vendi --scores-only \
  --skip 0 --limit 500 --tag gpu_000_500
```

分块后用 `scripts/merge_thread_scores.py` 或 `scripts/merge_jsonl.py` 合并，再在全量分数上切 q80/q20。不要把 concat 200 的形状阈值套到 mean-pool 分数上。

## 5. 回传什么

请回传 `outputs/order_params/` 下的 jsonl 与 `*_summary.json`（不要回传 Hugging Face 模型缓存）。仓库已忽略 `*.jsonl` 与 `outputs/`，不会被误提交。

口径说明见 `docs/protocol_for_shiyang.md`。
