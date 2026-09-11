"""Stage D of the persona demo: one self-contained page comparing both lines.

Each thread gets the five order parameters with the human line and the persona
line drawn on shared axes, both motion labels, the persona cards, and the two
transcripts position by position. Real usernames never reach this file; only the
Agent-N alias does.

    python scripts/export_persona_demo_html.py
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mas_collapse.config import load_config


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ctm = _load("classify_thread_motions")
MOTIONS = ctm.MOTIONS
MOTION_ZH = ctm.MOTION_ZH
COLORS = ctm.COLORS

HUMAN_COLOR = "#6e7781"
SIM_COLOR = "#c44e52"

SERIES = (
    ("sims_to_first", "S　还像开头吗", 0.35, 1.02, "越高越像开场。掉下去=云心搬走。掉了再爬回=绕圈。"),
    ("sigma", "σ　说得有多散", None, None, "往上=各说各话。往下=收成一套说法。"),
    ("eff_dim", "d　占了几个方向", None, None, "往上=好几件不相关的事同时在。往下=被压窄。"),
    ("k_modes", "k　有几派", 0.5, 5.5, "往上=裂成对立阵营。平在 1～2=仍是一团。"),
    ("chi", "χ　方位转了没", -0.02, 1.02, "贴 0=没转。中间鼓包=绕了一圈。后段突然抬=转开了。"),
)

LIMITS = [
    "没有「无人设」对照臂。这页能说的是带人设的 agent 离真人多远，说不了人设本身有没有用。",
    "候选帖被政治类 Ask 版主导。要求「本帖发言 ≥3 条且在其它采样帖里出现过」的人，AITA 那类一次性发言的版块几乎凑不齐。",
    "人设料的多少极不均：最多的 112 条跨帖评论，最少的只有 1 条。卡上都标了条数，薄的那几张基本是猜的。",
    "只有 3 个帖、6 条轨迹。下面的运动标签是描述，不是结论。",
    "长度是混淆项：模拟评论平均只有真人的一半左右（人设按各人跨帖历史的中位长度下指示，"
    "而这几个人在本帖里写得比平时长）。σ 和 d 偏低有可能只是文本短，要把真人截到同长再嵌一遍才能排除。",
]


def _arr(vals) -> list[float | None]:
    out: list[float | None] = []
    for x in vals or []:
        try:
            v = float(x)
        except (TypeError, ValueError):
            out.append(None)
            continue
        out.append(None if not np.isfinite(v) else v)
    return out


def svg_pair(
    human: list[float | None],
    sim: list[float | None],
    *,
    y_lo: float | None,
    y_hi: float | None,
    w: int = 300,
    h: int = 130,
) -> tuple[str, bool]:
    pad_l, pad_r, pad_t, pad_b = 34, 10, 10, 18
    finite = [v for v in (human + sim) if v is not None]
    if not finite:
        return f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}"></svg>', False
    zoomed = False
    if y_lo is None or y_hi is None:
        lo, hi = min(finite), max(finite)
        span = hi - lo
        if span < 0.04:
            mid = 0.5 * (lo + hi)
            y_lo, y_hi, zoomed = mid - 0.03, mid + 0.03, True
        else:
            y_lo, y_hi = lo - 0.08 * span, hi + 0.08 * span
    if abs(y_hi - y_lo) < 1e-9:
        y_lo, y_hi = y_lo - 0.05, y_hi + 0.05

    inner_w, inner_h = w - pad_l - pad_r, h - pad_t - pad_b
    n = max(max(len(human), len(sim)) - 1, 1)

    def path(ys: list[float | None], color: str, dash: str) -> str:
        d, drawing, dots = [], False, []
        for i, v in enumerate(ys):
            if v is None:
                drawing = False
                continue
            x = pad_l + inner_w * (i / n)
            y = pad_t + inner_h * (1.0 - (v - y_lo) / (y_hi - y_lo))
            d.append(f"{'L' if drawing else 'M'}{x:.1f},{y:.1f}")
            dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="{color}"/>')
            drawing = True
        if not d:
            return ""
        return (
            f'<path d="{" ".join(d)}" fill="none" stroke="{color}" stroke-width="1.9" '
            f'stroke-linejoin="round" stroke-linecap="round" {dash}/>' + "".join(dots)
        )

    parts = [
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{h - pad_b}" stroke="#d0d7de"/>',
        f'<line x1="{pad_l}" y1="{h - pad_b}" x2="{w - pad_r}" y2="{h - pad_b}" stroke="#d0d7de"/>',
        f'<text x="2" y="{pad_t + 8}" font-size="9" fill="#656d76">{y_hi:.2f}</text>',
        f'<text x="2" y="{h - pad_b}" font-size="9" fill="#656d76">{y_lo:.2f}</text>',
        f'<text x="{w - pad_r - 26}" y="{h - 4}" font-size="9" fill="#656d76">时间 →</text>',
        path(human, HUMAN_COLOR, 'stroke-dasharray="4 3"'),
        path(sim, SIM_COLOR, ""),
    ]
    return f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img">{"".join(parts)}</svg>', zoomed


def svg_scores(scores: dict[str, float], winner: str) -> str:
    lo, hi = min(scores.values()), max(scores.values())
    span = max(hi - lo, 1e-6)
    w, row_h, pad = 380, 17, 4
    h = pad + row_h * len(MOTIONS)
    rows = []
    for i, mot in enumerate(MOTIONS):
        y = pad + i * row_h
        val = scores[mot]
        bar = 8 + 180 * (val - lo) / span
        fill = COLORS[mot] if mot == winner else "#afb8c1"
        weight = "700" if mot == winner else "400"
        mark = " ←判成这个" if mot == winner else ""
        rows.append(
            f'<rect x="72" y="{y + 3}" width="{bar:.1f}" height="11" rx="2" fill="{fill}" opacity="0.9"/>'
            f'<text x="4" y="{y + 12}" font-size="11" font-weight="{weight}" fill="#1f2328">{MOTION_ZH[mot]}</text>'
            f'<text x="{80 + bar:.1f}" y="{y + 12}" font-size="10" fill="#424a53">{val:.2f}{mark}</text>'
        )
    return f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}">{"".join(rows)}</svg>'


def card_block(card: dict, in_thread: int) -> str:
    s = card["stats"]
    subs = "、".join(f"r/{x} ×{n}" for x, n in s["top_subreddits"]) or "—"
    thin = (
        '<span class="warn">料太薄，这张卡基本是猜的</span>'
        if s["thin"]
        else ""
    )
    samples = "".join(
        f'<div class="samp">“{html.escape(x[:240])}{"…" if len(x) > 240 else ""}”</div>' for x in card["samples"]
    )
    err = f'<div class="warn">卡生成异常：{html.escape(card["error"])}</div>' if card.get("error") else ""
    return f"""
    <div class="card">
      <div class="card-h"><b>{html.escape(card["alias"])}</b>
        <span class="muted">本帖发言 {in_thread} 条 · 跨帖历史 {s["n_comments"]} 条 · 长度 {s["len_p10"]}–{s["len_p90"]}（中位 {s["len_median"]}）</span>
        {thin}</div>
      <div class="muted">常混：{html.escape(subs)}</div>
      {err}
      <div class="fld"><span class="k">立场</span>{html.escape(card["stance"]) or "—"}</div>
      <div class="fld"><span class="k">语气</span>{html.escape(card["tone"]) or "—"}</div>
      <div class="fld"><span class="k">习惯</span>{html.escape(card["quirks"]) or "—"}</div>
      <div class="fld"><span class="k">原话</span><div>{samples or "—"}</div></div>
    </div>"""


def transcript(sim_blob: dict) -> str:
    rows = []
    for h, g in zip(sim_blob["human"], sim_blob["generated"]):
        bad = f'<div class="warn">生成失败：{html.escape(g["error"])}</div>' if g["error"] else ""
        rows.append(
            f"""<tr>
              <td class="pos">#{h["position"]}<br><span class="muted">{html.escape(h["alias"])}</span></td>
              <td class="txt">{html.escape(h["body"])}<div class="muted">{h["n_chars"]} 字符</div></td>
              <td class="txt sim">{html.escape(g["body"])}{bad}<div class="muted">{g["n_chars"]} 字符</div></td>
            </tr>"""
        )
    return (
        '<table class="tr"><thead><tr><th>位置</th><th>真人原文</th><th>人设 agent 生成</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Export the persona-demo comparison page")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tag = cfg["persona_demo"]["tag"]
    persona_dir = Path(cfg["paths"]["personas"]) / tag
    raw = json.loads((persona_dir / "cast_and_history.json").read_text(encoding="utf-8"))
    cards_blob = json.loads((persona_dir / "cards.json").read_text(encoding="utf-8"))
    sims_dir = Path(cfg["paths"]["sims"]) / tag

    op_path = Path(cfg["paths"]["outputs"]) / "order_params" / f"order_params_{tag}.jsonl"
    op = {}
    for line in op_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            op[(r["thread_id"], r["side"])] = r

    lab_path = Path(cfg["paths"]["labels"]) / f"thread_motions_{tag}.jsonl"
    lab = {}
    for line in lab_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            lab[(r["thread_id"], r["side"])] = r

    sections = []
    for pid in cfg["persona_demo"]["posts"]:
        thread = raw["threads"][pid]
        blob = json.loads((sims_dir / f"{pid}.json").read_text(encoding="utf-8"))
        h_row, s_row = op.get((pid, "human")), op.get((pid, "sim"))
        h_lab, s_lab = lab.get((pid, "human")), lab.get((pid, "sim"))
        if not (h_row and s_row):
            continue

        charts = []
        for key, label, lo, hi, hint in SERIES:
            svg, zoomed = svg_pair(_arr(h_row.get(key)), _arr(s_row.get(key)), y_lo=lo, y_hi=hi)
            note = '<div class="warn">两边幅度都很小，纵轴被放大了</div>' if zoomed else ""
            charts.append(
                f'<div class="chart"><div class="ct">{html.escape(label)}</div>{svg}'
                f'<div class="muted">{html.escape(hint)}</div>{note}</div>'
            )

        cards = "".join(
            card_block(cards_blob["cards"][a], thread["in_thread_counts"][a]) for a in thread["cast"]
        )
        share = len(thread["turns"]) / max(1, thread["n_comments_thread"])
        same = h_lab["motion"] == s_lab["motion"]
        verdict = (
            f'两边都判成<b>{MOTION_ZH[h_lab["motion"]]}</b>'
            if same
            else f'真人判成<b>{MOTION_ZH[h_lab["motion"]]}</b>，模拟判成<b>{MOTION_ZH[s_lab["motion"]]}</b>'
        )

        sections.append(f"""
  <section>
    <h2>r/{html.escape(thread["subreddit"])} · {html.escape(thread["title"])}</h2>
    <p class="muted">全帖 {thread["n_comments_thread"]} 条，这 5 个人写了 {len(thread["turns"])} 条（{share:.0%}）。
    两条线都只保留这 {len(thread["turns"])} 条，位置、发言人、条数完全一致。
    平均长度：真人 {h_row["mean_chars"]:.0f} 字符，模拟 {s_row["mean_chars"]:.0f} 字符。</p>
    <div class="legend"><span class="ln h"></span>真人　<span class="ln s"></span>人设 agent</div>
    <div class="charts">{"".join(charts)}</div>
    <p>{verdict}（分差 真人 {h_lab["margin"]:.2f} / 模拟 {s_lab["margin"]:.2f}）。</p>
    <div class="two">
      <div><div class="ct">真人这条线的九类打分</div>{svg_scores(h_lab["scores"], h_lab["motion"])}</div>
      <div><div class="ct">模拟这条线的九类打分</div>{svg_scores(s_lab["scores"], s_lab["motion"])}</div>
    </div>
    <h3>这一场的人设卡</h3>
    <div class="cards">{cards}</div>
    <h3>逐条对照</h3>
    {transcript(blob)}
  </section>""")

    css = """
    body{font-family:"Microsoft YaHei","Segoe UI",system-ui,sans-serif;max-width:1180px;margin:0 auto;
         padding:24px 18px 80px;color:#1f2328;line-height:1.65;background:#fff}
    h1{font-size:24px;margin:0 0 6px} h2{font-size:18px;margin:6px 0 4px;border-bottom:2px solid #eaeef2;padding-bottom:6px}
    h3{font-size:15px;margin:18px 0 8px;color:#424a53}
    section{margin:34px 0;padding:18px;border:1px solid #eaeef2;border-radius:10px;background:#fcfcfd}
    .muted{color:#656d76;font-size:12px}
    .warn{color:#9a6700;font-size:12px;background:#fff8c5;border-radius:4px;padding:1px 6px;display:inline-block;margin-top:3px}
    .charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin:10px 0}
    .chart{border:1px solid #eaeef2;border-radius:8px;padding:8px;background:#fff}
    .ct{font-size:13px;font-weight:700;margin-bottom:4px}
    .legend{font-size:12px;color:#424a53;margin:6px 0}
    .ln{display:inline-block;width:22px;height:0;border-top:2px solid;vertical-align:middle;margin-right:5px}
    .ln.h{border-color:#6e7781;border-top-style:dashed} .ln.s{border-color:#c44e52}
    .two{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:8px 0 4px}
    .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:10px}
    .card{border:1px solid #eaeef2;border-radius:8px;padding:10px;background:#fff;font-size:13px}
    .card-h{margin-bottom:2px}
    .fld{margin-top:5px} .fld .k{display:inline-block;min-width:34px;color:#656d76;font-size:12px;margin-right:6px}
    .samp{color:#424a53;font-size:12px;background:#f6f8fa;border-radius:4px;padding:4px 7px;margin-top:3px}
    table.tr{width:100%;border-collapse:collapse;font-size:13px;table-layout:fixed}
    table.tr th{text-align:left;background:#f6f8fa;padding:6px 8px;border:1px solid #eaeef2;font-size:12px}
    table.tr td{border:1px solid #eaeef2;padding:7px 9px;vertical-align:top;white-space:pre-wrap;word-break:break-word}
    td.pos{width:66px;color:#656d76;font-size:12px} td.sim{background:#fffaf8}
    ol.lim li{margin:4px 0}
    """

    limits = "".join(f"<li>{html.escape(x)}</li>" for x in LIMITS)
    page = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>人设 LLM 模拟 demo · 真人 vs 同班底 agent</title>
<style>{css}</style></head><body>
<h1>人设 LLM 模拟 demo</h1>
<p>同一个帖、同一批人、同样的发言位置，把真人换成按其历史写成的 LLM 角色，再用同一套五个序参量量两条线。</p>
<h3>协议</h3>
<ul>
<li><b>班底</b>：本帖发言 ≥3 条、且在其它采样帖里出现过 ≥2 次的人，按发言量取前 5。两条线都只保留这 5 个人的评论。</li>
<li><b>人设料</b>：该账号在<i>其它</i>采样帖里的真实评论。目标帖本身从不进入人设，也从不给 agent 看。
    历史条数、长度分位、版块分布由代码统计；模型只写立场、语气、习惯三项。</li>
<li><b>生成</b>：给原帖标题与正文，给已生成的前 k−1 条，<b>不给任何真人评论</b>，不给回复树提示（扁平）。
    长度按各人自己的历史中位数下指示，token 上限按其 p90 换算。</li>
<li><b>模型</b>：{html.escape(cards_blob["model"])}（人设卡与评论同一个）。换 gpt-5.6-luna 只改 provider。</li>
<li><b>指标</b>：bge-m3，窗口 10，支撑 30，k 的距离阈 0.55。运动打分的尺度取自 200 帖那次运行，不由这 3 个帖自己定。</li>
</ul>
<h3>已知局限</h3>
<ol class="lim">{limits}</ol>
{"".join(sections)}
</body></html>"""

    out = Path(args.out) if args.out else ROOT / "docs" / f"{tag}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
