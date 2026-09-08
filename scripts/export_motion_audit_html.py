"""Build a self-contained HTML audit of whole-thread motion labels.

Picks the 1–2 highest-margin winners per occupied class. Empty classes get
the nearest miss: highest score for that row whose argmax was something else.

    python scripts/export_motion_audit_html.py --tag chik200
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_classifier():
    spec = importlib.util.spec_from_file_location(
        "classify_thread_motions", ROOT / "scripts" / "classify_thread_motions.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ctm = _load_classifier()
MOTIONS = ctm.MOTIONS
MOTION_ZH = ctm.MOTION_ZH
COLORS = ctm.COLORS

CLASS_STORY = {
    "translation": {
        "want": "话题中心搬走，而且搬走之后就停在新地方。另外四条没有整段胀大、收紧、裂开或绕圈。",
        "look": "S 从高处掉下去、不再回到 1；σ / d / k / χ 大体平。",
        "not": "若 S 先掉再爬回，会改判轨道。若 k 明显增多，会改判裂变。",
    },
    "diffusion": {
        "want": "大家说得越来越散，语义方位也转开；不必裂成对立派。",
        "look": "σ 抬头，χ 后段抬高；k 不必上升。",
        "not": "k 明显增多时，散开更像裂变，不叫扩散。",
    },
    "condensation": {
        "want": "说法收紧，绕开的方位也收回来。",
        "look": "σ 往下走，χ 回落；S 仍比较高。",
        "not": "只是 χ 冻在 0、云心也不走，更像结晶。",
    },
    "fission": {
        "want": "讨论裂成几坨，不再是连续的一团。",
        "look": "k 后段高于前段。σ 可以跟着胀。",
        "not": "只有 σ 胀、k 不动，仍是扩散。",
    },
    "crystallization": {
        "want": "方位不再转，讨论冻住。",
        "look": "χ 贴着 0，后期几乎不再跳。",
        "not": "云心已经搬走时，冻住也赢不过平移。",
    },
    "vortex": {
        "want": "内部一直在转，但不是绕一圈再回来。",
        "look": "χ 不平，却不成浪；S 不太掉。",
        "not": "χ 中间鼓包、两端低，是轨道，不是涡流。",
    },
    "orbit": {
        "want": "话题走远一圈，结尾又像开头。",
        "look": "S 先降后升；χ 中间高、两头低。",
        "not": "S 降了不回来，是平移。只有 χ 动、S 不回，不够。",
    },
    "breathing": {
        "want": "散开和换位一起鼓胀再收回，云心还在。",
        "look": "σ 与 χ 同步起落。",
        "not": "只有 χ 起浪、σ 不起，更像轨道。",
    },
    "cascade": {
        "want": "不是慢慢漂，而是突然掉一截。",
        "look": "S 和 χ 都有台阶，不是平滑斜线。",
        "not": "平滑离开仍是平移。",
    },
}

SERIES = (
    ("sims_to_first", "S　还像开头吗", 0.55, 1.02, "越高越像开场。掉下去=云心搬走。掉了再爬回=绕圈。"),
    ("sigma", "σ　说得有多散", None, None, "往上=各说各话。往下=收成一套说法。"),
    ("eff_dim", "d　占了几个方向", None, None, "往上=好几件不相关的事同时在。往下=被压窄。"),
    ("k_modes", "k　有几派", 0.5, 5.5, "往上=裂成对立阵营。平在 1～2=仍是一团。"),
    ("chi", "χ　方位转了没", -0.02, 1.02, "贴 0=没转。中间鼓包=绕了一圈。后段突然抬=转开了。"),
)


def _arr(vals) -> list[float | None]:
    out: list[float | None] = []
    for x in vals:
        if x is None:
            out.append(None)
            continue
        try:
            v = float(x)
        except (TypeError, ValueError):
            out.append(None)
            continue
        out.append(None if not np.isfinite(v) else v)
    return out


def _finite(xs: list[float | None]) -> list[float]:
    return [v for v in xs if v is not None]


def _head_tail(ys: list[float | None]) -> tuple[float, float, float]:
    a = _finite(ys)
    if not a:
        return float("nan"), float("nan"), float("nan")
    n_early = max(1, len(a) // 3)
    early = float(np.mean(a[:n_early]))
    late = float(np.mean(a[-min(3, len(a)) :]))
    return early, late, late - early


def score_parts(f: dict, sc: dict) -> tuple[dict[str, float], dict[str, float], dict[str, list[tuple[str, float]]]]:
    sig_up, sig_dn = ctm._signed(f["d_sigma"], sc["sigma"])
    k_up, _k_dn = ctm._signed(f["d_k"], sc["k"])
    chi_up, chi_dn = ctm._signed(f["d_chi"], sc["chi"])
    sig_st = ctm._stable(f["d_sigma"], sc["sigma"])
    dim_st = ctm._stable(f["d_dim"], sc["dim"])
    k_st = ctm._stable(f["d_k"], sc["k"])
    chi_st = ctm._stable(f["d_chi"], sc["chi"])
    m_move = float(np.clip(f["m_leave"] / (sc["m"] * 2.0 + 1e-9), 0.0, 1.0))
    m_stay = 1.0 - m_move
    freeze = ctm._stable(f["chi_step_mean"], sc["step"]) * ctm._stable(f["chi_late_std"], sc["chi_std"])
    flow = 1.0 - freeze
    mw, cw = f["m_wave"], f["chi_wave"]
    sw, mstep, cstep = f["sigma_wave"], f["m_step"], f["chi_step_shape"]

    terms = {
        "m搬走": m_move,
        "m还在": m_stay,
        "m浪": mw,
        "m台阶": mstep,
        "σ升": sig_up,
        "σ降": sig_dn,
        "σ稳": sig_st,
        "σ浪": sw,
        "d稳": dim_st,
        "k升": k_up,
        "k稳": k_st,
        "χ升": chi_up,
        "χ降": chi_dn,
        "χ稳": chi_st,
        "χ浪": cw,
        "χ台阶": cstep,
        "冻住": freeze,
        "流动": flow,
    }
    recipes: dict[str, list[tuple[str, float]]] = {
        "translation": [("m搬走", 1.4), ("σ稳", 0.6), ("d稳", 0.4), ("k稳", 0.4), ("χ稳", 0.8), ("m浪", -0.8)],
        "diffusion": [("σ升", 1.2), ("χ升", 1.0), ("m还在", 0.4), ("k稳", 0.3)],
        "condensation": [("σ降", 1.2), ("χ降", 1.0), ("m还在", 0.4)],
        "fission": [("σ升", 1.0), ("k升", 1.4), ("χ降", 0.8), ("m还在", 0.3)],
        "crystallization": [("冻住", 1.6), ("σ稳", 0.5), ("k稳", 0.4), ("m还在", 0.4), ("流动", -0.6)],
        "vortex": [("流动", 1.4), ("m还在", 0.5), ("σ稳", 0.4), ("k稳", 0.4), ("χ浪", -0.7), ("冻住", -0.5)],
        "orbit": [("m浪", 1.3), ("χ浪", 1.3), ("σ稳", 0.3)],
        "breathing": [("σ浪", 1.2), ("χ浪", 1.2), ("m还在", 0.3)],
        "cascade": [("m台阶", 1.3), ("χ台阶", 1.1), ("σ稳", 0.3)],
    }
    scores = {mot: sum(w * terms[name] for name, w in recipes[mot]) for mot in MOTIONS}
    return scores, terms, recipes


def pick_examples(labeled: list[dict], per_class: int = 2) -> dict[str, list[dict]]:
    chosen: dict[str, list[dict]] = {m: [] for m in MOTIONS}
    for mot in MOTIONS:
        winners = [r for r in labeled if r["motion"] == mot]
        if winners:
            winners = sorted(winners, key=lambda r: (-r["margin"], -r["scores"][mot], r["post_id"]))
            for row in winners[:per_class]:
                item = dict(row)
                item["pick"] = "winner"
                chosen[mot].append(item)
            continue
        misses = sorted(labeled, key=lambda r: (-r["scores"][mot], r["post_id"]))
        if not misses:
            continue
        item = dict(misses[0])
        item["pick"] = "near_miss"
        item["miss_for"] = mot
        chosen[mot].append(item)
    return chosen


def svg_line(
    ys: list[float | None],
    *,
    color: str,
    y_lo: float | None = None,
    y_hi: float | None = None,
    w: int = 220,
    h: int = 96,
) -> tuple[str, bool]:
    pad_l, pad_r, pad_t, pad_b = 28, 8, 8, 16
    finite = _finite(ys)
    if not finite:
        return f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}"></svg>', False
    raw_lo, raw_hi = min(finite), max(finite)
    zoomed = False
    if y_lo is None or y_hi is None:
        span = raw_hi - raw_lo
        if span < 0.04:
            mid = 0.5 * (raw_lo + raw_hi)
            y_lo = mid - 0.03
            y_hi = mid + 0.03
            zoomed = True
        else:
            pad = 0.08 * span
            y_lo = raw_lo - pad
            y_hi = raw_hi + pad
    if abs(y_hi - y_lo) < 1e-9:
        y_lo -= 0.05
        y_hi += 0.05
    inner_w = w - pad_l - pad_r
    inner_h = h - pad_t - pad_b
    n = max(len(ys) - 1, 1)

    def xy(i: int, v: float) -> tuple[float, float]:
        x = pad_l + inner_w * (i / n)
        y = pad_t + inner_h * (1.0 - (v - y_lo) / (y_hi - y_lo))
        return x, y

    parts: list[str] = []
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{h - pad_b}" stroke="#d0d7de" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{pad_l}" y1="{h - pad_b}" x2="{w - pad_r}" y2="{h - pad_b}" stroke="#d0d7de" stroke-width="1"/>'
    )
    parts.append(f'<text x="2" y="{pad_t + 8}" font-size="9" fill="#656d76">{y_hi:.2f}</text>')
    parts.append(f'<text x="2" y="{h - pad_b}" font-size="9" fill="#656d76">{y_lo:.2f}</text>')
    d: list[str] = []
    drawing = False
    for i, v in enumerate(ys):
        if v is None:
            drawing = False
            continue
        x, y = xy(i, v)
        d.append(f"{'L' if drawing else 'M'}{x:.1f},{y:.1f}")
        drawing = True
    if d:
        parts.append(
            f'<path d="{" ".join(d)}" fill="none" stroke="{color}" '
            f'stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for i, v in enumerate(ys):
            if v is None:
                continue
            x, y = xy(i, v)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.1" fill="{color}"/>')
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img">{"".join(parts)}</svg>',
        zoomed,
    )


def svg_scores(scores: dict[str, float], winner: str, accent: str | None = None) -> str:
    lo = min(scores.values())
    hi = max(scores.values())
    span = max(hi - lo, 1e-6)
    rows = []
    w, row_h, pad = 420, 18, 4
    h = pad + row_h * len(MOTIONS)
    for i, mot in enumerate(MOTIONS):
        y = pad + i * row_h
        val = scores[mot]
        frac = (val - lo) / span
        bar_w = 8 + 210 * frac
        if mot == winner or (accent and mot == accent):
            fill = COLORS[mot]
            weight = "700"
        else:
            fill = "#afb8c1"
            weight = "400"
        mark = " ←判成这个" if mot == winner else (" ←本行分数" if accent and mot == accent else "")
        rows.append(
            f'<rect x="78" y="{y + 3}" width="{bar_w:.1f}" height="12" rx="2" fill="{fill}" opacity="0.9"/>'
            f'<text x="4" y="{y + 13}" font-size="11" font-weight="{weight}" fill="#1f2328">{html.escape(MOTION_ZH[mot])}</text>'
            f'<text x="{86 + bar_w:.1f}" y="{y + 13}" font-size="11" fill="#424a53">{val:.2f}{mark}</text>'
        )
    return f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}">{"".join(rows)}</svg>'


def read_curve(key: str, ys: list[float | None], feat: dict) -> str:
    early, late, delta = _head_tail(ys)
    if not np.isfinite(early):
        return "这条没有数。"
    if key == "sims_to_first":
        wave = float(feat.get("m_wave") or 0.0)
        if wave >= 0.35 and late >= 0.85:
            return f"先离开再回来：中途不像开头，期末又回到 {late:.2f}。"
        if late <= 0.90:
            return f"搬走了：期末只有 {late:.2f} 还像开头，没有回到 1。"
        if late >= 0.94:
            return f"几乎没离开开头：期末仍有 {late:.2f} 像开场。"
        return f"略微离开：期末 {late:.2f} 还像开头。"
    if key == "sigma":
        if delta > 0.02:
            return f"变散了：从 {early:.2f} 升到 {late:.2f}。"
        if delta < -0.02:
            return f"收紧了：从 {early:.2f} 降到 {late:.2f}。"
        return f"散布几乎没动：头 {early:.2f}，尾 {late:.2f}，差 {delta:+.3f}。图被放大了，别把小抖动看成大起落。"
    if key == "eff_dim":
        if delta > 0.25:
            return f"方向变多：从 {early:.2f} 到 {late:.2f}。"
        if delta < -0.25:
            return f"方向变少：从 {early:.2f} 到 {late:.2f}。"
        return f"有效维差不多：头 {early:.2f}，尾 {late:.2f}。"
    if key == "k_modes":
        if delta >= 0.8:
            return f"裂开了：平均从 {early:.1f} 派升到 {late:.1f} 派。"
        if delta <= -0.8:
            return f"派别在合并：从 {early:.1f} 降到 {late:.1f}。"
        return f"没有裂成更多派：头 {early:.1f}，尾 {late:.1f}。"
    if key == "chi":
        wave = float(feat.get("chi_wave") or 0.0)
        if max(_finite(ys) or [0]) < 0.05:
            return "整段贴着 0：方位几乎没转。"
        if wave >= 0.35:
            return f"中间鼓起来再落下（浪）：头 {early:.2f}，尾 {late:.2f}。这是绕圈，不是单向转开。"
        if delta > 0.12:
            return f"后段转开了：从 {early:.2f} 升到 {late:.2f}。"
        if delta < -0.12:
            return f"转开的方位又收回来：从 {early:.2f} 降到 {late:.2f}。"
        return f"有过转动，但头尾差不多：{early:.2f} → {late:.2f}。"
    return ""


def verdict(focus: str, winner: str, pick: str, feat: dict, terms: dict, scores: dict, margin: float) -> tuple[str, str]:
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    second = ranked[1][0] if len(ranked) > 1 else winner
    s_end = float(feat.get("s_end") or 0.0)
    leave = float(feat.get("m_leave") or 0.0)
    k_delta = float(feat.get("d_k") or 0.0)
    sig_delta = float(feat.get("d_sigma") or 0.0)
    chi_delta = float(feat.get("d_chi") or 0.0)
    m_wave = float(feat.get("m_wave") or 0.0)
    chi_wave = float(feat.get("chi_wave") or 0.0)

    seen = []
    if leave >= 0.08:
        seen.append(f"云心离开了开头（期末还像开头的程度只有 {s_end:.2f}）")
    else:
        seen.append(f"云心还比较像开头（S<sub>end</sub>={s_end:.2f}）")
    if m_wave >= 0.35:
        seen.append("但 S 是先降后升，像走了一圈")
    if sig_delta > 0.02:
        seen.append("说法变散了")
    elif sig_delta < -0.02:
        seen.append("说法收紧了")
    else:
        seen.append("散开程度没有整段变")
    if k_delta >= 0.8:
        seen.append("阵营数上升")
    elif k_delta <= -0.8:
        seen.append("阵营在合并")
    else:
        seen.append("没有裂成更多派")
    if chi_wave >= 0.35:
        seen.append("χ 起了浪")
    elif abs(chi_delta) < 0.05 and float(feat.get("chi_late_std") or 0) < 0.05:
        seen.append("χ 几乎冻住")
    elif chi_delta > 0.12:
        seen.append("方位后段转开")
    elif chi_delta < -0.12:
        seen.append("方位又转了回来")

    seen_txt = "；".join(seen) + "。"

    if pick == "near_miss":
        lead = (
            f"本批没有「{MOTION_ZH[focus]}」赢家。这条是{MOTION_ZH[focus]}分数最高的帖，"
            f"但机器仍判成<strong>{MOTION_ZH[winner]}</strong>。"
            f"{MOTION_ZH[focus]}得 {scores[focus]:.2f}，{MOTION_ZH[winner]}得 {scores[winner]:.2f}。"
        )
    else:
        lead = (
            f"机器判成<strong>{MOTION_ZH[winner]}</strong>，"
            f"比第二名{MOTION_ZH[second]}高 {margin:.2f}。"
        )

    why_not = {
        ("translation", "orbit"): "S 降了之后没有爬回 1，所以不是轨道。",
        ("translation", "fission"): "k 没有整段升高，所以不是裂变。",
        ("translation", "diffusion"): "σ 没有整段胀大，所以不是扩散。",
        ("translation", "crystallization"): "云心已经搬走，冻住也压不过「搬走」。",
        ("translation", "vortex"): "云心已经搬走，涡流那一行加不过平移。",
        ("diffusion", "fission"): "散开了，但 k 升高不够，所以还是扩散不是裂变。",
        ("diffusion", "translation"): "不只是中心搬走，σ 和 χ 后段都抬起来了。",
        ("diffusion", "cascade"): "后段抬升更像胀开，不像突然掉一截。",
        ("condensation", "orbit"): "σ 在收，不只是绕圈回来。",
        ("condensation", "translation"): "中心虽有移动，收紧和 χ 回落更明显。",
        ("condensation", "breathing"): "呼吸要 σ 与 χ 一起鼓再收回；这里主趋势是往回收。",
        ("fission", "diffusion"): "不只是变散，k 升高了，所以是裂成几派。",
        ("fission", "translation"): "搬走之外，阵营数也上去了。",
        ("orbit", "translation"): "期末又像开头，过程里 χ 也起浪，所以是绕圈不是搬走就停。",
        ("orbit", "crystallization"): "χ 不是冻住，中间鼓过包；S 也先走再回。",
        ("orbit", "breathing"): "主信号在 S 先走再回，不是 σ 跟着呼吸。",
        ("condensation", "fission"): "主趋势是收紧，不是裂成更多派。",
        ("cascade", "translation"): "S 和 χ 都是突然掉一截，不是平滑漂走。",
        ("crystallization", "translation"): "这条 χ 确实比较静，但云心搬走了，结晶加分加不过平移。",
        ("vortex", "translation"): "内部有流动，可云心已经离开，涡流加分加不过平移。",
        ("breathing", "condensation"): "σ、χ 有过起伏，但整段主趋势仍是收紧，所以判成了凝聚。",
    }
    extra = why_not.get((winner, second), f"第二名是{MOTION_ZH[second]}，但对应那几条线对不上。")
    if pick == "near_miss":
        extra = why_not.get((winner, focus), extra)

    return lead + seen_txt, extra


def example_card(src: dict, lab: dict, terms: dict, recipes: dict, scores: dict, focus: str) -> str:
    color = COLORS[focus]
    feat = {k: (float("nan") if v is None else float(v)) for k, v in (lab.get("features") or {}).items()}
    series_html = []
    readings = []
    for key, title, y_lo, y_hi, hint in SERIES:
        ys = _arr(src.get(key, []))
        svg, zoomed = svg_line(ys, color=color, y_lo=y_lo, y_hi=y_hi)
        reading = read_curve(key, ys, feat)
        zoom_note = " <span class=\"zoom\">纵轴被放大</span>" if zoomed else ""
        series_html.append(
            "<figure>"
            f"<figcaption>{html.escape(title)}</figcaption>"
            f"{svg}"
            f"<p class=\"hint\">{html.escape(hint)}{zoom_note}</p>"
            "</figure>"
        )
        readings.append(f"<li><strong>{html.escape(title.split('　')[0])}</strong>　{html.escape(reading)}</li>")

    title = lab.get("title") or src.get("title") or "(无标题)"
    sub = src.get("subreddit") or ""
    winner = lab["motion"]
    pick = lab.get("pick")
    lead, extra = verdict(focus, winner, pick, feat, terms, scores, float(lab["margin"]))
    if pick == "near_miss":
        banner_cls = "miss"
        banner = (
            f"近失样本：想找「{html.escape(MOTION_ZH[focus])}」，"
            f"机器却判成了「{html.escape(MOTION_ZH[winner])}」。"
        )
    else:
        banner_cls = "win"
        banner = f"判成{html.escape(MOTION_ZH[winner])}　分差 {lab['margin']:.2f}"

    rows = []
    for name, w in recipes[focus if pick == "near_miss" else winner]:
        contrib = w * terms[name]
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{terms[name]:.2f}</td>"
            f"<td>{w:+.1f}</td><td>{contrib:+.2f}</td></tr>"
        )

    return f"""
<article class="card">
  <h3>{html.escape(title)}</h3>
  <p class="meta">
    <code>{html.escape(str(lab['post_id']))}</code>
    · r/{html.escape(str(sub))}
    · {lab['n_windows']} 个时间窗
    · 期末还像开头 {lab['s_end']:.2f}
  </p>
  <p class="banner {banner_cls}">{banner}</p>
  <p class="verdict">{lead}</p>
  <p class="why-not">{html.escape(extra)}</p>
  <div class="curves">{"".join(series_html)}</div>
  <ol class="readings">{"".join(readings)}</ol>
  <div class="audit">
    <div>
      <h4>九个类各得多少分</h4>
      <p class="hint">最高的那一行就是标签。分差=第一名减第二名。</p>
      {svg_scores(scores, winner, accent=focus if pick == "near_miss" else None)}
    </div>
    <div>
      <h4>这几项在加分</h4>
      <p class="hint">数字只是核对用。先读上面的人话，不必从这里反推。</p>
      <table class="feat">
        <thead><tr><th>看见了什么</th><th>强度</th><th>权重</th><th>加分</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </div>
  </div>
</article>
"""


CSS = """
:root { color-scheme: light; }
body {
  margin: 0;
  font-family: "Segoe UI", "PingFang SC", "Noto Sans SC", "Microsoft YaHei", sans-serif;
  line-height: 1.65;
  color: #1f2328;
  background: #f6f8fa;
}
.wrap {
  max-width: 1080px;
  margin: 0 auto;
  padding: 32px 24px 72px;
  background: #fff;
  box-shadow: 0 0 0 1px #d0d7de;
}
h1 { font-size: 1.7rem; line-height: 1.3; margin-top: 0; }
h2 { font-size: 1.35rem; border-bottom: 1px solid #d0d7de; padding-bottom: 6px; margin-top: 2.2rem; }
h3 { font-size: 1.12rem; margin: 0 0 0.3rem; }
h4 { font-size: 0.98rem; margin: 0.2rem 0 0.4rem; }
p, li { font-size: 15.5px; }
.math { font-family: "Cambria Math", "Times New Roman", serif; }
code { background: #f6f8fa; padding: 0.1em 0.35em; border-radius: 4px; font-size: 13px; }
table { border-collapse: collapse; width: 100%; margin: 0.6rem 0; font-size: 14px; }
th, td { border: 1px solid #d0d7de; padding: 5px 8px; text-align: left; }
th { background: #f6f8fa; }
.toc a { color: #0969da; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.card {
  border: 1px solid #d0d7de;
  border-radius: 8px;
  padding: 14px 16px 12px;
  margin: 14px 0 22px;
  background: #fff;
}
.meta { color: #656d76; margin: 0 0 8px; font-size: 13.5px; }
.banner { margin: 0 0 10px; padding: 6px 10px; border-radius: 6px; font-weight: 600; }
.banner.win { background: #ddf4ff; border-left: 4px solid #0969da; }
.banner.miss { background: #fff1e5; border-left: 4px solid #bc4c00; }
.verdict { margin: 0 0 6px; }
.why-not { margin: 0 0 12px; color: #424a53; }
.curves {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 10px;
  margin: 8px 0 8px;
}
@media (max-width: 900px) { .curves { grid-template-columns: 1fr 1fr; } }
figure { margin: 0; }
figcaption { font-size: 13px; font-weight: 600; color: #1f2328; margin-bottom: 2px; }
.hint { font-size: 12px; color: #656d76; margin: 4px 0 0; line-height: 1.45; }
.zoom { color: #bc4c00; }
.readings { margin: 0 0 12px; padding-left: 1.2em; }
.readings li { margin: 0.25em 0; font-size: 14.5px; }
.audit { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 16px; }
@media (max-width: 900px) { .audit { grid-template-columns: 1fr; } }
.empty-note { color: #656d76; font-weight: 400; }
.story { background: #f6f8fa; padding: 10px 12px; border-radius: 6px; }
.story p { margin: 0.25em 0; }
nav.toc ul { columns: 2; }
hr { border: 0; border-top: 1px solid #d0d7de; margin: 1.6rem 0; }
"""


HOWTO = """
<h2>怎么读这五条线</h2>
<p>横轴都是时间，从左到右是评论窗往前走。不要把图读成二维地图：云心 <span class="math">m</span> 在时间图上只能画成
「现在还像不像开头」，也就是左边第一张 <span class="math">S</span>。</p>
<table>
<thead><tr><th>线</th><th>问的是</th><th>掉下去 / 变小</th><th>升上去 / 变大</th><th>平</th><th>中间鼓包再回来</th></tr></thead>
<tbody>
<tr><td>S</td><td>话题中心还像开场吗</td><td>搬走了</td><td>又像开场了</td><td>停在某处</td><td>绕了一圈（轨道）</td></tr>
<tr><td>σ</td><td>这会儿大家说得有多散</td><td>收紧（凝聚）</td><td>胀开（扩散）</td><td>散开程度没变</td><td>呼吸</td></tr>
<tr><td>d</td><td>同时占了几个意思方向</td><td>被压窄</td><td>岔路变多</td><td>方向数没变</td><td>—</td></tr>
<tr><td>k</td><td>是一团还是几派</td><td>派别合并</td><td>裂开（裂变）</td><td>没有新的对立派</td><td>—</td></tr>
<tr><td>χ</td><td>语义方位转了没</td><td>转回来</td><td>转开了</td><td>冻住（结晶）</td><td>绕圈 / 呼吸</td></tr>
</tbody>
</table>
<p>分类不是人眼扫图，是按整段走势打九个分、取最高。下面每条帖先用人话对线，再给分数。
S 和 χ 的纵轴是固定的，方便帖和帖比。σ、d 若只动了一点点，图会被放大，图下会标出来。</p>
"""


def build_html(tag: str, labeled: list[dict], sources: dict[str, dict], scales: dict, counts: dict[str, int]) -> str:
    chosen = pick_examples(labeled)
    n = len(labeled)
    toc = []
    body = []
    for mot in MOTIONS:
        n_hit = counts.get(mot, 0)
        toc.append(f'<li><a href="#{mot}">{html.escape(MOTION_ZH[mot])}</a>　{n_hit} / {n}</li>')
        story = CLASS_STORY[mot]
        head = f'<h2 id="{mot}">{html.escape(MOTION_ZH[mot])}　<span class="empty-note">{n_hit} 条</span></h2>'
        head += (
            '<div class="story">'
            f"<p><strong>要找的现象：</strong>{html.escape(story['want'])}</p>"
            f"<p><strong>五条线该长什么样：</strong>{html.escape(story['look'])}</p>"
            f"<p><strong>别和谁搞混：</strong>{html.escape(story['not'])}</p>"
            "</div>"
        )
        if n_hit == 0:
            head += "<p>本批没有人被判成这一类。下面放的是「这一行分数最高、但最终仍输给别人」的近失，用来看规则卡在哪里。</p>"
        elif n_hit == 1:
            head += "<p>只有 1 条，全部放出。</p>"
        elif mot == "translation":
            head += "<p>两条都是分差最大的平移。高分差平移本来就长得像，这不是抽样失败。</p>"
        cards = []
        for lab in chosen[mot]:
            src = sources[lab["post_id"]]
            feat = {k: (float("nan") if v is None else float(v)) for k, v in lab["features"].items()}
            _scores, terms, recipes = score_parts(feat, scales)
            cards.append(example_card(src, lab, terms, recipes, lab["scores"], mot))
        body.append(head + "".join(cards))

    count_rows = "".join(
        f"<tr><td>{html.escape(MOTION_ZH[m])}</td><td>{counts.get(m, 0)}</td>"
        f"<td>{100 * counts.get(m, 0) / n:.1f}%</td></tr>"
        for m in MOTIONS
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>整帖运动分类审计 · {html.escape(tag)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<h1>整帖运动分类审计 · {html.escape(tag)}</h1>
<p>用五条随时间变化的线，判断一场讨论在语义空间里怎么动。
一条帖只贴一个类名。机器给老师表里的九行各打一个分，最高的那行就是标签。成丝已去掉。</p>
<p>这页抽的是<strong>机器最有把握</strong>的例子：有赢家的类取分差最大的 1–2 条；
空着的类取「这一行分数最高、但没赢」的近失。样本是文件顺序前 {n} 条合格 Reddit 帖。</p>
<table>
<thead><tr><th>类</th><th>帖数</th><th>占比</th></tr></thead>
<tbody>{count_rows}</tbody>
</table>
{HOWTO}
<nav class="toc"><h2>目录</h2><ul>{"".join(toc)}</ul></nav>
<hr/>
{"".join(body)}
<p class="hint">由 <code>scripts/export_motion_audit_html.py</code> 生成。</p>
</div>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="chik200")
    args = ap.parse_args()

    src_path = ROOT / "outputs" / "order_params" / f"order_params_{args.tag}.jsonl"
    lab_path = ROOT / "outputs" / "labels" / f"thread_motions_{args.tag}.jsonl"
    sources = {
        json.loads(line)["post_id"]: json.loads(line)
        for line in src_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    labeled = [json.loads(line) for line in lab_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    feats = [{k: (float("nan") if v is None else float(v)) for k, v in r["features"].items()} for r in labeled]
    scales = ctm._scales(feats)
    counts = Counter(r["motion"] for r in labeled)
    html_text = build_html(args.tag, labeled, sources, scales, counts)
    out = ROOT / "docs" / f"thread_motions_{args.tag}.html"
    out.write_text(html_text, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
