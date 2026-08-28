"""Export a report markdown file to a self-contained HTML (images inlined)."""

from __future__ import annotations

import argparse
import base64
import mimetypes
import re
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MD = ROOT / "docs" / "phase1_status_report.md"
DEFAULT_HTML = ROOT / "docs" / "phase1_status_report.html"
FIG_DIR = ROOT / "docs" / "figures"

IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
MATH_RE = re.compile(r"\\\((.+?)\\\)")


def latex_inline_to_html(tex: str) -> str:
    s = tex.strip()
    s = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\overline\{([^}]+)\}", r"<span style='text-decoration:overline'>\1</span>", s)
    s = s.replace(r"\min", "min")
    s = s.replace(r"\cos", "cos")
    s = s.replace(r"\alpha", "α")
    s = s.replace(r"\sigma", "σ")
    s = s.replace(r"\chi", "χ")
    s = s.replace(r"\ge", "≥")
    s = s.replace(r"\le", "≤")
    s = s.replace(r"\to", "→")
    s = s.replace(r"\cdot", "·")
    s = s.replace(r"\Delta", "Δ")
    s = s.replace(r"\in", "∈")
    s = s.replace(r"\_", "_")
    s = re.sub(r"([A-Za-z])_\{([^}]+)\}", r"\1<sub>\2</sub>", s)
    s = re.sub(r"([A-Za-z])_([A-Za-z0-9]+)", r"\1<sub>\2</sub>", s)
    s = s.replace("^*", "<sup>*</sup>")
    return f'<span class="math">{s}</span>'


def inline_images(md_text: str, md_dir: Path) -> str:
    def repl(match: re.Match[str]) -> str:
        alt, src = match.group(1), match.group(2).strip()
        path = (md_dir / src).resolve()
        if not path.exists():
            raise FileNotFoundError(f"image not found: {src} -> {path}")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"![{alt}](data:{mime};base64,{b64})"

    return IMG_RE.sub(repl, md_text)


def protect_math(md_text: str) -> str:
    return MATH_RE.sub(lambda m: latex_inline_to_html(m.group(1)), md_text)


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
  max-width: 860px;
  margin: 0 auto;
  padding: 32px 24px 64px;
  background: #fff;
  box-shadow: 0 0 0 1px #d0d7de;
}
h1 { font-size: 1.7rem; line-height: 1.3; margin-top: 0; }
h2 { font-size: 1.35rem; border-bottom: 1px solid #d0d7de; padding-bottom: 6px; margin-top: 2rem; }
h3 { font-size: 1.15rem; margin-top: 1.6rem; }
h4 { font-size: 1.05rem; margin-top: 1.4rem; }
p, li { font-size: 15.5px; }
blockquote {
  margin: 1rem 0;
  padding: 0.2rem 1rem;
  color: #424a53;
  border-left: 4px solid #0969da;
  background: #f6f8fa;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 1rem 0;
  font-size: 14.5px;
}
th, td {
  border: 1px solid #d0d7de;
  padding: 6px 10px;
  text-align: left;
  vertical-align: top;
}
th { background: #f6f8fa; }
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 12px 0 4px;
  border: 1px solid #d0d7de;
}
code, pre {
  font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
  font-size: 13.5px;
}
code { background: #f6f8fa; padding: 0.1em 0.35em; border-radius: 4px; }
pre {
  background: #f6f8fa;
  padding: 12px 14px;
  overflow-x: auto;
  border: 1px solid #d0d7de;
  border-radius: 6px;
}
pre code { background: none; padding: 0; }
.math { font-family: "Cambria Math", "Times New Roman", serif; }
hr { border: 0; border-top: 1px solid #d0d7de; margin: 2rem 0; }
"""


def first_heading(md_text: str) -> str:
    for line in md_text.splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()
    return "report"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(DEFAULT_MD))
    ap.add_argument("--html", default="")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    md_path = Path(args.md)
    if not md_path.is_absolute():
        md_path = ROOT / md_path
    html_path = Path(args.html) if args.html else md_path.with_suffix(".html")
    if not html_path.is_absolute():
        html_path = ROOT / html_path

    raw = md_path.read_text(encoding="utf-8")
    title = args.title or first_heading(raw)
    raw = inline_images(raw, md_path.parent)
    raw = protect_math(raw)
    md = MarkdownIt("gfm-like", {"html": True, "linkify": False})
    body = md.render(raw)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
{body}
</div>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    size_mb = html_path.stat().st_size / (1024 * 1024)
    print(f"wrote {html_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
