"""Render `docs/hf_article.md` as a static page and publish it as a free HF Space.

Hub Articles need PRO (or a Team/Enterprise org), so a static Space is the free way to
host the same long-form write-up on the Hub.

    python scripts/build_article_space.py                    # write article/index.html
    python scripts/build_article_space.py --repo primce/hinglish-track3-article
"""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "hf_article.md"
OUT = ROOT / "article"

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light dark; }}
body {{
  margin: 0 auto; max-width: 780px; padding: 2.5rem 1.25rem 5rem;
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: #1f2328; background: #fff;
}}
a {{ color: #0969da; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
h1 {{ font-size: 2rem; line-height: 1.25; margin: 0 0 .5rem; }}
h2 {{ margin-top: 2.5rem; border-bottom: 1px solid #d0d7de; padding-bottom: .3rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1.25rem 0; font-size: .94rem; }}
th, td {{ border: 1px solid #d0d7de; padding: .4rem .6rem; text-align: left; }}
th {{ background: #f6f8fa; }}
td:nth-child(n+2), th:nth-child(n+2) {{ text-align: right; }}
code {{ background: #f6f8fa; padding: .1rem .3rem; border-radius: 4px; font-size: .9em; }}
pre {{ background: #f6f8fa; padding: 1rem; border-radius: 6px; overflow-x: auto; }}
pre code {{ background: none; padding: 0; }}
blockquote {{ margin: 1rem 0; padding: 0 1rem; color: #57606a; border-left: 4px solid #d0d7de; }}
.result {{
  margin: 1.25rem 0 1.75rem; padding: 1rem 1.15rem;
  border: 1px solid #1a7f37; border-left: 6px solid #1a7f37; border-radius: 8px;
  background: #f2fdf5; font-size: 1.02rem;
}}
.result p {{ margin: .4rem 0; }}
.result strong {{ color: #0f5132; }}
.links {{ font-size: .92rem; margin-bottom: 2rem; }}
.links a {{ margin-right: 1rem; }}
@media (prefers-color-scheme: dark) {{
  body {{ color: #e6edf3; background: #0d1117; }}
  a {{ color: #58a6ff; }}
  h2 {{ border-color: #30363d; }}
  th, td {{ border-color: #30363d; }}
  th, code, pre {{ background: #161b22; }}
  blockquote {{ color: #8b949e; border-color: #30363d; }}
  .result {{ background: #0f1f16; border-color: #2ea043; }}
  .result strong {{ color: #7ee787; }}
}}
</style>
</head>
<body>
<nav class="links">
<a href="https://github.com/PRINCE2-AI/lexsi-agentic-hinglish">GitHub</a>
<a href="https://huggingface.co/spaces/primce/hinglish-expense-agent">Live demo</a>
<a href="https://huggingface.co/datasets/primce/lexsi-runs/resolve/main/demo/lexsi_track3_demo.mp4">Demo video</a>
<a href="https://huggingface.co/primce/tiny-aya-hinglish-lora">Adapter</a>
<a href="https://huggingface.co/datasets/primce/lexsi-runs">Run logs</a>
</nav>
{body}
</body>
</html>
"""

SPACE_README = """---
title: Lexsi Track 3 - Tiny Aya Hinglish agent
sdk: static
pinned: false
---

# Lexsi Track 3 write-up

Static page rendered from `docs/hf_article.md` in the GitHub repo. Hub Articles require a
PRO subscription, so the same article is hosted here as a free static Space.
"""


def render() -> str:
    text = SOURCE.read_text(encoding="utf-8")
    title = text.splitlines()[0].lstrip("# ").strip()
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
        output_format="html5",
    )
    return TEMPLATE.format(title=html.escape(title), body=body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="", help="e.g. primce/hinglish-track3-article")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    (OUT / "index.html").write_text(render(), encoding="utf-8")
    (OUT / "README.md").write_text(SPACE_README, encoding="utf-8")
    print("wrote", OUT / "index.html")

    if not args.repo:
        return

    from huggingface_hub import HfApi, create_repo
    from huggingface_hub.constants import HF_TOKEN_PATH

    token = os.environ.get("HF_TOKEN")
    if not token and not Path(HF_TOKEN_PATH).exists():
        raise SystemExit("Not logged in. Run `hf auth login` first.")

    create_repo(args.repo, repo_type="space", space_sdk="static", exist_ok=True)
    api = HfApi(token=token)
    for name in ("index.html", "README.md"):
        api.upload_file(
            repo_id=args.repo,
            repo_type="space",
            path_or_fileobj=str(OUT / name),
            path_in_repo=name,
        )
        print("uploaded", name)
    print(f"live: https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
