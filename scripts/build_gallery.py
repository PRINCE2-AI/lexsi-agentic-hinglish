"""Render the 90 held-out tasks as a static page: base model and shipped adapter, side by side.

    python scripts/build_gallery.py                          # write gallery/index.html
    python scripts/build_gallery.py --repo primce/hinglish-episodes

Every row is a recorded run (`runs/base_hard90`, `runs/sft4_hard90`), not a re-generation: the
tool trace, the closing ledger and the final answer are copied from `trajectories.jsonl` and the
per-task verdicts from `results.jsonl`. No GPU and no model are involved.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hinglish_agent.ledger import public_balances  # noqa: E402
OUT = ROOT / "gallery"
TASKS = ROOT / "data" / "tasks_hard90_eval.jsonl"
RUNS = {"base": ROOT / "runs" / "base_hard90", "sft4": ROOT / "runs" / "sft4_hard90"}
SIDES = {"base": "Base Tiny Aya", "sft4": "Shipped adapter (sft4)"}


def load_tasks() -> list[dict]:
    with TASKS.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_run(path: Path) -> dict[str, dict]:
    episodes: dict[str, dict] = {}
    with (path / "trajectories.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                episodes[row["metadata"]["task_id"]] = row
    with (path / "results.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                episodes[row["task_id"]]["scores"] = row
    return episodes


def final_balances(steps: list[dict]) -> dict[str, float]:
    for step in reversed(steps):
        ledger = (step.get("result") or {}).get("ledger")
        if ledger:
            return public_balances(ledger)
    return {}


def short(value, limit: int = 150) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def verdict(scores: dict) -> tuple[str, str]:
    if scores["success"]:
        return "match", "MATCH — closing ledger and answer match the oracle"
    bits = [
        "ledger " + ("ok" if scores["ledger_ok"] else "wrong"),
        "answer " + ("ok" if scores["answer_ok"] else "wrong"),
        f"{scores['calls']} calls",
    ]
    if scores["errors"]:
        errors = scores["errors"]
        bits.append(f"{errors} tool error" + ("" if errors == 1 else "s"))
    return "miss", "MISMATCH — " + ", ".join(bits)


def trace_html(steps: list[dict]) -> str:
    items = []
    for step in steps:
        if not step.get("name"):
            items.append("<li class=parse>unparsed turn</li>")
            continue
        ok = (step.get("result") or {}).get("ok", True)
        extra = ""
        result = step.get("result") or {}
        if step["name"] == "add_expense" and result.get("total_inr") is not None:
            extra = f" -> total {result['total_inr']} INR, share {result.get('share_inr')}"
        if step["name"] == "settle_up" and result.get("transfers"):
            extra = f" -> {len(result['transfers'])} transfers"
        if not ok:
            extra = f" -> error: {short(result.get('error'), 90)}"
        items.append(
            f"<li><code>{html.escape(step['name'])}</code>"
            f"<span class=args>{html.escape(short(step.get('arguments'), 120))}</span>"
            f"<span class=res>{html.escape(extra)}</span></li>"
        )
    return "<ol class=steps>" + "".join(items) + "</ol>"


def balances_html(got: dict[str, float], expected: dict[str, float]) -> str:
    if not got:
        return "<p class=empty>no ledger written</p>"
    rows = []
    for person in expected:
        got_value = got.get(person)
        if got_value is None:
            rows.append(f"<tr><td>{html.escape(person)}</td><td>missing</td><td>{expected[person]:.2f}</td></tr>")
        else:
            flag = "" if abs(got_value - expected[person]) < 0.005 else " class=bad"
            rows.append(
                f"<tr{flag}><td>{html.escape(person)}</td>"
                f"<td>{got_value:.2f}</td><td>{expected[person]:.2f}</td></tr>"
            )
    return (
        "<table class=ledger><tr><th></th><th>got</th><th>oracle</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def side_html(side: str, episode: dict, task: dict) -> str:
    scores = episode["scores"]
    kind, text = verdict(scores)
    steps = episode.get("steps") or []
    got = final_balances(steps)
    answer = episode.get("final_response") or "(the model never called `finish`)"
    return (
        f"<section class='side {kind}'>"
        f"<h3>{SIDES[side]}</h3>"
        f"<p class=verdict>{html.escape(text)}</p>"
        f"{balances_html(got, task.get('expected_balances') or {})}"
        f"<details><summary>Trace ({len(steps)} steps)</summary>{trace_html(steps)}</details>"
        f"<details><summary>Final answer</summary><p class=answer>{html.escape(answer)}</p></details>"
        "</section>"
    )


def page(tasks: list[dict], episodes: dict[str, dict[str, dict]]) -> str:
    rows = []
    stats = {"n": len(tasks), "base": 0, "sft4": 0, "base_only": 0, "both_miss": 0}
    for task in tasks:
        base = episodes["base"][task["id"]]
        sft4 = episodes["sft4"][task["id"]]
        base_ok = bool(base["scores"]["success"])
        sft4_ok = bool(sft4["scores"]["success"])
        stats["base"] += base_ok
        stats["sft4"] += sft4_ok
        stats["base_only"] += base_ok and not sft4_ok
        stats["both_miss"] += not base_ok and not sft4_ok
        flag = "adapter-miss" if base_ok and not sft4_ok else ("adapter-match" if sft4_ok else "both-miss")
        expected = task.get("expected_balances") or {}
        oracle = ", ".join(f"{name} {value:+.2f}" for name, value in expected.items()) or "no ledger"
        rows.append(
            f"<article class=ep data-family='{html.escape(task['family'])}' data-flag='{flag}'>"
            f"<header><span class=id>{html.escape(task['id'])}</span>"
            f"<span class=chip>{html.escape(task['family'])}</span>"
            f"<p class=task>{html.escape(task['task'])}</p></header>"
            f"<div class=cols>{side_html('base', base, task)}{side_html('sft4', sft4, task)}</div>"
            f"<details class=oracle><summary>Oracle</summary>"
            f"<p>Balances: {html.escape(oracle)}</p>"
            f"<p>Answer: {html.escape(task.get('answer_text') or '')}</p></details>"
            "</article>"
        )

    families = sorted({task["family"] for task in tasks})
    family_options = "".join(f"<option value='{html.escape(name)}'>{html.escape(name)}</option>" for name in families)
    return TEMPLATE.replace("<!--ROWS-->", "\n".join(rows)).replace(
        "<!--STATS-->",
        f"{stats['n']} held-out tasks · shipped adapter {stats['sft4']}/{stats['n']} · "
        f"base {stats['base']}/{stats['n']} · base passes where the adapter fails: {stats['base_only']} · "
        f"both fail: {stats['both_miss']}",
    ).replace("<!--FAMILIES-->", family_options)


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Held-out hard-90 — every episode, base vs adapter</title>
<style>
:root { color-scheme: light dark; --line:#d0d7de; --muted:#57606a; --ok:#1a7f37; --bad:#cf222e; --bg:#fff; --fg:#1f2328; --soft:#f6f8fa; }
@media (prefers-color-scheme: dark) {
  :root { --line:#30363d; --muted:#8b949e; --ok:#3fb950; --bad:#f85149; --bg:#0d1117; --fg:#e6edf3; --soft:#161b22; }
}
* { box-sizing: border-box; }
body { margin:0 auto; max-width:1180px; padding:2rem 1.2rem 4rem; font:15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color:var(--fg); background:var(--bg); }
a { color:#0969da; text-decoration:none; } a:hover { text-decoration:underline; }
h1 { font-size:1.7rem; margin:0 0 .4rem; }
p.stats { font-size:1.02rem; margin:.2rem 0 1rem; }
p.note { color:var(--muted); font-size:.9rem; margin-bottom:1.2rem; }
.controls { position:sticky; top:0; z-index:5; display:flex; gap:.5rem; flex-wrap:wrap; padding:.7rem 0; background:var(--bg); border-bottom:1px solid var(--line); }
.controls input, .controls select { font:inherit; padding:.35rem .5rem; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--fg); }
.controls input { flex:1 1 220px; }
.ep { border:1px solid var(--line); border-radius:10px; margin:1rem 0; padding:.8rem 1rem; }
.ep header { display:flex; gap:.6rem; align-items:baseline; flex-wrap:wrap; }
.id { font-weight:600; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; }
.chip { font-size:.78rem; padding:.05rem .45rem; border-radius:999px; background:var(--soft); border:1px solid var(--line); color:var(--muted); }
.task { flex:1 1 100%; margin:.25rem 0 .6rem; }
.cols { display:grid; grid-template-columns:1fr 1fr; gap:.9rem; }
@media (max-width:820px) { .cols { grid-template-columns:1fr; } }
.side { border:1px solid var(--line); border-left:4px solid var(--line); border-radius:8px; padding:.6rem .7rem; background:var(--soft); }
.side.match { border-left-color:var(--ok); }
.side.miss { border-left-color:var(--bad); }
.side h3 { font-size:.86rem; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin:0 0 .3rem; }
.verdict { margin:.1rem 0 .5rem; font-size:.92rem; }
.match .verdict { color:var(--ok); } .miss .verdict { color:var(--bad); }
table.ledger { border-collapse:collapse; font-size:.85rem; margin:.2rem 0 .5rem; }
table.ledger th, table.ledger td { border:1px solid var(--line); padding:.1rem .4rem; text-align:right; }
table.ledger td:first-child, table.ledger th:first-child { text-align:left; }
table.ledger tr.bad td { background:rgba(207,34,46,.12); }
p.empty { font-size:.85rem; color:var(--muted); margin:.1rem 0 .5rem; }
details { margin:.25rem 0; }
summary { cursor:pointer; font-size:.86rem; color:var(--muted); }
ol.steps { margin:.3rem 0; padding-left:1.1rem; font-size:.83rem; }
ol.steps li { margin:.15rem 0; }
ol.steps code { background:var(--bg); padding:.02rem .25rem; border-radius:4px; }
.args { color:var(--muted); margin-left:.35rem; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.78rem; }
.res { color:var(--muted); }
.answer { font-size:.88rem; white-space:pre-wrap; margin:.2rem 0; }
details.oracle { margin-top:.5rem; }
</style>
</head>
<body>
<h1>Held-out hard-90 — every episode, side by side</h1>
<p class=stats><!--STATS--></p>
<p class=note>Recorded runs only: the trace, the closing ledger and the answer come from
<code>runs/base_hard90</code> and <code>runs/sft4_hard90</code> (also on Hugging Face as
<code>primce/lexsi-runs</code>). Scoring is the repository's own oracle over ledger state; no LLM
judge and no regeneration. This is the seed-42 run; three seeds pooled are <b>251/270</b>
(<a href="https://github.com/PRINCE2-AI/lexsi-agentic-hinglish/blob/main/docs/STATS.md">docs/STATS.md</a>).
Six tasks fail for both models — the <b>Both fail</b> filter lists them.
<a href="https://github.com/PRINCE2-AI/lexsi-agentic-hinglish">Repo</a> ·
<a href="https://huggingface.co/spaces/primce/hinglish-expense-agent">Live demo</a> ·
<a href="https://huggingface.co/spaces/primce/hinglish-track3-article">Write-up</a> ·
<a href="https://huggingface.co/spaces/primce/hinglish-attacks">72 injection episodes, guard off vs on</a></p>
<div class=controls>
  <input id=search type=search placeholder="Search task text, id or family…">
  <select id=family><option value="">All families</option><!--FAMILIES--></select>
  <select id=flag>
    <option value="">All results</option>
    <option value="adapter-match">Adapter matches</option>
    <option value="adapter-miss">Adapter misses</option>
    <option value="both-miss">Both fail</option>
  </select>
</div>
<main id=list>
<!--ROWS-->
</main>
<script>
const search = document.getElementById('search');
const family = document.getElementById('family');
const flag = document.getElementById('flag');
const episodes = [...document.querySelectorAll('.ep')];
function apply() {
  const needle = search.value.trim().toLowerCase();
  const fam = family.value, kind = flag.value;
  for (const ep of episodes) {
    const text = ep.textContent.toLowerCase();
    const show = (!needle || text.includes(needle)) &&
                 (!fam || ep.dataset.family === fam) &&
                 (!kind || ep.dataset.flag === kind);
    ep.style.display = show ? '' : 'none';
  }
}
search.addEventListener('input', apply);
family.addEventListener('change', apply);
flag.addEventListener('change', apply);
const params = new URLSearchParams(location.hash.replace(/^#/, ''));
if (params.get('family')) family.value = params.get('family');
if (params.get('flag')) flag.value = params.get('flag');
if (params.get('q')) search.value = params.get('q');
apply();
</script>
</body>
</html>
"""

SPACE_README = """---
title: Lexsi Track 3 - held-out hard-90 episodes
sdk: static
pinned: false
---

# Held-out hard-90, every episode

Static page rendered from the recorded runs by `scripts/build_gallery.py` in the GitHub repo:
base model and shipped adapter side by side on all 90 unseen tasks, with the tool trace, the
closing ledger and the per-task verdict. No GPU, no model, nothing regenerated.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="", help="e.g. primce/hinglish-episodes")
    args = parser.parse_args()

    tasks = load_tasks()
    episodes = {side: load_run(path) for side, path in RUNS.items()}
    missing = [task["id"] for task in tasks if task["id"] not in episodes["base"] or task["id"] not in episodes["sft4"]]
    if missing:
        raise SystemExit(f"no recorded episode for {len(missing)} tasks, e.g. {missing[:3]}")

    body = page(tasks, episodes)
    OUT.mkdir(exist_ok=True)
    (OUT / "index.html").write_text(body, encoding="utf-8")
    (OUT / "README.md").write_text(SPACE_README, encoding="utf-8")
    print(f"wrote {OUT / 'index.html'} ({len(body) // 1024} KB, {len(tasks)} tasks)")

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
