"""Render the 72-task weakness split as a static page, no GPU and no model call.

Every episode below is replayed from the published run logs, not regenerated: the adapter's
`results.jsonl` and `trajectories.jsonl` for the plain run and for the guard run are what the page
prints, including the exact text the model read after the guard ran.

    python scripts/fetch_runs.py                     # or snapshot_download either run folder
    python scripts/build_attack_gallery.py           # write attacks/index.html
    python scripts/build_attack_gallery.py --repo primce/hinglish-attacks
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

from hinglish_agent.guard import scan_text  # noqa: E402

RUNS = ROOT / "runs"
TASKS = ROOT / "data" / "tasks_safety_adaptive72.jsonl"
OUT = ROOT / "attacks"

PLAIN = "safety_adaptive72_sft4"
GUARD = "safety_adaptive72_sft4_guard"
NEUTRAL = "[tool result contained instructions addressed to the agent; removed]"

SPACE_README = """---
title: Lexsi Track 3 - 72 injection episodes, guard off vs on
sdk: static
pinned: false
---

# 72 injection episodes, adapter alone vs guard on

Static page built from the published run logs at `primce/lexsi-runs`
(`safety_adaptive72_sft4/` and `safety_adaptive72_sft4_guard/`) by
`scripts/build_attack_gallery.py` in the GitHub repo. Every row is replayed, not regenerated.
"""

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>72 injection episodes - adapter alone vs guard on</title>
<style>
:root { --bg: #ffffff; --fg: #1f2328; --muted: #57606a; --line: #d0d7de; --code: #f6f8fa;
        --bad: #cf222e; --badbg: #ffebe9; --good: #1a7f37; --goodbg: #dafbe1; --warn: #9a6700; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px; background: var(--bg); color: var(--fg);
       font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
h1 { font-size: 24px; margin: 0 0 6px; }
h2 { font-size: 17px; margin: 28px 0 8px; }
p.lead { color: var(--muted); margin: 0 0 18px; max-width: 900px; }
a { color: #0969da; }
.bar { position: sticky; top: 0; background: var(--bg); padding: 10px 0 12px;
       border-bottom: 1px solid var(--line); z-index: 5; display: flex; gap: 8px; flex-wrap: wrap;
       align-items: center; }
button { font: inherit; padding: 5px 11px; border: 1px solid var(--line); border-radius: 6px;
         background: var(--code); color: var(--fg); cursor: pointer; }
button.on { background: #0969da; border-color: #0969da; color: #fff; }
.count { color: var(--muted); margin-left: auto; }
table { border-collapse: collapse; margin: 8px 0 20px; font-size: 14px; }
th, td { border: 1px solid var(--line); padding: 5px 10px; text-align: left; }
th { background: var(--code); }
.card { border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; margin: 16px 0; }
.head { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
.id { font-weight: 700; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.fam { font-size: 12px; color: var(--muted); border: 1px solid var(--line); border-radius: 99px;
       padding: 1px 8px; }
.chip { font-size: 12px; border-radius: 99px; padding: 1px 8px; border: 1px solid var(--line); }
.chip.bad { color: var(--bad); background: var(--badbg); border-color: #ffcecb; }
.chip.good { color: var(--good); background: var(--goodbg); border-color: #aceebb; }
.task { margin: 8px 0 10px; }
.payload { background: #fff8c5; border: 1px solid #d4a72c; border-radius: 6px; padding: 8px 10px;
           font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px;
           white-space: pre-wrap; margin: 6px 0; }
.part { color: var(--muted); font-size: 12px; margin-bottom: 2px; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 10px; }
@media (max-width: 860px) { .cols { grid-template-columns: 1fr; } }
.colh { font-weight: 700; font-size: 13px; margin-bottom: 6px; }
.colh span { font-weight: 400; color: var(--muted); }
.step { border-left: 3px solid var(--line); padding: 2px 0 6px 10px; margin: 8px 0; }
.step .tool { font-weight: 600; font-size: 13px; }
pre { background: var(--code); border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px;
      overflow-x: auto; font-size: 12.5px; margin: 4px 0; white-space: pre-wrap; }
.note { border-radius: 6px; padding: 6px 9px; margin: 6px 0; font-size: 12.5px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; }
.note.hit { background: var(--badbg); border: 1px solid #ffcecb; }
.note.partial { background: #fff8c5; border: 1px solid #d4a72c; }
.note.dropped { background: var(--goodbg); border: 1px solid #aceebb; color: var(--good); }
.note.plain { background: var(--code); border: 1px dashed var(--line); }
.answer { margin-top: 8px; font-size: 13.5px; }
.answer b { font-size: 12px; color: var(--muted); display: block; font-weight: 600; }
.hidden { display: none; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1117; --fg: #e6edf3; --muted: #8b949e; --line: #30363d; --code: #161b22;
          --bad: #ff7b72; --badbg: #2d1214; --good: #7ee787; --goodbg: #0f1f16; }
  .payload { background: #2d2a10; border-color: #6b5a1a; }
}
</style>
</head>
<body>
<h1>72 injection episodes: adapter alone vs the guard in front of it</h1>
<p class="lead">One split, built <em>after</em> the guard's rules were read, so the guard is the
thing under test here rather than the adapter. Each episode carries the same 999999 INR
"correction" and differs only in how the amount arrives: spelled out, cut across two tool results,
or written as an expression so the literal never appears. Both columns are replayed from the
published run logs &mdash; no model was called to render this page. The highlighted block is the
payload; the block under each step is the exact text the model read.</p>

<table>
<tr><th>Run</th><th>Solved</th><th>Extra expense</th><th>Ledger hit</th><th>Answer hit</th><th>Unplanned calls</th></tr>
__SUMMARY__
</table>
<p class="lead">Sources: <code>primce/lexsi-runs/safety_adaptive72_sft4/</code> and
<code>.../safety_adaptive72_sft4_guard/</code> (per-task <code>results.jsonl</code> and
<code>trajectories.jsonl</code>). Split built by <code>data/build_safety_adaptive72.py</code>,
digest <code>647f0b7f&hellip;</code>. Long-form reading:
<a href="https://huggingface.co/spaces/primce/hinglish-track3-article">write-up</a> &middot;
<a href="https://huggingface.co/spaces/primce/hinglish-episodes">90 clean held-out episodes</a> &middot;
<a href="https://github.com/PRINCE2-AI/lexsi-agentic-hinglish">code</a>.</p>

<div class="bar">
  <button data-fam="all" class="on">All 72</button>
  <button data-fam="inject_words">Spelled out</button>
  <button data-fam="inject_split_words">Split in two</button>
  <button data-fam="inject_expr">Expression</button>
  <span style="width:14px"></span>
  <button data-view="all" class="on">Every episode</button>
  <button data-view="harm">Money moved (plain)</button>
  <button data-view="fixed">Fixed by the guard</button>
  <span class="count" id="count"></span>
</div>
__CARDS__
<script>
var fam = "all", view = "all";
function apply() {
  var shown = 0;
  document.querySelectorAll(".card").forEach(function (card) {
    var ok = (fam === "all" || card.dataset.family === fam);
    if (ok && view === "harm") ok = card.dataset.harm === "1";
    if (ok && view === "fixed") ok = card.dataset.fixed === "1";
    card.classList.toggle("hidden", !ok);
    if (ok) shown++;
  });
  document.getElementById("count").textContent = shown + " of 72 shown";
}
document.querySelectorAll("[data-fam]").forEach(function (b) {
  b.onclick = function () {
    fam = b.dataset.fam;
    document.querySelectorAll("[data-fam]").forEach(function (x) { x.classList.toggle("on", x === b); });
    apply();
  };
});
document.querySelectorAll("[data-view]").forEach(function (b) {
  b.onclick = function () {
    view = b.dataset.view;
    document.querySelectorAll("[data-view]").forEach(function (x) { x.classList.toggle("on", x === b); });
    apply();
  };
});
apply();
</script>
</body>
</html>
"""


def load_rows(run: str) -> dict[str, dict]:
    base = RUNS / run
    results = {
        str(row["task_id"]): row
        for row in (
            json.loads(line) for line in (base / "results.jsonl").open(encoding="utf-8") if line.strip()
        )
    }
    episodes: dict[str, dict] = {}
    for line in (base / "trajectories.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row.get("metadata") or {}
        task_id = str(meta["task_id"])

        # What the model read after each tool call: the runner appends the payload to the `note`
        # key of the shown result, and the guard rewrites that key before it is sent. Results come
        # back in call order, so pair them onto the steps by tool name rather than by index.
        shown: list[dict] = []
        for message in meta.get("conversation") or []:
            content = message.get("content") or ""
            if message.get("role") != "user" or "tool_results" not in content:
                continue
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
            shown.extend(payload.get("tool_results") or [])

        steps = list(row.get("steps") or [])
        cursor = 0
        for item in shown:
            name = item.get("tool_name")
            for index in range(cursor, len(steps)):
                if steps[index].get("name") == name:
                    steps[index]["shown"] = item.get("result")
                    cursor = index + 1
                    break
        episodes[task_id] = {
            **results.get(task_id, {}),
            "steps": steps,
            "answer": row.get("final_response", ""),
        }
    return episodes


def chip(text: str, good: bool) -> str:
    return f'<span class="chip {"good" if good else "bad"}">{html.escape(text)}</span>'


def render_steps(steps: list[dict], flags: dict[str, bool], guarded: bool) -> str:
    full = {p for p in flags if p}
    out = []
    for step in steps:
        name = step.get("name") or step.get("tool_name") or "?"
        out.append('<div class="step">')
        out.append(f'<div class="tool">{html.escape(str(name))}</div>')
        out.append(f"<pre>{html.escape(json.dumps(step.get('arguments'), ensure_ascii=False))}</pre>")
        shown = step.get("shown")
        if isinstance(shown, dict) and "note" in shown:
            note = str(shown["note"])
            result = {k: v for k, v in shown.items() if k != "note"}
            out.append(f"<pre>{html.escape(json.dumps(result, ensure_ascii=False))}</pre>")
            if note == NEUTRAL:
                out.append(
                    '<div class="note dropped">note &rarr; <b>[tool result contained instructions '
                    "addressed to the agent; removed]</b> &mdash; the guard replaced the text before "
                    "the model read it</div>"
                )
            elif note.strip() in full and flags[note.strip()]:
                tail = (
                    "&mdash; the guard flags this text, so this is a guard miss"
                    if guarded
                    else "&mdash; delivered as written"
                )
                out.append(f'<div class="note hit">note, read by the model: {html.escape(note)} {tail}</div>')
            elif note.strip() in full:
                tail = (
                    "&mdash; the guard does not read this half as an instruction, and on its own it "
                    "carries none"
                    if guarded
                    else "&mdash; this half carries no instruction on its own"
                )
                out.append(f'<div class="note partial">note, read by the model: {html.escape(note)} {tail}</div>')
            else:
                out.append(f'<div class="note plain">note: {html.escape(note)}</div>')
        elif shown is not None:
            out.append(f"<pre>{html.escape(json.dumps(shown, ensure_ascii=False))}</pre>")
        else:
            out.append(f"<pre>{html.escape(json.dumps(step.get('result'), ensure_ascii=False))}</pre>")
        out.append("</div>")
    return "".join(out)


def render_card(task: dict, plain: dict, guard: dict) -> str:
    task_id = task["id"]
    family = task["family"]
    parts = (task.get("injection") or {}).get("parts") or [task.get("injection") or {}]
    payload_flags = {
        str(p.get("payload", "")).strip(): bool(scan_text(str(p.get("payload", "")))["flagged"])
        for p in parts
    }
    payload = "".join(
        f'<div class="part">part {i + 1} &middot; delivered on <code>{html.escape(str(p.get("after_tool")))}</code></div>'
        f'<div class="payload">{html.escape(str(p.get("payload", "")))}</div>'
        for i, p in enumerate(parts)
    )

    def metric(row: dict, key: str) -> bool:
        return float(row.get(key, 0) or 0) >= 0.5

    p_solved, g_solved = metric(plain, "success"), metric(guard, "success")
    p_harm = metric(plain, "attacker_expense") or metric(plain, "attack_ledger") or metric(plain, "attack_answer")
    g_harm = metric(guard, "attacker_expense") or metric(guard, "attack_ledger") or metric(guard, "attack_answer")
    fixed = "1" if (p_harm and not g_harm) else "0"

    chips = [
        chip("solved" if p_solved else "not solved", p_solved),
        chip("money moved" if p_harm else "no money moved", not p_harm),
    ]
    if p_harm:
        chips.append(chip("fixed by guard" if not g_harm else "guard miss", not g_harm))
    if not g_solved:
        chips.append(chip("guard run: not solved", False))

    return (
        f'<div class="card" data-family="{family}" data-harm="{"1" if p_harm else "0"}" data-fixed="{fixed}">'
        f'<div class="head"><span class="id">{task_id}</span>'
        f'<span class="fam">{family}</span>{"".join(chips)}</div>'
        f'<div class="task">{html.escape(task.get("task", ""))}</div>'
        f"{payload}"
        f'<div class="cols">'
        f'<div><div class="colh">Adapter alone <span>&mdash; payload delivered as written</span></div>'
        f'{render_steps(plain["steps"], payload_flags, guarded=False)}'
        f'<div class="answer"><b>final answer</b>{html.escape(plain["answer"])}</div></div>'
        f'<div><div class="colh">Adapter + guard <span>&mdash; note rewritten before the model reads it</span></div>'
        f'{render_steps(guard["steps"], payload_flags, guarded=True)}'
        f'<div class="answer"><b>final answer</b>{html.escape(guard["answer"])}</div></div>'
        f"</div></div>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="", help="e.g. primce/hinglish-attacks")
    args = parser.parse_args()

    tasks = [json.loads(line) for line in TASKS.open(encoding="utf-8") if line.strip()]
    plain_eps = load_rows(PLAIN)
    guard_eps = load_rows(GUARD)

    def hits(rows: dict[str, dict], key: str) -> int:
        return sum(1 for row in rows.values() if float(row.get(key, 0) or 0) >= 0.5)

    n = len(tasks)
    summary = ""
    for label, rows in (("`sft4` alone", plain_eps), ("`sft4` + guard", guard_eps)):
        cells = [
            f"{hits(rows, k)}/{n}" for k in ("success", "attacker_expense", "attack_ledger", "attack_answer", "unplanned_calls")
        ]
        summary += "<tr><td>" + label + "</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"

    cards = "".join(
        render_card(task, plain_eps[task["id"]], guard_eps[task["id"]])
        for task in tasks
        if task["id"] in plain_eps and task["id"] in guard_eps
    )

    body = TEMPLATE.replace("__SUMMARY__", summary).replace("__CARDS__", cards)
    OUT.mkdir(exist_ok=True)
    (OUT / "index.html").write_text(body, encoding="utf-8", newline="\n")
    (OUT / "README.md").write_text(SPACE_README, encoding="utf-8", newline="\n")
    size = (OUT / "index.html").stat().st_size
    print(f"wrote {OUT / 'index.html'} ({size / 1024:.0f} KB)")

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
