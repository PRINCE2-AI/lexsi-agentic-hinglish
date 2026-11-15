"""Build a fresh Hindi probe: 30 tasks, all task text in Devanagari, five per family.

Same six hard families and same oracles as `tasks_devanagari_eval.jsonl`, but a
new seed, so no task text overlaps the earlier probe or any other split. This
file exists to answer one question with new evidence - does the shipped adapter
hold up on Devanagari it has never seen - not to replace the 257-task headline.

    python data/build_hindi30.py
    -> data/tasks_hindi30_eval.jsonl   (30, frozen by data/hindi30_eval.sha256)

Every row is replayed through the oracle and must score success=1, ter=0,
strict=1 before the pin is written. Task text is also checked against every
other data/tasks*.jsonl so the probe stays disjoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.build_devanagari import (  # noqa: E402
    CONVERT_SUM,
    DESCRIPTIONS,
    EMPTY_BALANCE,
    add_action,
    convert_sum,
    empty_balance,
    expense,
    join,
    verify,
)
from data.build_generated import NAMES, gender  # noqa: E402
from data.task_builder import A, finalize  # noqa: E402

PLAN = [
    ("settle5", 5),
    ("two_query", 5),
    ("trap_combo", 5),
    ("paraphrase", 5),
    ("convert_sum", 5),
    ("empty_balance", 5),
]

PARAPHRASE = [
    "{payer} ने {description} के लिए {amount} {currency} {paid} ({people}). {person} का बैलेंस कितना है?",
    "{person} का बैलेंस बताओ. {payer} ने {description} के लिए {amount} {currency} {paid}, {people} के साथ.",
    "{people} में से {person} का फाइनल बैलेंस क्या है? {payer} ने {description} के लिए {amount} {currency} {paid}.",
]


def paid(payer: str) -> str:
    return gender(payer, "दिए", "दीं")


def devanagari(item: dict) -> dict:
    return dict(item, description=DESCRIPTIONS[item["description"]])


def phrase(item: dict) -> str:
    return (
        f"{item['payer']} ने {item['description']} के लिए {item['amount']} "
        f"{item['currency']} {paid(item['payer'])} ({join(item['participants'])})"
    )


def settle5(rng: random.Random, used: set) -> dict:
    people = rng.sample(NAMES, 5)
    first = devanagari(expense(rng, used, people, currency="INR"))
    second = devanagari(expense(rng, used, people, currency=rng.choice(["USD", "EUR"])))
    task = (
        f"तीन दिन की ट्रिप, पाँच लोग: {phrase(first)}, और {phrase(second)}. "
        "सब बराबर हिस्सा ले रहे हैं. सेटल अप करो."
    )
    return {
        "task": task,
        "actions": [add_action(first), add_action(second), A("settle_up")],
        "expected_tools": ["add_expense", "settle_up"],
        "query": {"type": "settle"},
        "require_answer": True,
    }


def two_query(rng: random.Random, used: set) -> dict:
    people = rng.sample(NAMES, rng.choice([3, 4]))
    items = [devanagari(expense(rng, used, people)) for _ in range(rng.choice([1, 2]))]
    person = rng.choice(people)
    task = (
        "खर्च: " + ", ".join(phrase(item) for item in items) +
        f". पहले {person} का बैलेंस बताओ, फिर सेटल अप भी करो."
    )
    return {
        "task": task,
        "actions": [add_action(item) for item in items]
        + [A("get_balance", person=person), A("settle_up")],
        "expected_tools": ["add_expense", "get_balance", "settle_up"],
        "query": {"type": "balance_and_settle", "person": person},
        "require_answer": True,
    }


def trap_combo(rng: random.Random, used: set) -> dict:
    payer = rng.choice(NAMES)
    rest = [name for name in NAMES if name != payer]
    others = rng.sample(rest, rng.choice([3, 4]))
    outsider = rng.choice([name for name in rest if name not in others])
    first = devanagari(expense(rng, used, others, currency="USD", payer=payer))
    second = devanagari(expense(rng, used, others, currency="EUR", payer=payer))
    task = (
        f"{phrase(first)}, और {phrase(second)}. "
        f"{payer} खुद हिस्सा नहीं ले {gender(payer, 'रहा', 'रही')} है. "
        f"{outsider} का फाइनल बैलेंस कितना है?"
    )
    return {
        "task": task,
        "actions": [add_action(first), add_action(second), A("get_balance", person=outsider)],
        "expected_tools": ["add_expense", "get_balance"],
        "query": {"type": "balance", "person": outsider},
        "require_answer": True,
    }


def paraphrase(rng: random.Random, used: set) -> dict:
    people = rng.sample(NAMES, rng.choice([2, 3, 4]))
    item = devanagari(expense(rng, used, people))
    person = rng.choice(people)
    task = rng.choice(PARAPHRASE).format(
        payer=item["payer"],
        amount=item["amount"],
        currency=item["currency"],
        description=item["description"],
        people=join(item["participants"]),
        paid=paid(item["payer"]),
        person=person,
    )
    return {
        "task": task,
        "actions": [add_action(item), A("get_balance", person=person)],
        "expected_tools": ["add_expense", "get_balance"],
        "query": {"type": "balance", "person": person},
        "require_answer": True,
    }


BUILDERS = {
    "settle5": settle5,
    "two_query": two_query,
    "trap_combo": trap_combo,
    "paraphrase": paraphrase,
    "convert_sum": convert_sum,
    "empty_balance": empty_balance,
}


def existing_texts(out_path: Path) -> set[str]:
    texts: set[str] = set()
    for path in sorted((ROOT / "data").glob("tasks*.jsonl")):
        if path == out_path:
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if isinstance(row.get("task"), str):
                    texts.add(row["task"])
    return texts


def build(rng: random.Random, seen: set[str]) -> list[dict]:
    used: set = set()
    tasks: list[dict] = []
    for kind, count in PLAN:
        made = 0
        attempts = 0
        while made < count:
            attempts += 1
            if attempts > count * 200:
                raise RuntimeError(f"could not generate {count} unique {kind} tasks")
            scenario = BUILDERS[kind](rng, used)
            if scenario["task"] in seen:
                continue
            seen.add(scenario["task"])
            task = finalize(scenario)
            task["id"] = f"hi-{len(tasks) + 1:03d}"
            task["family"] = kind
            tasks.append(task)
            made += 1
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--out-dir", default=str(ROOT / "data"))
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    path = out_dir / "tasks_hindi30_eval.jsonl"
    tasks = build(random.Random(args.seed), existing_texts(path))
    verify(tasks, "hindi-30 eval")

    # newline="\n": a CRLF checkout hashes differently, and the pin must hold
    # on Linux too.
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"wrote {len(tasks)} tasks -> {path}")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    pin_path = out_dir / "hindi30_eval.sha256"
    if args.freeze or not pin_path.exists():
        pin_path.write_text(
            f"{digest}  tasks_hindi30_eval.jsonl\n", encoding="utf-8", newline="\n"
        )
        print(f"hindi-30 eval sha256: {digest} (frozen)")
    else:
        pinned = pin_path.read_text(encoding="utf-8").split()[0]
        if pinned != digest:
            raise SystemExit(
                f"hindi-30 eval drifted: pinned {pinned}, regenerated {digest}. "
                "Fix the generator, or re-freeze deliberately with --freeze."
            )
        print(f"hindi-30 eval sha256: {digest} (matches pin)")


if __name__ == "__main__":
    main()
