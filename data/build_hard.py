"""Build the hard split: 30 held-out eval tasks and 40 training tasks.

The easy split (`tasks_train.jsonl` / `tasks_eval.jsonl`) puts one expense per
sentence, 2-4 participants and a single question at the end. This split keeps
the same oracle but forces behaviours the easy split never tests:

    settle5        five-way split with paise remainders, two currencies
    two_query      balance *and* settle up in one task
    trap_combo     payer is not a participant, queried person is in no expense
    paraphrase     reordered / half-Devanagari wording, same oracle
    convert_sum    two conversions that must be added
    empty_balance  balance on an empty ledger

    python data/build_hard.py
    -> data/tasks_hard_eval.jsonl   (30, frozen by data/hard_eval.sha256)
    -> data/tasks_hard_train.jsonl  (40)

Every row is replayed through the oracle and must score success=1, ter=0,
strict=1. Text is deduplicated across both splits.
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

from data.build_generated import DESCRIPTIONS, NAMES, gender, join_names  # noqa: E402
from data.task_builder import A, finalize  # noqa: E402
from hinglish_agent.runner import run_oracle, score_episode  # noqa: E402

INR_PAISE = [1001, 1234.56, 2003, 2500.37, 3301, 4501.75, 1601, 875.43]
USD_PAISE = [19, 47, 47.11, 61, 83.07]
EUR_PAISE = [19.07, 37, 61.23, 83]
POOLS = {"INR": INR_PAISE, "USD": USD_PAISE, "EUR": EUR_PAISE}

EVAL_PLAN = [
    ("settle5", 6),
    ("two_query", 6),
    ("trap_combo", 6),
    ("paraphrase", 6),
    ("convert_sum", 3),
    ("empty_balance", 3),
]

TRAIN_PLAN = [
    ("settle5", 8),
    ("two_query", 8),
    ("trap_combo", 8),
    ("paraphrase", 8),
    ("convert_sum", 4),
    ("empty_balance", 4),
]

PARAPHRASE = [
    "{payer} ne {amount} {currency} diye {description} ke liye ({people}). {person} ka balance kitna hai?",
    "{description} के लिए {payer} ने {amount} {currency} दिए ({people}). {person} का balance कितना है?",
    "Balance kitna hai {person} ka? {payer} ne {amount} {currency} diye the {description} ke liye, {people} ke saath.",
    "{people} में से {person} ko batao — {payer} ne {amount} {currency} diye {description} के लिए, final balance kya hai?",
]

EMPTY_BALANCE = [
    "{person} ka balance kitna hai? Abhi tak koi expense add nahi hua.",
    "Ledger khaali hai. {person} ka final balance batao.",
    "{person} ne kuch nahi diya aur kisi expense me shamil nahi hai. Uska balance kya hai?",
]


def expense(rng: random.Random, used: set, participants: list[str],
            currency: str | None = None, payer: str | None = None,
            amount=None) -> dict:
    for _ in range(200):
        who = payer or rng.choice(participants)
        cur = currency or rng.choice(["INR", "USD", "EUR"])
        amt = amount if amount is not None else rng.choice(POOLS[cur])
        description = rng.choice(DESCRIPTIONS)
        key = (who, amt, cur, tuple(participants), description)
        if key not in used:
            used.add(key)
            return {
                "payer": who,
                "amount": amt,
                "currency": cur,
                "participants": list(participants),
                "description": description,
            }
    raise RuntimeError("could not build a fresh expense")


def phrase(item: dict) -> str:
    return (
        f"{item['payer']} ne {item['amount']} {item['currency']} diye "
        f"{item['description']} ke liye ({join_names(item['participants'])})"
    )


def add_action(item: dict) -> dict:
    return A("add_expense", **{k: item[k] for k in
                               ("payer", "amount", "currency", "participants", "description")})


def settle5(rng: random.Random, used: set) -> dict:
    people = rng.sample(NAMES, 5)
    first = expense(rng, used, people, currency="INR")
    second = expense(rng, used, people, currency=rng.choice(["USD", "EUR"]))
    task = (
        f"Trip ke teen din, paanch log: {phrase(first)}, aur {phrase(second)}. "
        "Har koi barabar share kar raha hai. Settle up karo."
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
    items = [expense(rng, used, people) for _ in range(rng.choice([1, 2]))]
    person = rng.choice(people)
    task = (
        "Expenses: " + ", ".join(phrase(item) for item in items) +
        f". Pehle {person} ka balance batao, phir settle up bhi karo."
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
    first = expense(rng, used, others, currency="USD", payer=payer)
    second = expense(rng, used, others, currency="EUR", payer=payer)
    task = (
        f"{phrase(first)}, aur {phrase(second)}. "
        f"{payer} khud share nahi kar {gender(payer, 'raha', 'rahi')}. "
        f"{outsider} ka final balance kitna hai?"
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
    item = expense(rng, used, people)
    person = rng.choice(people)
    template = rng.choice(PARAPHRASE)
    task = template.format(
        payer=item["payer"],
        amount=item["amount"],
        currency=item["currency"],
        description=item["description"],
        people=join_names(item["participants"]),
        person=person,
    )
    return {
        "task": task,
        "actions": [add_action(item), A("get_balance", person=person)],
        "expected_tools": ["add_expense", "get_balance"],
        "query": {"type": "balance", "person": person},
        "require_answer": True,
    }


def convert_sum(rng: random.Random, used: set) -> dict:
    for _ in range(200):
        usd = rng.choice(USD_PAISE)
        eur = rng.choice(EUR_PAISE)
        key = ("convert_sum", usd, eur)
        if key not in used:
            used.add(key)
            break
    else:
        raise RuntimeError("could not build a fresh convert_sum")
    items = [{"amount": usd, "from": "USD"}, {"amount": eur, "from": "EUR"}]
    task = rng.choice(
        [
            f"{usd} USD aur {eur} EUR ko INR me badlo aur total batao.",
            f"Total kitna INR hoga agar {usd} USD aur {eur} EUR convert karein?",
        ]
    )
    return {
        "task": task,
        "actions": [
            A("convert_currency", amount=usd, from_currency="USD", to_currency="INR"),
            A("convert_currency", amount=eur, from_currency="EUR", to_currency="INR"),
        ],
        "expected_tools": ["convert_currency"],
        "query": {"type": "convert_sum", "items": items, "to": "INR"},
        "require_answer": True,
    }


def empty_balance(rng: random.Random, used: set) -> dict:
    person = rng.choice(NAMES)
    template = rng.choice(EMPTY_BALANCE)
    task = template.format(person=person)
    return {
        "task": task,
        "actions": [A("get_balance", person=person)],
        "expected_tools": ["get_balance"],
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


def build_split(plan, prefix: str, rng: random.Random, seen_text: set,
                used_expenses: set) -> list[dict]:
    tasks = []
    for kind, count in plan:
        made = 0
        attempts = 0
        while made < count:
            attempts += 1
            if attempts > count * 200:
                raise RuntimeError(f"could not generate {count} unique {kind} tasks")
            scenario = BUILDERS[kind](rng, used_expenses)
            if scenario["task"] in seen_text:
                continue
            seen_text.add(scenario["task"])
            task = finalize(scenario)
            task["id"] = f"{prefix}{len(tasks) + 1:03d}"
            task["family"] = kind
            tasks.append(task)
            made += 1
    return tasks


def verify(tasks: list[dict], name: str) -> None:
    failures = []
    for task in tasks:
        scores = score_episode(run_oracle(task), task)
        if scores["success"] != 1.0 or scores["ter"] != 0.0 or scores["strict"] != 1.0:
            failures.append((task["id"], scores))
    if failures:
        for task_id, scores in failures[:5]:
            print(f"FAIL {task_id}: {scores}")
        raise SystemExit(f"{name}: {len(failures)} oracle failures")
    print(f"{name}: {len(tasks)}/{len(tasks)} oracle success, 0 tool errors, strict JSON")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--out-dir", default=str(ROOT / "data"))
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="rewrite data/hard_eval.sha256; without it a drifted eval file aborts",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seen_text: set = set()
    used_expenses: set = set()

    eval_tasks = build_split(EVAL_PLAN, "h-e-", rng, seen_text, used_expenses)
    train_tasks = build_split(TRAIN_PLAN, "h-t-", rng, seen_text, used_expenses)

    verify(eval_tasks, "hard eval")
    verify(train_tasks, "hard train")

    out_dir = Path(args.out_dir)
    written = {}
    for filename, tasks in (
        ("tasks_hard_eval.jsonl", eval_tasks),
        ("tasks_hard_train.jsonl", train_tasks),
    ):
        path = out_dir / filename
        # newline="\n": the frozen hash must be byte-identical on Windows and
        # Linux, otherwise the pin only holds on the machine that wrote it.
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for task in tasks:
                handle.write(json.dumps(task, ensure_ascii=False) + "\n")
        written[filename] = path
        print(f"wrote {len(tasks)} -> {path}")

    digest = hashlib.sha256(written["tasks_hard_eval.jsonl"].read_bytes()).hexdigest()
    pin_path = out_dir / "hard_eval.sha256"
    if args.freeze or not pin_path.exists():
        pin_path.write_text(f"{digest}  tasks_hard_eval.jsonl\n", encoding="utf-8", newline="\n")
        print(f"hard eval sha256: {digest} (frozen)")
    else:
        pinned = pin_path.read_text(encoding="utf-8").split()[0]
        if pinned != digest:
            raise SystemExit(
                f"hard eval drifted: pinned {pinned}, regenerated {digest}. "
                "Fix the generator, or re-freeze deliberately with --freeze."
            )
        print(f"hard eval sha256: {digest} (matches pin)")

    families = {}
    for task in eval_tasks:
        families[task["family"]] = families.get(task["family"], 0) + 1
    print("eval families:", json.dumps(families, sort_keys=True))


if __name__ == "__main__":
    main()
