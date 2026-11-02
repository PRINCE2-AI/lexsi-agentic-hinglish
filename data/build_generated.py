"""Generate oracle-verified train/eval splits.

The starter set (`data/tasks.jsonl`) has 20 hand-written tasks. SFT needs
enough supervised turns to actually move the policy, and evaluation needs tasks
the model never trained on. This script builds both deterministically:

    python data/build_generated.py
    -> data/tasks_train.jsonl  (100 tasks)
    -> data/tasks_eval.jsonl   (24 held-out tasks)

Every task is replayed through the oracle and must score success=1, ter=0.
Task text is deduplicated across both splits.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.task_builder import A, finalize  # noqa: E402
from hinglish_agent.runner import run_oracle, score_episode  # noqa: E402

NAMES = ["Rahul", "Priya", "Amit", "Neha", "Sara", "Vikram"]
FEMALE = {"Priya", "Neha", "Sara"}
DESCRIPTIONS = [
    "pizza", "cab", "dinner", "groceries", "movie tickets", "petrol", "hotel",
    "flights", "food", "concert", "rent", "electricity", "resort", "taxi",
    "museum", "party", "decor", "cake", "drinks", "brunch", "lunch", "snacks",
    "tickets", "wifi", "furniture", "shopping", "coffee", "breakfast", "toll",
    "parking", "boat ride", "guide", "camping", "barbecue", "ice cream",
    "thali", "auto", "train", "bus", "dessert",
]
INR_AMOUNTS = [150, 200, 250, 300, 400, 450, 500, 600, 750, 800, 900, 1000,
               1200, 1500, 1800, 2000, 2400, 2500, 3000, 3600, 4000, 4500, 5000]
USD_AMOUNTS = [5, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 80, 90, 100]
EUR_AMOUNTS = [5, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 80, 90, 100]
INR_TO_USD = [830, 1660, 2490, 4150, 8300, 16600]
INR_TO_EUR = [900, 1800, 2700, 4500, 9000, 18000]


def join_names(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} aur {names[1]}"
    return ", ".join(names[:-1]) + " aur " + names[-1]


def gender(name: str, male: str, female: str) -> str:
    return female if name in FEMALE else male


def pick_amount(rng: random.Random, currency: str) -> int:
    pool = {"INR": INR_AMOUNTS, "USD": USD_AMOUNTS, "EUR": EUR_AMOUNTS}[currency]
    return rng.choice(pool)


def pick_expense(rng: random.Random, used: set) -> dict:
    for _ in range(50):
        participants = rng.sample(NAMES, rng.choice([2, 2, 3, 4]))
        payer = rng.choice(NAMES)
        currency = rng.choice(["INR", "USD", "EUR"])
        amount = pick_amount(rng, currency)
        description = rng.choice(DESCRIPTIONS)
        key = (payer, amount, currency, tuple(participants), description)
        if key not in used:
            used.add(key)
            return {
                "payer": payer,
                "amount": amount,
                "currency": currency,
                "participants": participants,
                "description": description,
            }
    raise RuntimeError("could not build a fresh expense")


def expense_phrase(expense: dict) -> str:
    names = join_names(expense["participants"])
    return (
        f"{expense['payer']} ne {expense['amount']} {expense['currency']} "
        f"diye {expense['description']} ke liye ({names})"
    )


def balance_single(rng: random.Random, used: set) -> dict:
    expense = pick_expense(rng, used)
    person = rng.choice(expense["participants"] + [n for n in NAMES if n not in expense["participants"]])
    names = join_names(expense["participants"])
    count = len(expense["participants"])
    task = (
        f"{expense['payer']} ne {expense['amount']} {expense['currency']} diye "
        f"{expense['description']} ke liye. {count} log the: {names}. "
        f"{person} ka final balance kitna hai?"
    )
    return {
        "task": task,
        "actions": [A("add_expense", **{k: expense[k] for k in
                                        ("payer", "amount", "currency", "participants", "description")}),
                    A("get_balance", person=person)],
        "expected_tools": ["add_expense", "get_balance"],
        "query": {"type": "balance", "person": person},
    }


def balance_multi(rng: random.Random, used: set) -> dict:
    count = rng.choice([2, 2, 3])
    expenses = [pick_expense(rng, used) for _ in range(count)]
    person = rng.choice(NAMES)
    parts = [
        f"{e['payer']} ne {e['amount']} {e['currency']} diye {e['description']} ke liye "
        f"({join_names(e['participants'])})"
        for e in expenses
    ]
    lead = rng.choice(["Expenses", "Trip ke expenses"])
    task = f"{lead}: " + ", ".join(parts) + f". {person} ka final balance kitna hai?"
    return {
        "task": task,
        "actions": [A("add_expense", **{k: e[k] for k in
                                        ("payer", "amount", "currency", "participants", "description")})
                    for e in expenses] + [A("get_balance", person=person)],
        "expected_tools": ["add_expense", "get_balance"],
        "query": {"type": "balance", "person": person},
    }


def balance_payer_out(rng: random.Random, used: set) -> dict:
    expense = pick_expense(rng, used)
    payer = expense["payer"]
    participants = [n for n in expense["participants"] if n != payer]
    if len(participants) < 2:
        participants = [n for n in NAMES if n != payer][:3]
    expense["participants"] = participants
    names = join_names(participants)
    task = (
        f"{payer} ne {expense['amount']} {expense['currency']} diye {names} ke liye. "
        f"{payer} khud share nahi kar {gender(payer, 'raha', 'rahi')}. "
        f"{payer} ka balance kitna hai?"
    )
    return {
        "task": task,
        "actions": [A("add_expense", **{k: expense[k] for k in
                                        ("payer", "amount", "currency", "participants", "description")}),
                    A("get_balance", person=payer)],
        "expected_tools": ["add_expense", "get_balance"],
        "query": {"type": "balance", "person": payer},
    }


def balance_outsider(rng: random.Random, used: set) -> dict:
    expense = pick_expense(rng, used)
    candidates = [n for n in NAMES if n not in expense["participants"]]
    person = rng.choice(candidates)
    names = join_names(expense["participants"])
    task = (
        f"{expense['payer']} ne {expense['amount']} {expense['currency']} diye {names} ke liye. "
        f"{person} is trip me shamil nahi {gender(person, 'tha', 'thi')}. "
        f"{person} ka balance kya hai?"
    )
    return {
        "task": task,
        "actions": [A("add_expense", **{k: expense[k] for k in
                                        ("payer", "amount", "currency", "participants", "description")}),
                    A("get_balance", person=person)],
        "expected_tools": ["add_expense", "get_balance"],
        "query": {"type": "balance", "person": person},
    }


def settle_task(rng: random.Random, used: set, count: int) -> dict:
    expenses = [pick_expense(rng, used) for _ in range(count)]
    if count == 1:
        e = expenses[0]
        task = (
            f"{e['payer']} ne {e['amount']} {e['currency']} diye {join_names(e['participants'])} ke liye. "
            "Settle up karo."
        )
    else:
        parts = [expense_phrase(e) for e in expenses]
        lead = rng.choice(["Trip ke expenses", "Expenses"])
        task = f"{lead}: " + ", ".join(parts) + ". Settle up karo."
    return {
        "task": task,
        "actions": [A("add_expense", **{k: e[k] for k in
                                        ("payer", "amount", "currency", "participants", "description")})
                    for e in expenses] + [A("settle_up")],
        "expected_tools": ["add_expense", "settle_up"],
        "query": {"type": "settle"},
    }


def convert_task(rng: random.Random, style: str) -> dict:
    if style == "usd_inr":
        amount, source, target = rng.choice(USD_AMOUNTS), "USD", "INR"
        task = f"{amount} USD kitne INR hote hain?"
    elif style == "eur_inr":
        amount, source, target = rng.choice(EUR_AMOUNTS), "EUR", "INR"
        task = f"{amount} EUR ko INR me convert karo."
    elif style == "inr_usd":
        amount, source, target = rng.choice(INR_TO_USD), "INR", "USD"
        task = f"{amount} INR kitne USD hain?"
    elif style == "inr_eur":
        amount, source, target = rng.choice(INR_TO_EUR), "INR", "EUR"
        task = f"{amount} INR ko EUR me badlo."
    else:
        raise ValueError(style)
    return {
        "task": task,
        "actions": [A("convert_currency", amount=amount, from_currency=source, to_currency=target)],
        "expected_tools": ["convert_currency"],
        "query": {"type": "convert", "amount": amount, "from": source, "to": target},
    }


TRAIN_PLAN = [
    ("balance_single", 25),
    ("balance_multi", 12),
    ("balance_payer_out", 5),
    ("balance_outsider", 3),
    ("settle_1", 15),
    ("settle_2", 10),
    ("settle_3", 7),
    ("settle_4", 3),
    ("convert_usd_inr", 7),
    ("convert_eur_inr", 6),
    ("convert_inr_usd", 4),
    ("convert_inr_eur", 3),
]

EVAL_PLAN = [
    ("balance_single", 5),
    ("balance_multi", 3),
    ("balance_payer_out", 1),
    ("balance_outsider", 1),
    ("settle_1", 2),
    ("settle_2", 3),
    ("settle_3", 2),
    ("settle_4", 1),
    ("convert_usd_inr", 2),
    ("convert_eur_inr", 2),
    ("convert_inr_usd", 1),
    ("convert_inr_eur", 1),
]


def build_one(kind: str, rng: random.Random, used: set) -> dict:
    if kind.startswith("balance_"):
        return {
            "balance_single": balance_single,
            "balance_multi": balance_multi,
            "balance_payer_out": balance_payer_out,
            "balance_outsider": balance_outsider,
        }[kind](rng, used)
    if kind.startswith("settle_"):
        return settle_task(rng, used, int(kind.split("_")[1]))
    if kind.startswith("convert_"):
        return convert_task(rng, kind.split("_", 1)[1])
    raise ValueError(kind)


def build_split(plan: list[tuple[str, int]], prefix: str, rng: random.Random,
                seen_text: set, used_expenses: set) -> list[dict]:
    tasks = []
    for kind, count in plan:
        made = 0
        attempts = 0
        while made < count:
            attempts += 1
            if attempts > count * 100:
                raise RuntimeError(f"could not generate {count} unique {kind} tasks")
            scenario = build_one(kind, rng, used_expenses)
            if scenario["task"] in seen_text:
                continue
            seen_text.add(scenario["task"])
            task = finalize(scenario)
            task["id"] = f"{prefix}{len(tasks) + 1:03d}"
            tasks.append(task)
            made += 1
    return tasks


def verify(tasks: list[dict], name: str) -> None:
    failures = []
    for task in tasks:
        episode = run_oracle(task)
        scores = score_episode(episode, task)
        if scores["success"] != 1.0 or scores["ter"] != 0.0 or scores["strict"] != 1.0:
            failures.append((task["id"], scores))
    if failures:
        for task_id, scores in failures[:5]:
            print(f"FAIL {task_id}: {scores}")
        raise SystemExit(f"{name}: {len(failures)} oracle failures")
    print(f"{name}: {len(tasks)}/{len(tasks)} oracle success, 0 tool errors, strict JSON")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--out-dir", default=str(ROOT / "data"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seen_text: set = set()
    used_expenses: set = set()

    eval_tasks = build_split(EVAL_PLAN, "e", rng, seen_text, used_expenses)
    train_tasks = build_split(TRAIN_PLAN, "g", rng, seen_text, used_expenses)

    verify(train_tasks, "train")
    verify(eval_tasks, "eval")

    out_dir = Path(args.out_dir)
    for filename, tasks in (("tasks_train.jsonl", train_tasks), ("tasks_eval.jsonl", eval_tasks)):
        with (out_dir / filename).open("w", encoding="utf-8") as handle:
            for task in tasks:
                handle.write(json.dumps(task, ensure_ascii=False) + "\n")
        print(f"wrote {len(tasks)} -> {out_dir / filename}")

    longest = max(len(json.dumps(t["task"], ensure_ascii=False)) for t in train_tasks + eval_tasks)
    print(f"longest task text: {longest} chars")


if __name__ == "__main__":
    main()
