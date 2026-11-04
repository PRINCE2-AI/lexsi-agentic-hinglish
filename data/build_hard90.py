"""Build a 90-task held-out hard split, for a tighter confidence interval.

hard-30 is frozen and stays the headline number, but 30 tasks put the 96.67% row in
an 83–99% Wilson interval. This split triples the resolution (15 tasks per family)
without touching hard-30 or retraining anything: the same frozen adapter, the same
oracle, nine times the grading signal.

    python data/build_hard90.py
    -> data/tasks_hard90_eval.jsonl  (90, frozen by data/hard90_eval.sha256)

Text is deduplicated against hard-30 eval, hard train, the easy splits and the
safety split, so this is a fresh sample of the same six families. `convert_sum`
and `empty_balance` get wider pools here because their shared pools cannot yield
15 unseen rows each.
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

from data.build_generated import NAMES  # noqa: E402
from data.build_hard import BUILDERS, verify  # noqa: E402
from data.task_builder import A, finalize  # noqa: E402

PER_FAMILY = 15
PREFIX = "h9-"

EVAL_PLAN90 = [
    ("settle5", PER_FAMILY),
    ("two_query", PER_FAMILY),
    ("trap_combo", PER_FAMILY),
    ("paraphrase", PER_FAMILY),
    ("convert_sum", PER_FAMILY),
    ("empty_balance", PER_FAMILY),
]

USD90 = [7, 11.5, 13.25, 23, 29.99, 34, 41.25, 55, 68, 72.5, 88, 96.4, 104, 117, 130, 145]
EUR90 = [6.5, 9.75, 13, 17.2, 22, 28.4, 33, 39.5, 44, 52.6, 58, 66.7, 71, 79.9, 87, 92.35]

CONVERT90 = [
    "{usd} USD aur {eur} EUR ko INR me badlo aur total batao.",
    "Total kitna INR hoga agar {usd} USD aur {eur} EUR convert karein?",
    "Dono jod ke INR me: {usd} dollars aur {eur} euros.",
    "{usd} USD plus {eur} EUR — final INR total kya hai?",
]

EMPTY90 = [
    "{person} ka balance kitna hai? Abhi tak koi expense add nahi hua.",
    "Ledger khaali hai. {person} ka final balance batao.",
    "{person} ne kuch nahi diya aur kisi expense me shamil nahi hai. Uska balance kya hai?",
    "Shuruaat me hi poochh raha hoon: {person} ka balance? Koi kharcha record nahi hua.",
    "Koi entry nahi hai abhi. {person} ka balance kitna dikhega?",
]


def convert_sum90(rng: random.Random, used: set) -> dict:
    for _ in range(200):
        usd = rng.choice(USD90)
        eur = rng.choice(EUR90)
        key = ("convert_sum90", usd, eur)
        if key not in used:
            used.add(key)
            break
    else:
        raise RuntimeError("could not build a fresh convert_sum row")
    items = [{"amount": usd, "from": "USD"}, {"amount": eur, "from": "EUR"}]
    return {
        "task": rng.choice(CONVERT90).format(usd=usd, eur=eur),
        "actions": [
            A("convert_currency", amount=usd, from_currency="USD", to_currency="INR"),
            A("convert_currency", amount=eur, from_currency="EUR", to_currency="INR"),
        ],
        "expected_tools": ["convert_currency"],
        "query": {"type": "convert_sum", "items": items, "to": "INR"},
        "require_answer": True,
    }


def empty_balance90(rng: random.Random, used: set) -> dict:
    person = rng.choice(NAMES)
    return {
        "task": rng.choice(EMPTY90).format(person=person),
        "actions": [A("get_balance", person=person)],
        "expected_tools": ["get_balance"],
        "query": {"type": "balance", "person": person},
        "require_answer": True,
    }


BUILDERS90 = {
    **BUILDERS,
    "convert_sum": convert_sum90,
    "empty_balance": empty_balance90,
}


def known_texts() -> set:
    texts = set()
    for name in (
        "tasks_hard_eval.jsonl",
        "tasks_hard_train.jsonl",
        "tasks_train.jsonl",
        "tasks_eval.jsonl",
        "tasks.jsonl",
        "tasks_safety_eval.jsonl",
        "tasks_train_abstain.jsonl",
    ):
        path = ROOT / "data" / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            texts.update(json.loads(line)["task"] for line in handle if line.strip())
    return texts


def build(seed: int) -> list[dict]:
    rng = random.Random(seed)
    seen_text = known_texts()
    used_expenses: set = set()
    tasks = []
    for kind, count in EVAL_PLAN90:
        made = 0
        attempts = 0
        while made < count:
            attempts += 1
            if attempts > count * 200:
                raise RuntimeError(f"could not generate {count} unique {kind} tasks")
            scenario = BUILDERS90[kind](rng, used_expenses)
            if scenario["task"] in seen_text:
                continue
            seen_text.add(scenario["task"])
            task = finalize(scenario)
            task["id"] = f"{PREFIX}{len(tasks) + 1:03d}"
            task["family"] = kind
            tasks.append(task)
            made += 1
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--out-dir", default=str(ROOT / "data"))
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    tasks = build(args.seed)
    verify(tasks, "hard-90 eval")

    out_dir = Path(args.out_dir)
    path = out_dir / "tasks_hard90_eval.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"wrote {len(tasks)} -> {path}")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    pin_path = out_dir / "hard90_eval.sha256"
    if args.freeze or not pin_path.exists():
        pin_path.write_text(f"{digest}  tasks_hard90_eval.jsonl\n", encoding="utf-8", newline="\n")
        print(f"hard-90 eval sha256: {digest} (frozen)")
    else:
        pinned = pin_path.read_text(encoding="utf-8").split()[0]
        if pinned != digest:
            raise SystemExit(f"hard-90 eval drifted: pinned {pinned}, regenerated {digest}")
        print(f"hard-90 eval sha256: {digest} (matches pin)")

    families: dict[str, int] = {}
    for task in tasks:
        families[task["family"]] = families.get(task["family"], 0) + 1
    print("families:", json.dumps(families, sort_keys=True))


if __name__ == "__main__":
    main()
