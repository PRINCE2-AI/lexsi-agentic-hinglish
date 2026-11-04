"""Minimal pairs for the one confusion the abstention SFT introduced.

The round-2 failure analysis on hard-90 is unambiguous: `empty_balance` fell to
4/15, and in 11 of the 15 misses the model never called `get_balance`. It called
`finish` straight away and asked for a number, sometimes reproducing a whole
clarifying sentence from `tasks_train_abstain.jsonl`. The distinction it lost is
"the ledger is empty, so the balance is 0" versus "the sentence is missing an
amount/currency/share, so ask".

This builder writes the rows that make that distinction learnable:

    contrast_empty     empty ledger, complete question -> get_balance -> 0 INR
    contrast_settle    empty ledger, settle request    -> no transfers
    contrast_convert   two conversions whose sum must be stated exactly

    python data/build_contrast.py
    -> data/tasks_train_contrast.jsonl  (56 rows, oracle-verified)

Every row is replayed through the oracle and must score success=1, ter=0,
strict=1. Texts are deduplicated against every existing split, so a contrast row
can never be an eval row.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.build_generated import NAMES  # noqa: E402
from data.task_builder import A, finalize  # noqa: E402
from hinglish_agent.runner import run_oracle, score_episode  # noqa: E402
from hinglish_agent.tools import execute_tool  # noqa: E402

SEED = 20261105

# Phrasings that say the same thing as an abstention row ("koi kharcha nahi
# hua") but leave nothing unspecified: the answer is 0 and asking is wrong.
EMPTY = [
    "{person} ka balance kitna hai? Abhi tak koi expense add nahi hua.",
    "Ledger khaali hai. {person} ka final balance batao.",
    "{person} ne kuch nahi diya aur kisi expense me shamil nahi hai. Uska balance kya hai?",
    "Koi entry nahi hai abhi. {person} ka balance kitna dikhega?",
    "Shuruaat me hi poochh raha hoon: {person} ka balance? Koi kharcha record nahi hua.",
    "{person} ka balance batao — ledger me ek bhi kharcha nahi hai.",
    "Abhi tak kuch record nahi kiya. {person} ka balance kitna hoga?",
    "{person} ka final balance? Koi bill, koi split, kuch nahi hua.",
    "Ledger empty hai, {person} ka balance zero hoga ya kuch aur? Batao.",
    "{person} ka balance kitna hai? Kisi ne kuch share nahi kiya abhi tak.",
    "Zero kharcha hone par {person} ka balance kya dikhega?",
    "{person} ka hisaab batao — abhi tak koi payment nahi hui.",
    "Naya ledger hai, koi expense nahi. {person} ka balance kitna hai?",
    "{person} का balance कितना है? अभी तक कोई खर्चा record नहीं हुआ।",
    "Ledger खाली है. {person} का final balance बताओ.",
    "{person} ka balance batao, par dhyan rakho ki abhi koi kharcha nahi hai.",
]

SETTLE = [
    "Ledger khaali hai. Settle up karo.",
    "Koi expense record nahi hua. Sab ka hisaab settle kar do.",
    "Abhi tak kuch add nahi hua, phir bhi settle up karo.",
    "Zero expenses hain. Settlement batao.",
    "Kisi ne kuch diya nahi. Kaun kisko paise dega?",
    "Ledger me kuch nahi hai — settle up karke batao ki kuch transfer chahiye ya nahi.",
]

CONVERT = [
    "{usd} USD aur {eur} EUR ko INR me badlo aur total batao.",
    "Total kitna INR hoga agar {usd} USD aur {eur} EUR convert karein?",
    "{usd} USD plus {eur} EUR — final INR total kya hai?",
    "Donon ko INR me convert karo aur sum batao: {usd} USD, {eur} EUR.",
    "{usd} dollars aur {eur} euros ka INR me total kitna banega?",
]

# Distinct from the hard-90 pools on purpose: a shared (usd, eur) pair would be a
# duplicate of an eval scenario even when the wording differs.
USD_CONTRAST = [4.5, 8.25, 12.6, 16.75, 21.4, 26.5, 31.85, 38.2, 45.75, 52.4]
EUR_CONTRAST = [5.5, 10.25, 14.6, 19.75, 24.4, 31.5, 36.85, 42.2, 49.75, 57.4]

SPLITS = [
    "tasks.jsonl",
    "tasks_train.jsonl",
    "tasks_eval.jsonl",
    "tasks_hard_train.jsonl",
    "tasks_hard_eval.jsonl",
    "tasks_hard90_eval.jsonl",
    "tasks_train_abstain.jsonl",
    "tasks_safety_eval.jsonl",
]


def known_texts() -> set[str]:
    texts = set()
    for name in SPLITS:
        path = ROOT / "data" / name
        if not path.exists():
            raise SystemExit(f"missing split {name}; build it before the contrast set")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                texts.add(json.loads(line)["task"].strip().lower())
    return texts


def empty_rows(rng: random.Random, used: set) -> list[dict]:
    rows = []
    people = list(NAMES)
    rng.shuffle(people)
    for person in people:
        for _ in range(3):
            options = [
                t
                for t in EMPTY
                if t.format(person=person).strip().lower() not in used
            ]
            if not options:
                break
            task = rng.choice(options)
            task = task.format(person=person)
            used.add(task.strip().lower())
            rows.append(
                {
                    "task": task,
                    "actions": [A("get_balance", person=person)],
                    "expected_tools": ["get_balance"],
                    "query": {"type": "balance", "person": person},
                    "require_answer": True,
                }
            )
    return rows


def settle_rows(rng: random.Random, used: set) -> list[dict]:
    rows = []
    for template in SETTLE:
        task = template.format()
        if task.strip().lower() in used:
            continue
        used.add(task.strip().lower())
        rows.append(
            {
                "task": task,
                "actions": [A("settle_up")],
                "expected_tools": ["settle_up"],
                "query": {"type": "settle"},
                "require_answer": True,
            }
        )
    return rows


def convert_rows(rng: random.Random, used: set) -> list[dict]:
    rows = []
    pairs = [(usd, eur) for usd in USD_CONTRAST for eur in EUR_CONTRAST]
    rng.shuffle(pairs)
    for usd, eur in pairs:
        if len(rows) == len(USD_CONTRAST) * 2:
            break
        task = rng.choice(CONVERT).format(usd=usd, eur=eur)
        if task.strip().lower() in used:
            continue
        used.add(task.strip().lower())
        rows.append(
            {
                "task": task,
                "actions": [
                    A("convert_currency", amount=usd, from_currency="USD", to_currency="INR"),
                    A("convert_currency", amount=eur, from_currency="EUR", to_currency="INR"),
                ],
                "expected_tools": ["convert_currency"],
                "query": {
                    "type": "convert_sum",
                    "to": "INR",
                    "items": [
                        {"amount": usd, "from": "USD"},
                        {"amount": eur, "from": "EUR"},
                    ],
                },
                "require_answer": True,
                # Half the rows show the addition in integer paise: a small model
                # drops a digit adding two decimals, and 5979.75 vs 6979.75 is a
                # miss even when both conversions were right.
                "answer_format": "paise" if len(rows) % 2 else "plain",
            }
        )
    return rows


def paise_answer(task: dict) -> str:
    values = [
        execute_tool(
            "convert_currency",
            {"amount": item["amount"], "from_currency": item["from"], "to_currency": "INR"},
        )["value"]
        for item in task["query"]["items"]
    ]
    paise = [round(value * 100) for value in values]
    parts = [
        f"{item['amount']} {item['from']} = {value:.2f} INR"
        for item, value in zip(task["query"]["items"], values)
    ]
    total = sum(paise)
    return (
        "; ".join(parts)
        + f". Paise me: {' + '.join(str(p) for p in paise)} = {total} paise, "
        + f"yaani {total / 100:.2f} INR."
    )


def main() -> None:
    rng = random.Random(SEED)
    used = known_texts()

    scenarios = empty_rows(rng, used) + settle_rows(rng, used) + convert_rows(rng, used)

    tasks = []
    failures = []
    for index, scenario in enumerate(scenarios, start=1):
        task = finalize(scenario)
        if scenario.get("answer_format") == "paise":
            task["answer_text"] = paise_answer(scenario)
        task["id"] = f"c-t-{index:03d}"
        task["family"] = {
            "balance": "contrast_empty",
            "settle": "contrast_settle",
            "convert_sum": "contrast_convert",
        }[scenario["query"]["type"]]
        scores = score_episode(run_oracle(task), task)
        if scores["success"] != 1.0 or scores["ter"] != 0.0 or scores["strict"] != 1.0:
            failures.append((task["id"], scores))
        tasks.append(task)

    if failures:
        for task_id, scores in failures[:5]:
            print(f"FAIL {task_id}: {scores}")
        raise SystemExit(f"contrast: {len(failures)} oracle failures")

    out = ROOT / "data" / "tasks_train_contrast.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for task in tasks:
        counts[task["family"]] = counts.get(task["family"], 0) + 1
    print(f"contrast: {len(tasks)}/{len(tasks)} oracle success, 0 tool errors, strict JSON")
    print(f"wrote {len(tasks)} -> {out}")
    print("families:", json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
