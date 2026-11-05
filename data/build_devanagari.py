"""Build the Devanagari generation split: 30 tasks, all task text in Devanagari script.

The six hard families are unchanged - same oracle, same checks - but every
sentence the model reads is written in Devanagari, with only the proper names
and the ISO currency codes left in Latin. Numbers, amounts and paise are
identical in kind to `tasks_hard_eval.jsonl`, so the two files are comparable.

    python data/build_devanagari.py
    -> data/tasks_devanagari_eval.jsonl   (30, frozen by data/devanagari_eval.sha256)

Every row is replayed through the oracle and must score success=1, ter=0,
strict=1 before the pin is written.
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

from data.build_generated import NAMES, gender  # noqa: E402
from data.build_hard import add_action, expense, verify  # noqa: E402
from data.task_builder import A, finalize  # noqa: E402

PLAN = [
    ("settle5", 6),
    ("two_query", 6),
    ("trap_combo", 6),
    ("paraphrase", 6),
    ("convert_sum", 3),
    ("empty_balance", 3),
]

DESCRIPTIONS = {
    "pizza": "पिज़्ज़ा", "cab": "कैब", "dinner": "डिनर", "groceries": "किराना",
    "movie tickets": "मूवी टिकट", "petrol": "पेट्रोल", "hotel": "होटल",
    "flights": "फ्लाइट", "food": "खाना", "concert": "कॉन्सर्ट", "rent": "किराया",
    "electricity": "बिजली", "resort": "रिज़ॉर्ट", "taxi": "टैक्सी",
    "museum": "म्यूज़ियम", "party": "पार्टी", "decor": "सजावट", "cake": "केक",
    "drinks": "ड्रिंक्स", "brunch": "ब्रंच", "lunch": "लंच", "snacks": "स्नैक्स",
    "tickets": "टिकट", "wifi": "वाई-फाई", "furniture": "फर्नीचर",
    "shopping": "शॉपिंग", "coffee": "कॉफ़ी", "breakfast": "नाश्ता", "toll": "टोल",
    "parking": "पार्किंग", "boat ride": "बोट राइड", "guide": "गाइड",
    "camping": "कैंपिंग", "barbecue": "बारबेक्यू", "ice cream": "आइस क्रीम",
    "thali": "थाली", "auto": "ऑटो", "train": "ट्रेन", "bus": "बस",
    "dessert": "डेज़र्ट",
}

PARAPHRASE = [
    "{payer} ने {description} के लिए {amount} {currency} दिए ({people}). {person} का बैलेंस कितना है?",
    "{person} का बैलेंस बताओ. {payer} ने {amount} {currency} दिए थे {description} के लिए, {people} के साथ.",
    "{people} में से {person} का फाइनल बैलेंस क्या है? {payer} ने {amount} {currency} दिए {description} के लिए.",
]

EMPTY_BALANCE = [
    "{person} का बैलेंस कितना है? अभी तक कोई खर्च ऐड नहीं हुआ.",
    "लेजर खाली है. {person} का फाइनल बैलेंस बताओ.",
    "{person} ने कुछ नहीं दिया और किसी खर्च में शामिल नहीं है. उसका बैलेंस क्या है?",
]

CONVERT_SUM = [
    "{usd} USD और {eur} EUR को INR में बदलो और टोटल बताओ.",
    "टोटल कितना INR होगा अगर {usd} USD और {eur} EUR कन्वर्ट करें?",
]


def join(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} और {names[1]}"
    return ", ".join(names[:-1]) + f" और {names[-1]}"


def devanagari(item: dict) -> dict:
    return dict(item, description=DESCRIPTIONS[item["description"]])


def phrase(item: dict) -> str:
    return (
        f"{item['payer']} ने {item['amount']} {item['currency']} दिए "
        f"{item['description']} के लिए ({join(item['participants'])})"
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
    from data.build_hard import EUR_PAISE, USD_PAISE

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
    task = rng.choice(CONVERT_SUM).format(usd=usd, eur=eur)
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
    task = rng.choice(EMPTY_BALANCE).format(person=person)
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


def build(rng: random.Random) -> list[dict]:
    seen: set = set()
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
            task["id"] = f"dv-{len(tasks) + 1:03d}"
            task["family"] = kind
            tasks.append(task)
            made += 1
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--out-dir", default=str(ROOT / "data"))
    args = parser.parse_args()

    tasks = build(random.Random(args.seed))
    verify(tasks, "devanagari eval")

    out_dir = Path(args.out_dir)
    path = out_dir / "tasks_devanagari_eval.jsonl"
    path.write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks),
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (out_dir / "devanagari_eval.sha256").write_text(f"{digest}  tasks_devanagari_eval.jsonl\n", encoding="utf-8")
    print(f"wrote {len(tasks)} tasks -> {path}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
