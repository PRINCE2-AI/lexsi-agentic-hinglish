"""Build the abstention training split.

The safety split measures what the agent does when the user never supplied enough
information to answer. The frozen adapter invents participants and records an
expense instead of asking: `abstained` 1/8, `invented_expense` 7/8. This script
builds the training data for the behaviour that is missing.

Every row is a minimal pair over one expense, so the policy can learn the boundary
and not the template:

    underspecified   one field withheld  -> the right answer is a question
    twin             that field present  -> the oracle records the expense and answers

    python data/build_abstain.py
    -> data/tasks_train_abstain.jsonl  (64 rows: 32 underspecified + 32 twins)

The eight pinned safety tasks are held out: a pair whose text or expense signature
collides with the safety split aborts the build.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.build_generated import DESCRIPTIONS, NAMES, gender, join_names, pick_amount  # noqa: E402
from data.task_builder import A, finalize  # noqa: E402
from hinglish_agent.runner import run_oracle  # noqa: E402
from hinglish_agent.safety import score_safety  # noqa: E402

PER_KIND = 8

MISSING_PEOPLE = [
    "{payer} ne {amount} {currency} diye {description} ke liye. {count} log the, par naam yaad nahi. "
    "{payer} ka balance batao.",
    "Hum {count} log the {description} pe, {payer} ne {amount} {currency} diye. Kisi ke naam yaad nahi. "
    "{payer} ka final balance kya hai?",
    "{description} ke liye {payer} ne {amount} {currency} diya. Group me {count} log the, naam bhool gaya. "
    "{payer} ko kitna milna chahiye?",
    "{payer} ne {amount} {currency} diye {description} ke liye, {count} logon me. Naam list nahi hai. "
    "{payer} ka balance kitna hai?",
]

MISSING_PEOPLE_ASKS = [
    "Kaun kaun the? {count} logon ke naam batao, phir split kar ke balance bata dunga.",
    "Participants ke naam missing hain — kaun kaun tha? Naam batao.",
    "{count} logon ke naam kya hain? Sirf count se split nahi kar sakta.",
    "Kis kis ne share kiya? Naam batao, phir {payer} ka balance nikal dunga.",
]

MISSING_AMOUNT = [
    "{payer} ne {description} ke liye kuch paise diye ({people}). {payer} ka balance batao.",
    "{description} ka bill {payer} ne bhara, {people} the. {payer} ko kitna milna chahiye?",
    "{people} ne {description} share kiya, paise {payer} ne diye. {payer} ka final balance kya hai?",
    "{payer} ne {description} ke liye payment kiya tha, amount yaad nahi. {people} the. "
    "{payer} ka balance batao.",
]

MISSING_AMOUNT_ASKS = [
    "{payer} ne exactly kitne diye the? Amount batao, phir balance nikal dunga.",
    "Amount missing hai — {payer} ne kitne paise diye the?",
    "{payer} ne kitna pay kiya tha {description} ke liye? Number batao.",
    "Paisa kitna tha? Amount ke bina {payer} ka balance nahi nikal sakta.",
]

MISSING_CURRENCY = [
    "{payer} ne {amount} diye {description} ke liye ({people}). {payer} ka balance batao.",
    "{description} ke liye {payer} ne {amount} diya, {people} the. {payer} ko kitna milega?",
    "{people} me {description} ka kharcha {amount} hua, {payer} ne diya. {payer} ka balance kya hai?",
    "{payer} ne {amount} diya {description} ke liye, sab {people} ne share kiya. "
    "{payer} ka final balance batao.",
]

MISSING_CURRENCY_ASKS = [
    "{amount} kaunsi currency me the — INR, USD ya EUR? Currency batao.",
    "Amount ki currency missing hai — INR, USD ya EUR?",
    "{amount} rupees the, dollars the ya euro? Currency batao phir balance nikalta hoon.",
    "Currency confirm karo ({amount} INR/USD/EUR?), warna conversion galat hoga.",
]

AMBIGUOUS_SHARE = [
    "{payer} ne {amount} {currency} diye {description} ke liye. {people} pakka the, par ek aur dost "
    "aaya tha — pata nahi wo share kar {ra} ya nahi. {payer} ka balance batao.",
    "{description} ke liye {payer} ne {amount} {currency} diye, {people} the. Ek extra banda bhi tha, "
    "uska share confirm nahi. {payer} ka final balance kya hai?",
    "{payer} ne {amount} {currency} diye {description} ke liye ({people}). Ek dost baad me join hua, "
    "usne share kiya ya nahi clear nahi. {payer} ko kitna milna chahiye?",
    "{people} aur ek dost, sab {description} pe the. {payer} ne {amount} {currency} diye. "
    "Extra dost ka share pakka nahi. {payer} ka balance batao.",
]

AMBIGUOUS_SHARE_ASKS = [
    "Wo extra dost share kar {ra} hai ya nahi? Confirm karo, phir {payer} ka balance bata dunga.",
    "Baad me jo dost join hua, usne paise share kiye ya nahi? Ye batao.",
    "Extra dost ka share count karna hai ya nahi? Confirm karo pehle.",
    "Wo teesra banda {description} ka hissa de raha hai ya sirf aaya tha? Batao, phir split karunga.",
]


def clean(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())


def sample_expense(rng: random.Random, used: set) -> dict:
    for _ in range(200):
        participants = rng.sample(NAMES, rng.choice([2, 3, 4]))
        payer = rng.choice(participants)
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


def underspecified(task: str, answer_text: str) -> dict:
    return {
        "task": task,
        "expected_tools": [],
        "checks_ledger": False,
        "expected_balances": {},
        "expected_settlements": [],
        "answer_contains": ["?"],
        "answer_text": answer_text,
        "require_answer": True,
        "actions": [],
        "abstain": True,
    }


def twin(item: dict) -> dict:
    return finalize(
        {
            "task": (
                f"{item['payer']} ne {item['amount']} {item['currency']} diye {item['description']} "
                f"ke liye ({join_names(item['participants'])}). {item['payer']} ka balance batao."
            ),
            "actions": [
                A(
                    "add_expense",
                    **{key: item[key] for key in
                       ("payer", "amount", "currency", "participants", "description")},
                ),
                A("get_balance", person=item["payer"]),
            ],
            "expected_tools": ["add_expense", "get_balance"],
            "query": {"type": "balance", "person": item["payer"]},
            "require_answer": True,
        }
    )


def pair(kind: str, item: dict, rng: random.Random) -> tuple[dict, dict]:
    payer = item["payer"]
    people = join_names(item["participants"])
    if kind == "missing_people":
        text = rng.choice(MISSING_PEOPLE).format(
            payer=payer, amount=item["amount"], currency=item["currency"],
            description=item["description"], count=len(item["participants"]),
        )
        ask = rng.choice(MISSING_PEOPLE_ASKS).format(
            count=len(item["participants"]), payer=payer
        )
    elif kind == "missing_amount":
        text = rng.choice(MISSING_AMOUNT).format(
            payer=payer, description=item["description"], people=people
        )
        ask = rng.choice(MISSING_AMOUNT_ASKS).format(payer=payer, description=item["description"])
    elif kind == "missing_currency":
        text = rng.choice(MISSING_CURRENCY).format(
            payer=payer, amount=item["amount"], description=item["description"], people=people
        )
        ask = rng.choice(MISSING_CURRENCY_ASKS).format(amount=item["amount"])
    else:
        text = rng.choice(AMBIGUOUS_SHARE).format(
            payer=payer, amount=item["amount"], currency=item["currency"],
            description=item["description"], people=people,
            ra=gender(payer, "raha", "rahi"),
        )
        ask = rng.choice(AMBIGUOUS_SHARE_ASKS).format(
            payer=payer, description=item["description"], ra=gender(payer, "raha", "rahi")
        )
    return underspecified(text, ask), twin(item)


def load_texts(name: str) -> list[str]:
    path = ROOT / "data" / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line)["task"] for line in handle if line.strip()]


def safety_rows() -> list[dict]:
    path = ROOT / "data" / "tasks_safety_eval.jsonl"
    if not path.exists():
        raise SystemExit("run data/build_safety.py first; the hold-out must be checked")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def collides(item: dict, text: str, held_out: list[dict]) -> bool:
    """True if a pair reuses a safety task's wording or its (payer, amount, description)."""
    numbers = {str(item["amount"]), f"{item['amount']:.2f}"}
    for task in held_out:
        if clean(text) == clean(task["task"]):
            return True
        other = task["task"]
        if (
            str(item["amount"]) in other
            and item["payer"] in other
            and item["description"] in other
        ):
            return True
        if numbers & set(re.findall(r"\d+(?:\.\d+)?", other)) and item["description"] in other:
            return True
    return False


def build(seed: int) -> list[dict]:
    rng = random.Random(seed)
    used: set = set()
    held_out = safety_rows()
    eval_texts = {
        clean(text)
        for name in ("tasks_hard_eval.jsonl", "tasks_eval.jsonl", "tasks_safety_eval.jsonl")
        for text in load_texts(name)
    }
    tasks: list[dict] = []
    for kind in ("missing_people", "missing_amount", "missing_currency", "ambiguous_share"):
        made = 0
        attempts = 0
        while made < PER_KIND:
            attempts += 1
            if attempts > PER_KIND * 200:
                raise SystemExit(f"could not build {PER_KIND} unique {kind} pairs")
            item = sample_expense(rng, used)
            under, full = pair(kind, item, rng)
            if clean(under["task"]) in eval_texts or clean(full["task"]) in eval_texts:
                continue
            if collides(item, under["task"], held_out) or collides(item, full["task"], held_out):
                continue
            eval_texts.add(clean(under["task"]))
            eval_texts.add(clean(full["task"]))
            under["id"] = f"a-u-{len(tasks) + 1:03d}"
            full["id"] = f"a-t-{len(tasks) + 1:03d}"
            under["family"] = kind
            full["family"] = f"twin_{kind}"
            tasks.extend([under, full])
            made += 1
    return tasks


def verify(tasks: list[dict]) -> None:
    failures = []
    for task in tasks:
        episode = run_oracle(task)
        if task.get("abstain"):
            scores = score_safety(episode, dict(task, family="underspecified"))
            if scores["asked"] != 1.0 or scores["invented_expense"] != 0.0:
                failures.append((task["id"], scores))
        else:
            if episode["ledger"]["expense_count"] == 0:
                failures.append((task["id"], "twin recorded no expense"))
    if failures:
        for task_id, detail in failures[:5]:
            print(f"FAIL {task_id}: {detail}")
        raise SystemExit(f"abstain split: {len(failures)} oracle failures")
    asks = sum(1 for task in tasks if task.get("abstain"))
    print(f"abstain: {asks}/{asks} ask without touching the ledger, {asks} twins record an expense")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--out", default=str(ROOT / "data" / "tasks_train_abstain.jsonl"))
    parser.add_argument("--check", action="store_true", help="verify only, do not rewrite")
    args = parser.parse_args()

    tasks = build(args.seed)
    verify(tasks)
    if args.check:
        return

    path = Path(args.out)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"wrote {len(tasks)} -> {path}")
    print(f"example ask: {next(t for t in tasks if t.get('abstain'))['answer_text']}")


if __name__ == "__main__":
    main()
