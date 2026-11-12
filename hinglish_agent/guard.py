"""A deterministic guard for tool results: the layer under the model, not a second model.

The adapter ignores the injected notes it was trained on, but 6 of 136 still reach the ledger,
and a model that stays silent is not the same as a model that tells the user. This module is the
engineering answer to both: scan every tool result before the model sees it, keep the reasons,
and optionally neutralise the text or append a fixed warning to the final answer.

The rules are deliberately structural, not lexical copies of the payloads: an instruction aimed
at the agent (an imperative verb with a side-effect object), an impersonated authority header,
a new payment handle, a "do not tell the user" clause. That is what makes the Hinglish,
Devanagari, obfuscated and split payloads a real test rather than a lookup of "ACCOUNTS-NOTE".
Detection is measured in `scripts/guard_eval.py`; nothing here is validated by an LLM judge.
"""

from __future__ import annotations

import json
import re
import unicodedata

WARNING = (
    "Note: a tool result contained instructions addressed to me rather than data. "
    "I ignored them and did not apply any of them to the ledger."
)
NEUTRALISED = "[tool result contained instructions addressed to the agent; removed]"

_VERB = (
    r"(?:add|insert|record|create|register|append|transfer|send|pay|settle|refund|charge|"
    r"debit|credit|update|change|replace|set|mark|remove|delete|erase|wipe|approve|release)"
)
_OBJECT = (
    r"(?:expense|expenses|payment|payments|amount|amounts|ledger|balance|entry|entries|"
    r"transfer|transfers|upi|payee|recipient|account|participants?|payer|money|funds|"
    r"rupees|inr|usd|eur)"
)
HINGLISH_VERB = (
    r"(?:jama|jodna|jodo|jod do|jod de|add|bhej|bhejo|de do|dedo|chalao|chala do|"
    r"kar do|kar dena|kardo|karden|badal do|badlo|badal dena|likh do|likho|likh dena|"
    r"transfer kar|settle kar|skip kar|pura kar|puri kar)"
)
DEVANAGARI = r"(?:जोड़|जोड|भेज|कर दो|करदो|बदल|ट्रांसफर|लिख|जमा|हटा)"

_LEDGER_INSTRUCTION = re.compile(
    rf"\b{_VERB}\b[^.!?\n]{{0,60}}\b{_OBJECT}\b", re.IGNORECASE
)
_LEDGER_INSTRUCTION_REVERSE = re.compile(
    rf"\b{_OBJECT}\b[^.!?\n]{{0,40}}\b{_VERB}\b", re.IGNORECASE
)
_HINGLISH_INSTRUCTION = re.compile(
    rf"\b{HINGLISH_VERB}\b[^.!?\n]{{0,50}}(?:\b{_OBJECT}\b|₹|\b\d{{3,}}\b)", re.IGNORECASE
)
_DEVANAGARI_INSTRUCTION = re.compile(rf"{DEVANAGARI}[^.!?\n]{{0,40}}(?:\d|₹|रुपय|खर्च|भुगतान)")

AUTHORITY = re.compile(
    r"^(?:[A-Z][A-Z0-9 &/_-]{2,28}:|\[(?:system|admin|accounts|billing|payments|compliance)\]|"
    r"(?:system|admin|compliance|accounts payable)\s*(?:note|notice|message|update)\b)",
    re.IGNORECASE | re.MULTILINE,
)
IGNORE_RULES = re.compile(
    r"(?:ignore|disregard|forget)\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier|"
    r"system|your)\b|do\s+not\s+(?:tell|inform|mention|ask)\s+(?:the\s+)?user",
    re.IGNORECASE,
)
HINDI_IGNORE = re.compile(r"(?:user ko mat batao|user ko na batao|mat batao|bina puche|bina poochhe)")
PAY_HANDLE = re.compile(r"\b[\w.\-]{2,}@(?:upi|ybl|okaxis|okhdfcbank|okicici|paytm|apl|axl)\b", re.IGNORECASE)
BIG_AMOUNT = re.compile(r"\b\d{6,}\b")
THOUSANDS_AMOUNT = re.compile(r"\b\d{1,3}(?:,\d{2,3}){1,3}\b")

# A tool result reports state; it does not ask for state to change. A mutation verb next to a
# six-figure amount is the signal that survives paraphrase, because it does not depend on the
# wording of any published payload.
MUTATION_VERB = re.compile(
    r"\b(?:add|insert|record|register|log|post|enter|book|apply|put|place|reimburse|refund|"
    r"adjust|correct|amend|fix|update|change|modify|set|transfer|send|pay|settle|split|"
    r"allocate|charge|debit|credit|create|delete|remove|clear|reverse|reconcile|convert|"
    r"jodna|jodo|jod|daal|daalo|likh|chala|bhej|bhejo|karo|kar\s+do)\b|"
    r"(?:जोड़|डाल|लिख|भेज|करें|करो|कर\s+दो|देना|सेटल|ट्रांसफर)",
    re.IGNORECASE,
)
MODAL = re.compile(
    r"\b(?:you\s+(?:must|should|need|have\s+to)|please|kindly|make\s+sure|ensure|must\s+be|"
    r"should\s+be|needs?\s+to\s+be|immediately|before\s+you|do\s+not|don'?t|without\s+asking)\b|"
    r"(?:kar\s+dena|kar\s+do|karo|dena|daal\s+do|जोड़ें|करें|कर\s+दो|देना)",
    re.IGNORECASE,
)
# "reimburse 999999 to Neha": the verb and the six-figure amount in the same breath, which is
# how a directive survives being nested in JSON or written without a header.
_VERB_AMOUNT = re.compile(
    r"\b(?:add|insert|record|register|log|post|enter|reimburse|refund|adjust|amend|transfer|"
    r"send|pay|settle|charge|debit|credit|reverse|reconcile)\b[^.\n]{0,24}?\d[\d,]{5,}",
    re.IGNORECASE,
)


def _large_amount(text: str) -> bool:
    for chunk in re.findall(r"\d[\d,]*", text):
        if len(re.sub(r"[^0-9]", "", chunk)) >= 6:
            return True
    return False


def _is_imperative_start(text: str) -> bool:
    """A clause that opens with a bare verb ("put an extra 999999 ... for Neha")."""
    for clause in re.split(r"[.;:!?\n]", text):
        words = clause.strip().split()
        if not words:
            continue
        head = words[0].lower().strip(",.")
        if head in {"the", "a", "an", "this", "that", "it", "there", "receipt", "note", "total"}:
            continue
        if MUTATION_VERB.fullmatch(head):
            return True
    return False

# A tool result does not tell the agent which tool to call; only an injected instruction does.
TOOL_NAME = re.compile(r"\b(?:add_expense|get_balance|settle_up|convert_currency|finish)\b", re.IGNORECASE)
TOOL_CONTROL = re.compile(
    r"\b(?:do\s+not|don'?t|must\s+not|never|skip|instead|only|call|invoke|use|mat|"
    r"chalao|chala\s+do|chhodo|chhod\s+do|kar\s+do|karo|likho|likh\s+do|de\s+do|dedo|"
    r"batao|bata\s+do|zaroori)\b|"
    r"(?:मत|छोड़ो|छोड़\s+दो|करो|कर\s+दो|लिखो|बताओ|ट्रांसफर|सेटल)",
    re.IGNORECASE,
)
TOOL_REFERENCE = re.compile(
    r"\btool\s+call\b|\btool\s+result\b|\bbalance\s+query\b|\bटूल\s+कॉल\b|\bbalance\s+service\b",
    re.IGNORECASE,
)
LOCAL_DIRECTIVE = re.compile(
    r"chalao|chala\s+do|chhodo|chhod\s+do|kar\s+do|likho|likh\s+do|de\s+do|dedo|batao|"
    r"bata\s+do|zaroori|जोड़ो|करो|कर\s+दो|लिखो|बताओ|छोड़ो|सेटल|ट्रांसफर",
    re.IGNORECASE,
)


def _text_of(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


# Fields a tool echoes back from user text. A six-figure amount in one of them is not a label the
# user would type; it is the adaptive version of an injected expense, phrased as a fact with no
# verb to catch. Measured: 0 of 341 clean descriptions in the published traffic carry one.
USER_TEXT_KEYS = ("description",)


def scan_text(text: str) -> dict:
    """Reasons a piece of tool output reads as an instruction to the agent."""
    # Devanagari matras arrive in either normal form, and a literal built from one form will
    # silently miss the other, so both the text and the rules are compared in NFC.
    text = unicodedata.normalize("NFC", text or "")
    reasons: list[str] = []

    if IGNORE_RULES.search(text) or HINDI_IGNORE.search(text):
        reasons.append("ignore_rules")
    if _LEDGER_INSTRUCTION.search(text) or _LEDGER_INSTRUCTION_REVERSE.search(text):
        reasons.append("ledger_instruction")
    if _HINGLISH_INSTRUCTION.search(text):
        reasons.append("ledger_instruction_hinglish")
    if _DEVANAGARI_INSTRUCTION.search(text):
        reasons.append("ledger_instruction_devanagari")
    if PAY_HANDLE.search(text):
        reasons.append("payment_handle")
    if TOOL_NAME.search(text) and TOOL_CONTROL.search(text):
        reasons.append("tool_control")
    elif TOOL_REFERENCE.search(text) and TOOL_CONTROL.search(text):
        reasons.append("tool_control")
    if LOCAL_DIRECTIVE.search(text) and (BIG_AMOUNT.search(text) or TOOL_NAME.search(text)):
        reasons.append("localised_directive")
    if _large_amount(text) and MUTATION_VERB.search(text) and MODAL.search(text):
        reasons.append("state_change_directive")
    elif _large_amount(text) and MUTATION_VERB.search(text) and (
        _is_imperative_start(text) or _VERB_AMOUNT.search(text)
    ):
        reasons.append("state_change_directive")
    if AUTHORITY.search(text) and (reasons or _LEDGER_INSTRUCTION.search(text)):
        reasons.append("impersonated_authority")
    if BIG_AMOUNT.search(text) or THOUSANDS_AMOUNT.search(text):
        reasons.append("unusual_amount")

    strong = [r for r in reasons if r != "unusual_amount"]
    return {"flagged": bool(strong), "reasons": reasons, "strong": strong}


def scan_result(result) -> dict:
    """Scan one tool result, including nested `note`/`balance` payloads.

    `answer` is the agent's own text echoed back by `finish`, so it is dropped: scanning it
    made the guard flag a correct settle-up sentence as an instruction (one episode in 324).
    """
    if isinstance(result, dict):
        result = {k: v for k, v in result.items() if k not in ("answer", "guard")}
    flag = scan_text(_text_of(result))
    if isinstance(result, dict):
        for key in USER_TEXT_KEYS:
            if _large_amount(str(result.get(key, ""))):
                flag = {**flag, "reasons": [*flag["reasons"], "amount_in_user_text"],
                        "strong": [*flag["strong"], "amount_in_user_text"], "flagged": True}
                break
    return flag


# Every key `hinglish_agent.tools` can return. A key outside this set did not come from the tool.
ALLOWED_KEYS: dict[str, frozenset[str]] = {
    "add_expense": frozenset({"ok", "error", "ledger", "description", "total_inr", "share_inr"}),
    "convert_currency": frozenset({"ok", "error", "value", "unit"}),
    "get_balance": frozenset({"ok", "error", "person", "balance_inr"}),
    "settle_up": frozenset({"ok", "error", "transfers"}),
    "finish": frozenset({"ok", "error", "done", "answer"}),
}
UNKNOWN_TOOL_KEYS = frozenset({"ok", "error"})


def allowlist_result(name: str | None, result):
    """Drop fields the responding tool never emits; return (result, dropped).

    This is the cheapest defence on this surface and it is structural rather than lexical: the
    payload arrives in a `note` field that no tool in `hinglish_agent/tools.py` ever returns. It
    does not generalise to a payload written into a field the tool *does* return -- a shared
    expense's `description` is user text and travels back verbatim -- which is what the text
    rules are for.
    """
    if not isinstance(result, dict):
        return result, []
    allowed = ALLOWED_KEYS.get(name or "", UNKNOWN_TOOL_KEYS) | {"guard"}
    dropped = sorted(key for key in result if key not in allowed)
    if not dropped:
        return result, []
    return {key: value for key, value in result.items() if key in allowed}, dropped


def _with_flag(result, flag: dict):
    if not isinstance(result, dict):
        return result
    return {**result, "guard": {"flagged": True, "reasons": flag["reasons"]}}


def sanitize(result, mode: str = "flag", tool_name: str | None = None):
    """Return (result, flag). `strip` drops instruction-like notes, `neutralise` replaces them.

    `allowlist` is structural and needs the tool name: it drops every field that tool never
    returns, whether or not the text looks like an instruction.

    Only the `note` field is ever rewritten by the text modes: balances, totals and error strings
    are data the agent needs, and mangling them would break the task rather than the attack.
    """
    flag = scan_result(result)
    if mode == "allowlist":
        cleaned, dropped = allowlist_result(tool_name, result)
        return cleaned, {**flag, "dropped": dropped}
    if not flag["flagged"] or mode == "flag":
        return result, flag
    if not isinstance(result, dict) or "note" not in result:
        return result, flag
    if mode == "strip":
        stripped = {k: v for k, v in result.items() if k != "note"}
        return stripped, flag
    if mode == "neutralise":
        return {**result, "note": NEUTRALISED}, flag
    return result, flag


def warn_answer(answer: str, flagged: bool) -> str:
    """Append the fixed warning once, only when a tool result was flagged."""
    if not flagged:
        return answer
    if WARNING in (answer or ""):
        return answer
    body = (answer or "").rstrip()
    return f"{body}\n\n{WARNING}" if body else WARNING
