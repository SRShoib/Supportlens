"""Regex/rules NER baseline (SPEC M4: "compare against a regex/rules
baseline"). Entity definitions: docs/ner-annotation-guidelines.md.

Stdlib `re` only, no `ml.data` import. `infra/api.Dockerfile` syncs
`--no-group ml`, so ftfy/emoji/spacy/pandas (all imported transitively by
ml.data.cleaning) are absent from the API image — apps/api and
ml/inference/* must be importable without them.

Never cleans its input: start/end are always character offsets into the
exact string passed to predict(), per the offset contract every M4 component
shares (see ml/inference/token_classification.py's module docstring for the
other half of that contract).
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ml.inference.base import EntityResult, EntitySpan

ENTITY_LABELS = ("ORDER_ID", "PRODUCT", "DATE", "AMOUNT", "ACCOUNT_REF")

# Stripped from a raw regex match so spans obey the guideline's "no leading/
# trailing whitespace or punctuation" rule.
_TRIM_CHARS = " \t\n\r.,!?;:)]}\"'"


@dataclass(frozen=True)
class Rule:
    """One named pattern. `group` selects which capture group is the actual
    span — lets a pattern require a trigger word ("order #12345") without
    including the trigger in the emitted span (span is "12345")."""

    name: str
    label: str
    pattern: re.Pattern[str]
    group: int | str = 0


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    surface = text[start:end]
    stripped = surface.strip(_TRIM_CHARS)
    if not stripped:
        return start, start
    left_pad = surface.index(stripped)
    return start + left_pad, start + left_pad + len(stripped)


def _trigger_pattern(trigger: str, value: str) -> re.Pattern[str]:
    """`<trigger word> [#/no/number/id] [:/#] <value>`, trigger matched
    case-insensitively via a scoped inline flag so the captured value (often
    meaningfully uppercase, e.g. order codes) is unaffected."""
    connector = r"\s*(?:number|num|no\.?|#|id)?\s*[:#]?\s*"
    return re.compile(rf"\b(?i:{trigger}){connector}(?P<v>{value})\b")


# --- ORDER_ID ---------------------------------------------------------------
# Wins on this task, decisively (docs/m4-rules-vs-model-report.md): order ids
# are a closed, high-precision *format* class with strong lexical triggers.

_ORDER_TRIGGER = r"(?:order|confirmation|booking|tracking|invoice|rma|return)s?"

_ORDER_ID_RULES = (
    Rule("order_trigger_numeric", "ORDER_ID", _trigger_pattern(_ORDER_TRIGGER, r"\d{4,12}"), "v"),
    Rule(
        "order_trigger_alnum",
        "ORDER_ID",
        _trigger_pattern(_ORDER_TRIGGER, r"[A-Z]{1,5}-?\d{3,}(?:-[A-Z0-9]+)*"),
        "v",
    ),
    Rule(
        "order_standalone_carrier_tracking",
        "ORDER_ID",
        re.compile(r"\b(?P<v>1Z[A-Z0-9]{16})\b"),
        "v",
    ),
)

# --- ACCOUNT_REF -------------------------------------------------------------
# @handles are already <USER>-masked before NER ever sees the text (see
# ml/data/masking.py), so ACCOUNT_REF here only ever means account/member/
# policy/case numbers and card last-4 — never social handles.

_ACCOUNT_TRIGGER = r"(?:account|membership|member|policy|case|reference|ref|subscriber|customer)s?"
# Requires at least one digit somewhere in the token, via lookahead — without
# it, "account number is closed" would greedily accept the bare word "number"
# as the value once the optional infix backtracks away from consuming it.
_ACCOUNT_VALUE = r"(?=[A-Za-z0-9-]*\d)[A-Za-z0-9][A-Za-z0-9-]{3,}"

_ACCOUNT_REF_RULES = (
    Rule(
        "account_trigger_alnum",
        "ACCOUNT_REF",
        _trigger_pattern(_ACCOUNT_TRIGGER, _ACCOUNT_VALUE),
        "v",
    ),
    Rule(
        "account_card_last_four",
        "ACCOUNT_REF",
        re.compile(r"\b(?i:ending\s+in|last\s+4\s+digits?(?:\s+(?:are|is))?:?)\s*(?P<v>\d{4})\b"),
        "v",
    ),
)

# --- AMOUNT -------------------------------------------------------------
# Wins or ties: currency grammar is compact and regular. The bare-number
# rule below deliberately excludes any leading "$" from its own capture —
# when a symbol is present, amount_symbol already produces the longer,
# preferred span and resolve_overlaps picks it.

_AMOUNT_RULES = (
    Rule(
        "amount_symbol",
        "AMOUNT",
        re.compile(r"(?P<v>[$£€]\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)"),
        "v",
    ),
    Rule(
        "amount_code",
        "AMOUNT",
        re.compile(r"\b(?P<v>(?:USD|EUR|GBP|CAD|AUD)\s?\d+(?:\.\d{2})?)\b"),
        "v",
    ),
    Rule(
        "amount_spelled",
        "AMOUNT",
        re.compile(
            r"\b(?P<v>\d+(?:\.\d+)?\s+(?:dollars?|bucks|quid|pounds?(?:\s+sterling)?|euros?))\b",
            re.IGNORECASE,
        ),
        "v",
    ),
    Rule(
        "amount_governed_bare_number",
        "AMOUNT",
        re.compile(
            r"\b(?i:charged|refund(?:ed)?|billed|paid|cost(?:s)?|priced?|took)\s+"
            r"(?:me\s+)?(?:of\s+)?\$?\s?(?P<v>\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\b"
        ),
        "v",
    ),
)

# --- DATE -----------------------------------------------------------------
# Loses: relative/colloquial dates are open-class (real-corpus measurement:
# relative forms are ~3.8x more common than absolute ones in real support
# tweets) and this closed list cannot enumerate them.

_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_NUMBER_WORD = r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)"

_RELATIVE_DATE_TERMS = (
    "yesterday", "today", "tonight", "tomorrow",
    "this morning", "this afternoon", "this evening",
    "this week", "this month", "this year", "this weekend",
    "last night", "last week", "last month", "last year",
    "last monday", "last tuesday", "last wednesday", "last thursday",
    "last friday", "last saturday", "last sunday",
    "over the weekend", "next week",
)  # fmt: skip

_DATE_RULES = (
    Rule(
        "date_numeric",
        "DATE",
        re.compile(r"\b(?P<v>\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"),
        "v",
    ),
    Rule(
        "date_month_day",
        "DATE",
        re.compile(rf"\b(?P<v>(?i:{_MONTH})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?)\b"),
        "v",
    ),
    Rule(
        "date_day_month",
        "DATE",
        re.compile(
            rf"\b(?P<v>\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?i:{_MONTH})\.?(?:,?\s+\d{{4}})?)\b"
        ),
        "v",
    ),
    Rule("date_month_year", "DATE", re.compile(rf"\b(?P<v>(?i:{_MONTH})\.?\s+\d{{4}})\b"), "v"),
    Rule(
        "date_relative_duration",
        "DATE",
        re.compile(
            rf"\b(?P<v>{_NUMBER_WORD}\s+(?:day|week|month|year)s?\s+(?:ago|back))\b",
            re.IGNORECASE,
        ),
        "v",
    ),
    Rule(
        "date_relative_term",
        "DATE",
        re.compile(
            r"\b(?P<v>"
            + "|".join(re.escape(t) for t in sorted(_RELATIVE_DATE_TERMS, key=len, reverse=True))
            + r")\b",
            re.IGNORECASE,
        ),
        "v",
    ),
    Rule(
        "date_holiday",
        "DATE",
        re.compile(
            r"\b(?P<v>christmas(?:\s+eve)?|black\s+friday|cyber\s+monday|thanksgiving|"
            r"new\s+year'?s(?:\s+day)?)\b",
            re.IGNORECASE,
        ),
        "v",
    ),
)

# --- PRODUCT -----------------------------------------------------------------
# Loses by a wide margin: open class, and a gazetteer covers exactly what's
# in it and nothing else. Sorted longest-first before compiling — regex
# alternation takes the first alternative that matches at a position, not
# the longest, so "iPhone 12" ahead of "iPhone 12 Pro Max" in this list would
# silently truncate every longer match.

_PRODUCT_GAZETTEER = (
    "iPhone 15 Pro Max", "iPhone 15 Pro", "iPhone 15", "iPhone 14 Pro Max", "iPhone 14 Pro",
    "iPhone 14", "iPhone 13 Pro Max", "iPhone 13 Pro", "iPhone 13", "iPhone 12 Pro Max",
    "iPhone 12 Pro", "iPhone 12", "iPhone SE", "iPad Pro", "iPad Air", "iPad Mini",
    "MacBook Pro", "MacBook Air", "Apple Watch Series 9", "Apple Watch Series 8",
    "Apple Watch SE", "AirPods Pro", "AirPods Max", "AirPods",
    "Samsung Galaxy S23", "Samsung Galaxy S22", "Samsung Galaxy S21",
    "Samsung Galaxy Note 20", "Samsung Galaxy Watch", "Samsung Galaxy Buds", "Galaxy Tab",
    "Kindle Paperwhite", "Kindle Fire", "Kindle",
    "PlayStation 5", "PlayStation 4", "PS5", "PS4",
    "Xbox Series X", "Xbox Series S", "Xbox Game Pass", "Xbox One",
    "Nintendo Switch",
    "Spotify Premium", "Spotify Family", "Prime Video", "Amazon Prime",
    "Netflix Premium", "Disney Plus", "Disney+",
    "Google Pixel 8", "Google Pixel 7", "Google Pixel",
    "Fitbit Charge", "Fitbit Versa", "Echo Dot", "Echo Show", "Amazon Echo",
    "Surface Pro", "Surface Laptop",
    "Delta Comfort+", "Comfort+", "Basic Economy",
    "Unlimited Plus", "Unlimited Plan", "Unlimited Data",
)  # fmt: skip

_PRODUCT_RULES = (
    Rule(
        "product_gazetteer",
        "PRODUCT",
        re.compile(
            r"\b(?P<v>"
            + "|".join(re.escape(p) for p in sorted(_PRODUCT_GAZETTEER, key=len, reverse=True))
            + r")\b",
            re.IGNORECASE,
        ),
        "v",
    ),
)

RULES: tuple[Rule, ...] = (
    *_ORDER_ID_RULES,
    *_ACCOUNT_REF_RULES,
    *_AMOUNT_RULES,
    *_DATE_RULES,
    *_PRODUCT_RULES,
)


def resolve_overlaps(spans: Sequence[EntitySpan]) -> list[EntitySpan]:
    """Entities are flat and non-overlapping (docs/ner-annotation-guidelines.md).
    Longest span wins; ties broken by which candidate came first in `spans`
    (extract_spans appends in RULES declaration order, so an earlier rule
    wins ties over a later one)."""
    indexed = list(enumerate(spans))
    indexed.sort(key=lambda pair: (-(pair[1].end - pair[1].start), pair[0]))

    selected: list[EntitySpan] = []
    occupied: list[tuple[int, int]] = []
    for _, span in indexed:
        if any(span.start < e and s < span.end for s, e in occupied):
            continue
        selected.append(span)
        occupied.append((span.start, span.end))

    return sorted(selected, key=lambda s: s.start)


def extract_spans(text: str, rules: Sequence[Rule] = RULES) -> list[EntitySpan]:
    candidates: list[EntitySpan] = []
    for rule in rules:
        for match in rule.pattern.finditer(text):
            start, end = trim_span(text, *match.span(rule.group))
            if start >= end:
                continue
            candidates.append(
                EntitySpan(start=start, end=end, label=rule.label, text=text[start:end], score=1.0)
            )
    return resolve_overlaps(candidates)


class RulesEntityPredictor:
    """The regex/rules NER baseline (SPEC M4). Deterministic, so every span
    carries score=1.0 — there is no calibrated confidence to report, unlike
    the transformer variant's per-token softmax."""

    def predict(self, texts: list[str]) -> list[EntityResult]:
        if not texts:
            return []
        return [EntityResult(entities=extract_spans(text)) for text in texts]
