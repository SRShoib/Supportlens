"""Sentence templates for the synthetic NER generator. A template's `text`
carries `{LABEL}` / `{LABEL#N}` slots; render() fills them and returns exact
char spans by tracking a running cursor -- never str.find(), which would
silently pick the wrong occurrence when a sampled value repeats within one
rendered sentence.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass

from ml.data.ner.schema import CharSpan

_SLOT_RE = re.compile(r"\{([A-Z_]+)(?:#\d+)?\}")


@dataclass(frozen=True)
class Template:
    id: str
    text: str


def template_slots(text: str) -> list[str]:
    """The slot keys in a template, in order, e.g. ["ORDER_ID", "DATE#2"] for
    a template with a repeated DATE slot. The label a slot samples/annotates
    as is the key with any "#N" suffix stripped."""
    return [m.group(0)[1:-1] for m in _SLOT_RE.finditer(text)]


def slot_label(slot_key: str) -> str:
    return slot_key.split("#", 1)[0]


def render(template_text: str, values: Mapping[str, str]) -> tuple[str, list[CharSpan]]:
    """values maps each slot key (as returned by template_slots) to its
    sampled surface form. Builds text and spans in one forward pass with a
    running cursor, so offsets are correct by construction even when the
    same value appears more than once in the rendered sentence."""
    out: list[str] = []
    spans: list[CharSpan] = []
    cursor = 0
    pos = 0
    for match in _SLOT_RE.finditer(template_text):
        literal = template_text[pos : match.start()]
        out.append(literal)
        cursor += len(literal)

        slot_key = match.group(0)[1:-1]
        label = slot_label(slot_key)
        value = values[slot_key]

        start = cursor
        end = cursor + len(value)
        spans.append(CharSpan(start=start, end=end, label=label, text=value))
        out.append(value)
        cursor = end
        pos = match.end()

    out.append(template_text[pos:])
    return "".join(out), spans


TEMPLATES: tuple[Template, ...] = (
    Template("order_shipped_date", "order {ORDER_ID} shipped {DATE} but never arrived"),
    Template("order_expected_date", "my order {ORDER_ID} was supposed to arrive {DATE}"),
    Template("order_status_check", "can you check the status of order {ORDER_ID}"),
    Template("order_charged_amount", "I was charged {AMOUNT} for order {ORDER_ID}"),
    Template("product_refund_request", "please refund {AMOUNT} for my {PRODUCT}"),
    Template("product_broken_date", "I bought a {PRODUCT} {DATE} and it's already broken"),
    Template("product_stopped_working", "my {PRODUCT} stopped working {DATE}"),
    Template("product_return_date", "can I return my {PRODUCT} bought {DATE}"),
    Template("account_charged_unauthorized", "account {ACCOUNT_REF} was charged {AMOUNT} without my permission"),
    Template("account_balance", "my account {ACCOUNT_REF} shows a balance of {AMOUNT}"),
    Template("case_open_since", "case {ACCOUNT_REF} has been open since {DATE}"),
    Template("called_about_product", "I called {DATE} about my {PRODUCT} and still no update"),
    Template("product_order_amount_date", "the {PRODUCT} I ordered {DATE} for {AMOUNT} never shipped"),
    Template(
        "cancel_order_refund_account",
        "please cancel order {ORDER_ID} and refund {AMOUNT} to account {ACCOUNT_REF}",
    ),
    Template("case_waiting_since", "still waiting on a response about case {ACCOUNT_REF} from {DATE}"),
    Template("charged_unknown_date", "charged {AMOUNT} on {DATE} for something I didn't order"),
    Template("subscription_renewed", "my {PRODUCT} subscription renewed {DATE} for {AMOUNT}"),
    Template("reference_for_order", "reference {ACCOUNT_REF} for order {ORDER_ID}, please advise"),
    Template("tracking_not_updated", "tracking on order {ORDER_ID} hasn't updated since {DATE}"),
    Template("refund_on_product", "I need a refund of {AMOUNT} on my {PRODUCT}"),
    Template(
        "billed_twice",
        "my {PRODUCT} order {ORDER_ID} was billed twice, {AMOUNT#1} then {AMOUNT#2}",
    ),
    Template("product_broke_warranty", "the {PRODUCT} broke {DATE}, still under warranty I think"),
    Template(
        "order_full_details",
        "order {ORDER_ID} placed {DATE}, charged {AMOUNT}, ref {ACCOUNT_REF}",
    ),
    Template("account_billed_date", "why was my account {ACCOUNT_REF} billed {AMOUNT} {DATE}"),
    Template("order_update_request", "requesting an update on order {ORDER_ID} from {DATE}"),
)  # fmt: skip
