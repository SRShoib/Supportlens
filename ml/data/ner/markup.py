"""Bracket annotation markup: `[surface|LABEL]`. The one human-facing format
used both by scripts/ner_gold_{export,import}.py (hand annotation in a
Markdown file) and ml/data/ner/paraphrase.py (round-tripping an LLM
rewrite). One parser/renderer for both keeps them from silently drifting
apart, and offsets are always computed from parse() -- never trusted from
whatever produced the markup -- which is what makes a malformed hand-edit or
a broken LLM response impossible to land silently.

Literal '\\', '[', ']', '|' in either the surrounding text or an entity's
surface form are backslash-escaped on render() and unescaped on parse(), so:
  parse(render(text, spans)) == (text, spans)              for any valid (text, spans)
  render(*parse(markup)) == markup   for any well-formed, canonically-escaped markup
"""

import re
from collections.abc import Sequence

from ml.data.ner.schema import CharSpan
from ml.inference.rules_ner import ENTITY_LABELS

_SPECIAL_CHARS = re.compile(r"[\\\[\]|]")


class MarkupError(ValueError):
    """Raised by parse() on malformed markup: an unterminated or unmatched
    bracket, a block with no '|' separator, or an unrecognized label."""


def _escape(text: str) -> str:
    return _SPECIAL_CHARS.sub(lambda m: "\\" + m.group(0), text)


def render(text: str, spans: Sequence[CharSpan]) -> str:
    """text + non-overlapping spans -> bracket markup. Spans need not be
    pre-sorted. Raises MarkupError if any two spans overlap."""
    ordered = sorted(spans, key=lambda s: s.start)
    out: list[str] = []
    cursor = 0
    for span in ordered:
        if span.start < cursor:
            raise MarkupError(f"overlapping span at position {span.start}")
        out.append(_escape(text[cursor : span.start]))
        out.append("[")
        out.append(_escape(span.text))
        out.append("|")
        out.append(span.label)
        out.append("]")
        cursor = span.end
    out.append(_escape(text[cursor:]))
    return "".join(out)


def parse(markup: str) -> tuple[str, list[CharSpan]]:
    """bracket markup -> (plain text, spans with offsets into that text).
    Never trusts the markup's own claims about the plain text -- it is
    reconstructed by stripping and unescaping, which is what lets a caller
    assert the result matches an independently-held original."""
    n = len(markup)
    i = 0
    plain_parts: list[str] = []
    spans: list[CharSpan] = []
    cursor = 0

    def read_until(stop_chars: str) -> str:
        nonlocal i
        buf: list[str] = []
        while i < n and markup[i] not in stop_chars:
            if markup[i] == "\\" and i + 1 < n:
                buf.append(markup[i + 1])
                i += 2
            else:
                buf.append(markup[i])
                i += 1
        return "".join(buf)

    while i < n:
        if markup[i] == "]":
            raise MarkupError(f"unmatched ']' at position {i}")
        if markup[i] != "[":
            segment = read_until("[]")
            plain_parts.append(segment)
            cursor += len(segment)
            continue

        i += 1  # consume '['
        surface = read_until("|]")
        if i >= n or markup[i] != "|":
            raise MarkupError(f"entity block missing '|' near position {i}")
        i += 1  # consume '|'
        label = read_until("]")
        if i >= n or markup[i] != "]":
            raise MarkupError(f"unterminated entity block near position {i}")
        i += 1  # consume ']'

        if not surface:
            raise MarkupError(f"empty entity surface near position {i}")
        if label not in ENTITY_LABELS:
            raise MarkupError(f"unknown label {label!r} near position {i}")

        start = cursor
        end = cursor + len(surface)
        spans.append(CharSpan(start=start, end=end, label=label, text=surface))
        plain_parts.append(surface)
        cursor = end

    return "".join(plain_parts), spans
