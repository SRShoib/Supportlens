"""Lexical-overlap highlighter for search result snippets (SPEC M8: "results
with highlighted matches"). Pure word-level overlap between the query and a
candidate document -- no ML, deliberately: the cross-encoder (or dense
cosine score) already ranks results, this only has to show *why* a specific
snippet surfaced, which raw term overlap does well enough for. Stopwords are
excluded so a query like "how do I track my order" doesn't light up every
"my"/"do"/"I" in the snippet.

Same start/end-character-offset contract ml/inference/base.py's EntitySpan
uses (offsets into the exact string passed in, never a re-cleaned copy), so
apps/dashboard can render these with the same span-highlighting idea
entity-highlighted-text.tsx already has for M4's entity chips.
"""

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_WORD_RE = re.compile(r"\w+")


@dataclass(frozen=True)
class HighlightSpan:
    start: int
    end: int


def _query_terms(query: str) -> set[str]:
    return {m.group().lower() for m in _WORD_RE.finditer(query)} - ENGLISH_STOP_WORDS


def highlight_matches(query: str, document: str) -> list[HighlightSpan]:
    """Character spans in `document` whose word matches (case-insensitive) a
    non-stopword term in `query`. Empty if the query has no non-stopword
    terms (e.g. a query that's entirely stopwords) or nothing matches."""
    terms = _query_terms(query)
    if not terms:
        return []
    return [
        HighlightSpan(m.start(), m.end())
        for m in _WORD_RE.finditer(document)
        if m.group().lower() in terms
    ]
