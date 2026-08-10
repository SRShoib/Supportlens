import re
from typing import Literal

UrgencyLabel = Literal["low", "medium", "high"]

_MASK_RE = re.compile(r"<(?:URL|USER|EMAIL|PHONE|EMOJI:[^>]*)>")

# Escalation, fraud/security, and strong-negative-sentiment language. Weighted
# heavier than medium signals since any one hit is a strong urgency indicator
# on its own.
_HIGH_PATTERNS = [
    r"\blawyer\b",
    r"\blawsuit\b",
    r"\bsue\b",
    r"\battorney\b",
    r"\blegal action\b",
    r"\bbbb\b",
    r"\bbetter business bureau\b",
    r"\bchargeback\b",
    r"\bfraud\b",
    r"\bscam\b",
    r"\bstolen\b",
    r"\bstealing\b",
    r"\bhacked\b",
    r"\bunauthorized\b",
    r"\b(someone|somebody)\s+(changed|accessed|hacked|logged)\b",
    r"\bunacceptable\b",
    r"\bdisgusting\b",
    r"\bhorrible\b",
    r"\bterrible\b",
    r"\bworst\b",
    r"\brip.?off\b",
    r"\bpathetic\b",
    r"\bdisgrace\b",
    r"\d+\s*(st|nd|rd|th)\s+(day|time)\b",
    r"\bstill no response\b",
    r"\bno response\b",
    r"\bcontact (my|your) (mp|lawyer|attorney)\b",
]

# Frustration, repeated-contact, and service-broken language. A weaker signal
# individually than the high-urgency list — several hits, or one hit plus
# punctuation/caps signal, are what push a message from low to medium.
_MEDIUM_PATTERNS = [
    r"\bfrustrat\w*\b",
    r"\bannoy\w*\b",
    r"\bdisappoint\w*\b",
    r"\bridiculous\b",
    r"\bplease help\b",
    r"\bplease (respond|fix|reply)\b",
    r"\bstill waiting\b",
    r"\bany update\b",
    r"\bagain\b",
    r"\bnot working\b",
    r"\bbroken\b",
    r"\boutage\b",
    r"\bcan.?t\b",
    r"\bcannot\b",
    r"\bdamn\b",
    r"\b(two|three|four|five|six|\d+)\s+(times|dms|emails|messages|days)\b",
]

_HIGH_RE = [re.compile(p) for p in _HIGH_PATTERNS]
_MEDIUM_RE = [re.compile(p) for p in _MEDIUM_PATTERNS]

HIGH_THRESHOLD = 2.0
MEDIUM_THRESHOLD = 0.7


def _caps_ratio(text: str) -> float:
    stripped = _MASK_RE.sub(" ", text)
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return 0.0
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters)


def _punctuation_intensity(text: str) -> float:
    bangs_and_questions = text.count("!") + text.count("?")
    return min(bangs_and_questions, 5) / 5


def compute_urgency_score(text: str) -> float:
    """Weighted combination of ALL-CAPS ratio, `!`/`?` intensity, and
    high/medium urgency keyword hits. Higher means more urgent."""
    lowered = text.lower()
    score = 0.0
    score += _caps_ratio(text) * 3.0
    score += _punctuation_intensity(text) * 1.0
    score += sum(1 for p in _HIGH_RE if p.search(lowered)) * 2.0
    score += sum(1 for p in _MEDIUM_RE if p.search(lowered)) * 1.0
    return score


def weak_label_urgency(text: str) -> UrgencyLabel:
    """Rule-based urgency weak label per SPEC §2: keywords, punctuation,
    ALL-CAPS ratio, refund/legal terms. Intended for customer-authored
    messages only — apply to `text_clean`, not `text_raw` (masking/cleaning
    already reflect the real signal, e.g. dropping HTML noise).

    This is deliberately a *weak* signal, not ground truth — SPEC frames
    urgency as a weak-supervision showcase, and ml/evaluation/kappa.py
    reports agreement against an LLM-labeled seed set rather than assuming
    this heuristic is correct."""
    score = compute_urgency_score(text)
    if score >= HIGH_THRESHOLD:
        return "high"
    if score >= MEDIUM_THRESHOLD:
        return "medium"
    return "low"
