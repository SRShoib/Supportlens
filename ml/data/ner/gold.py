"""Gold-set tooling for M4 (SPEC: "hand-verify a 200-example gold test
set"): candidate selection for scripts/ner_gold_export.py, the markdown
export/import format, and validate_gold_set(), run by
scripts/ner_gold_import.py right after a human has annotated the exported
candidates and safe to re-run any time against the committed
data/gold/ner_gold_v1.jsonl to catch drift.
"""

import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ml.data.ner.markup import MarkupError, parse, render
from ml.data.ner.schema import CharSpan, NerExample, validate_example
from ml.inference.base import EntitySpan
from ml.inference.rules_ner import ENTITY_LABELS, extract_spans, resolve_overlaps

EXPECTED_SIZE = 200
MIN_SPANS_PER_TYPE = 15
BLIND_COUNT = 40
TARGET_SPANS_PER_TYPE = 30
NEGATIVE_SHARE = 0.15
GUIDELINE_VERSION = "v1"

# spaCy's default en_core_web_sm NER labels mapped onto the M4 schema.
# ORDER_ID and ACCOUNT_REF have no spaCy equivalent -- those candidates come
# from the rules baseline only.
SPACY_LABEL_MAP = {"DATE": "DATE", "MONEY": "AMOUNT", "PRODUCT": "PRODUCT"}


class GoldSetError(ValueError):
    """Raised by validate_gold_set for a gold-set-level invariant (size,
    duplicate ids, leakage). A single example's own invariants (offsets,
    overlaps, the clean_text fixed-point check, ...) raise
    ml.data.ner.schema.NerValidationError instead, with that example's id."""


@dataclass(frozen=True)
class GoldSetReport:
    n_examples: int
    span_counts_by_label: dict[str, int]
    warnings: list[str]


def validate_gold_set(
    examples: Sequence[NerExample],
    *,
    expected_size: int = EXPECTED_SIZE,
    min_spans_per_type: int = MIN_SPANS_PER_TYPE,
    synthetic_ids: Sequence[str] = (),
) -> GoldSetReport:
    """Raises on: wrong count, duplicate ids, any id also present in
    synthetic_ids (the leakage gate -- pass ml.data.ner.generate's output
    ids here), or any single example failing schema.validate_example.
    Returns a report with per-type span counts and soft warnings for any
    label under min_spans_per_type -- reported, not fatal, since 200
    hand-annotated examples over 5 open-vocabulary types can legitimately
    land a rare type just under a chosen floor."""
    if len(examples) != expected_size:
        raise GoldSetError(f"expected exactly {expected_size} examples, got {len(examples)}")

    ids = [example.id for example in examples]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise GoldSetError(f"duplicate example ids: {duplicates}")

    overlap = sorted(set(synthetic_ids) & set(ids))
    if overlap:
        raise GoldSetError(f"gold examples overlap the synthetic training set: {overlap}")

    for example in examples:
        validate_example(example)

    span_counts: dict[str, int] = dict.fromkeys(ENTITY_LABELS, 0)
    for example in examples:
        for span in example.entities:
            span_counts[span.label] += 1

    warnings = [
        f"{label}: only {count} spans (below the {min_spans_per_type}-span floor for a "
        "readable per-entity F1)"
        for label, count in span_counts.items()
        if count < min_spans_per_type
    ]

    return GoldSetReport(
        n_examples=len(examples), span_counts_by_label=span_counts, warnings=warnings
    )


def propose_entities(text: str, nlp: Any = None) -> list[CharSpan]:
    """Union of the rules baseline and (if given) a loaded spaCy pipeline,
    mapped onto the M4 schema and overlap-resolved -- the pre-annotation
    proposer for gold-set candidates. `nlp` is duck-typed (any object whose
    `nlp(text).ents` yields items with start_char/end_char/label_/text) so
    tests can pass a lightweight fake instead of loading a real spaCy model.

    Pre-annotating with the rules baseline alone would bias the gold set
    toward exactly the system M4 evaluates; the spaCy pass is a genuinely
    independent second opinion (it also catches DATE/AMOUNT/PRODUCT phrasing
    the closed-list rules baseline doesn't enumerate, at the cost of being a
    noisier labeller -- both are suggestions for a human to accept, correct,
    or reject, never final)."""
    proposals: list[EntitySpan] = list(extract_spans(text))
    if nlp is not None:
        for ent in nlp(text).ents:
            label = SPACY_LABEL_MAP.get(ent.label_)
            if label is None:
                continue
            proposals.append(EntitySpan(ent.start_char, ent.end_char, label, ent.text, score=0.5))
    resolved = resolve_overlaps(proposals)
    return [CharSpan(s.start, s.end, s.label, s.text) for s in resolved]


@dataclass(frozen=True)
class GoldCandidate:
    id: str
    text: str
    proposed_entities: list[CharSpan]


@dataclass(frozen=True)
class GoldSelection:
    candidates: list[GoldCandidate]
    blind_ids: frozenset[str]


def select_gold_candidates(
    candidates: Sequence[GoldCandidate],
    rng: random.Random,
    *,
    total: int = EXPECTED_SIZE,
    blind_count: int = BLIND_COUNT,
    target_spans_per_type: int = TARGET_SPANS_PER_TYPE,
    negative_share: float = NEGATIVE_SHARE,
) -> GoldSelection:
    """Stratifies the candidate pool so every entity type clears
    target_spans_per_type proposed spans (a generous margin over
    validate_gold_set's MIN_SPANS_PER_TYPE floor, since annotation will
    reject some proposals), reserves a negative_share of entity-free
    candidates so the gold set measures precision honestly, and marks a
    fixed blind_count subset for annotation with no suggestions shown at
    all -- the control for pre-annotation bias."""
    shuffled = list(candidates)
    rng.shuffle(shuffled)

    with_entities = [c for c in shuffled if c.proposed_entities]
    without_entities = [c for c in shuffled if not c.proposed_entities]

    n_negative = min(round(total * negative_share), len(without_entities))
    selected: list[GoldCandidate] = list(without_entities[:n_negative])
    selected_ids = {c.id for c in selected}

    span_counts: dict[str, int] = dict.fromkeys(ENTITY_LABELS, 0)
    remaining = [c for c in with_entities if c.id not in selected_ids]
    exhausted_labels: set[str] = set()

    while len(selected) < total and remaining:
        under_target = [
            label
            for label in ENTITY_LABELS
            if span_counts[label] < target_spans_per_type and label not in exhausted_labels
        ]
        if not under_target:
            break
        label = min(under_target, key=lambda label: span_counts[label])
        match = next(
            (c for c in remaining if any(s.label == label for s in c.proposed_entities)), None
        )
        if match is None:
            exhausted_labels.add(label)
            continue
        selected.append(match)
        selected_ids.add(match.id)
        remaining.remove(match)
        for span in match.proposed_entities:
            span_counts[span.label] = span_counts.get(span.label, 0) + 1

    leftover = remaining + [c for c in without_entities if c.id not in selected_ids]
    for candidate in leftover:
        if len(selected) >= total:
            break
        selected.append(candidate)
        selected_ids.add(candidate.id)

    rng.shuffle(selected)
    selected = selected[:total]

    blind_pool = [c.id for c in selected]
    rng.shuffle(blind_pool)
    blind_ids = frozenset(blind_pool[:blind_count])

    return GoldSelection(candidates=selected, blind_ids=blind_ids)


def render_gold_markdown(
    selection: GoldSelection, guideline_version: str = GUIDELINE_VERSION
) -> str:
    """The human-facing annotation artifact: one heading + one body line per
    candidate. Pre-annotated candidates show bracket markup to correct;
    blind candidates show plain text with instructions to add brackets from
    scratch. scripts/ner_gold_import.py's parse_gold_markdown() is the exact
    inverse of this format."""
    lines = [f"<!-- guideline: docs/ner-annotation-guidelines.md {guideline_version} -->", ""]
    for candidate in selection.candidates:
        tag = "  [blind]" if candidate.id in selection.blind_ids else ""
        lines.append(f"## msg:{candidate.id}{tag}")
        if candidate.id in selection.blind_ids:
            # No suggestions shown, but still routed through render() (with
            # no spans, so no brackets appear) rather than appended raw --
            # otherwise an embedded newline in the message text (real
            # Twitter messages can have them) bypasses the escaping every
            # other path relies on and silently splits this entry across
            # multiple markdown lines.
            lines.append(render(candidate.text, []))
        else:
            lines.append(render(candidate.text, candidate.proposed_entities))
        lines.append("")
    return "\n".join(lines)


_HEADING_RE = re.compile(r"^## msg:(?P<id>\S+?)(?:\s+\[blind\])?\s*$")


def parse_gold_markdown(markdown: str) -> list[tuple[str, str, bool]]:
    """(candidate_id, body_line, is_blind) tuples in file order -- the exact
    inverse of render_gold_markdown()."""
    lines = markdown.splitlines()
    entries: list[tuple[str, str, bool]] = []
    i = 0
    while i < len(lines):
        match = _HEADING_RE.match(lines[i])
        if match is None:
            i += 1
            continue
        is_blind = "[blind]" in lines[i]
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        body = lines[j] if j < len(lines) else ""
        entries.append((match.group("id"), body, is_blind))
        i = j + 1
    return entries


class GoldImportError(ValueError):
    """Raised by import_gold_markdown for anything that would land a
    corrupted or unverifiable example: a heading with no matching export
    record, malformed markup, or edited prose that no longer matches the
    originally exported text."""


def import_gold_markdown(
    markdown: str, originals: Mapping[str, str], split: str = "gold"
) -> list[NerExample]:
    """Parses annotated markdown back into NerExamples. Never trusts the
    edited file's own claims about the underlying text: for every entry,
    the markup is stripped and the result is required to be byte-identical
    to `originals[candidate_id]` (the text originally exported, before any
    annotation) -- that equality is what makes a malformed or overreaching
    hand-edit impossible to land silently. Only bracket annotations may be
    added or corrected; the prose itself must not change."""
    seen_ids: set[str] = set()
    examples: list[NerExample] = []
    for candidate_id, body, _is_blind in parse_gold_markdown(markdown):
        if candidate_id in seen_ids:
            raise GoldImportError(f"duplicate heading for {candidate_id!r} in markdown")
        seen_ids.add(candidate_id)

        original = originals.get(candidate_id)
        if original is None:
            raise GoldImportError(f"{candidate_id!r} not found in the export manifest")

        try:
            text, spans = parse(body)
        except MarkupError as exc:
            raise GoldImportError(f"{candidate_id}: malformed markup: {exc}") from exc

        if text != original:
            raise GoldImportError(
                f"{candidate_id}: annotated text does not match the exported original -- "
                "only bracket annotations may be added or corrected, not the prose itself"
            )

        examples.append(
            NerExample(
                id=f"gold:{candidate_id}", text=text, entities=spans, source="gold", split=split
            )
        )
    return examples
