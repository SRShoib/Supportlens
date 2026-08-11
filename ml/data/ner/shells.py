"""Real-message ("shell") plumbing for the synthetic NER generator: the
seeded train/gold partition that makes leakage structurally impossible, and
the offset-arithmetic helpers that splice a rendered template clause into a
real message without disturbing its own (absent, by construction) entities.
"""

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ml.data.cleaning import clean_text
from ml.data.ner.schema import CharSpan
from ml.inference.base import EntitySpan
from ml.inference.rules_ner import extract_spans

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Partition:
    shell_ids: frozenset[str]
    gold_ids: frozenset[str]


def partition_message_ids(ids: Sequence[str], seed: int, gold_fraction: float = 0.10) -> Partition:
    """Splits candidate message ids into a shell pool (used only by the
    generator) and a gold pool (used only by the gold-set exporter), before
    either one runs. Doing this split first -- rather than trusting the
    generator and exporter to independently avoid the same messages -- is
    what makes "no gold message ever appears in training" a mechanical
    property instead of a hope."""
    ordered = sorted(ids)  # deterministic regardless of input/query order
    rng = random.Random(seed)
    shuffled = ordered.copy()
    rng.shuffle(shuffled)
    n_gold = int(len(shuffled) * gold_fraction)
    return Partition(shell_ids=frozenset(shuffled[n_gold:]), gold_ids=frozenset(shuffled[:n_gold]))


def is_fixed_point(text: str) -> bool:
    return clean_text(text) == text


def zero_entity_shell(text: str) -> bool:
    """A real message safe to use as a negative example or as a wrap/splice
    base: a clean_text fixed point in which the rules baseline finds
    nothing, so we never bury an unannotated true entity in training data
    and teach the model to suppress real ones."""
    return is_fixed_point(text) and not extract_spans(text)


def single_entity_span(text: str) -> EntitySpan | None:
    """A real message usable for slot-substitution: a fixed point in which
    the rules baseline finds exactly one entity, which gets replaced by a
    freshly sampled value of the same type."""
    if not is_fixed_point(text):
        return None
    spans = extract_spans(text)
    return spans[0] if len(spans) == 1 else None


def wrap(
    shell: str, rendered_text: str, rendered_spans: Sequence[CharSpan], *, shell_first: bool
) -> tuple[str, list[CharSpan]]:
    """Concatenates a zero-entity shell with a rendered template clause,
    giving the synthetic entity-bearing text a real messy surrounding
    context. `shell` must satisfy zero_entity_shell() -- its own text
    contributes no spans, so only rendered_spans need shifting."""
    if shell_first:
        offset = len(shell) + 1
        text = f"{shell} {rendered_text}"
    else:
        offset = 0
        text = f"{rendered_text} {shell}"
    shifted = [CharSpan(s.start + offset, s.end + offset, s.label, s.text) for s in rendered_spans]
    return text, shifted


def sentence_splice(
    shell: str, rendered_text: str, rendered_spans: Sequence[CharSpan], rng: random.Random
) -> tuple[str, list[CharSpan]] | None:
    """Inserts a rendered clause at a sentence boundary inside a
    zero-entity shell, e.g. "Thanks for the help. <clause>. Really
    appreciated." Returns None if the shell has no sentence boundary to
    splice at -- caller falls back to wrap()."""
    boundaries = [m.start() for m in _SENTENCE_BOUNDARY_RE.finditer(shell)]
    if not boundaries:
        return None
    at = rng.choice(boundaries)
    text = f"{shell[:at]}{rendered_text}. {shell[at:]}"
    shifted = [CharSpan(s.start + at, s.end + at, s.label, s.text) for s in rendered_spans]
    return text, shifted


def slot_substitute(shell: str, target: EntitySpan, new_value: str) -> tuple[str, list[CharSpan]]:
    """Replaces a real message's single rules-detected entity with a fresh
    sampled value of the same type -- the most realistic splice mode, but
    it inherits the rules baseline's own recall bias (it can only find
    shells the rules baseline already recognizes), so callers should cap how
    often it's used relative to the other modes."""
    text = f"{shell[: target.start]}{new_value}{shell[target.end :]}"
    span = CharSpan(target.start, target.start + len(new_value), target.label, new_value)
    return text, [span]
