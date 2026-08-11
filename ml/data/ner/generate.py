"""Synthetic NER dataset generator (SPEC M4 section 2: templates + LLM
paraphrase pass, injected into real ticket shells, ~3-5k sentences).
Produces a complete, trainable dataset with zero paid calls -- the optional
paraphrase pass (ml/data/ner/paraphrase.py) is a strict add-on, never a
prerequisite.

Composition of the n_total examples: ~48% pure template, ~15% template
wrapped around a real zero-entity shell, ~10% template spliced into a real
shell at a sentence boundary, ~12% (capped by availability) a real shell's
single rules-detected entity substituted with a fresh value, and whatever
remains (~15%) real zero-entity shells used unmodified as negatives -- the
cheapest lever against a token classifier that over-predicts. Splits are a
single global seeded 70/15/15 shuffle across the whole dataset, not
stratified by template or source.

Run (needs an ingested Twitter slice -- see docs/m4-how-to-run-locally.md):
  uv run python -m ml.data.ner.generate
"""

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from api.config import get_settings
from api.db.models import AuthorRole, Message, Ticket, TicketSource
from api.db.session import SessionLocal
from sqlalchemy import select
from sqlalchemy.orm import Session

from ml.data.ner.pools import (
    sample_account_ref,
    sample_amount,
    sample_date,
    sample_order_id,
    sample_product,
)
from ml.data.ner.schema import CharSpan, NerExample, write_jsonl
from ml.data.ner.shells import (
    Partition,
    is_fixed_point,
    partition_message_ids,
    sentence_splice,
    single_entity_span,
    slot_substitute,
    wrap,
)
from ml.data.ner.templates import TEMPLATES, Template, render, slot_label, template_slots
from ml.inference.rules_ner import extract_spans

OUTPUT_PATH = Path("data/splits/ner_v1.jsonl")
MANIFEST_PATH = Path("data/splits/ner_v1.manifest.json")
POOLS_PARTITION_PATH = Path("data/splits/ner_pools_v1.json")
GOLD_FRACTION = 0.10

_SAMPLERS = {
    "ORDER_ID": sample_order_id,
    "PRODUCT": sample_product,
    "DATE": sample_date,
    "AMOUNT": sample_amount,
    "ACCOUNT_REF": sample_account_ref,
}


@dataclass(frozen=True)
class GenerationStats:
    n_total: int
    by_source: dict[str, int]
    shells_considered: int
    shells_dropped_not_fixed_point: int


def _render_template(template: Template, rng: random.Random) -> tuple[str, list[CharSpan]]:
    values = {slot: _SAMPLERS[slot_label(slot)](rng) for slot in template_slots(template.text)}
    return render(template.text, values)


def _classify_shells(
    shells: Sequence[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], int]:
    """Splits a candidate real-message pool into (zero-entity, single-entity)
    buckets. A shell that isn't a clean_text fixed point is dropped and
    counted, never silently used (the offset contract every M4 component
    shares -- see ml.inference.base.EntitySpan). A shell with 2+
    rules-detected entities is neither bucket's business in v1 and is
    excluded without inflating the drop counter, since it isn't a bug --
    just unused."""
    zero_entity: list[tuple[str, str]] = []
    single_entity: list[tuple[str, str]] = []
    dropped = 0
    for msg_id, text in shells:
        if not is_fixed_point(text):
            dropped += 1
            continue
        n_spans = len(extract_spans(text))
        if n_spans == 0:
            zero_entity.append((msg_id, text))
        elif n_spans == 1:
            single_entity.append((msg_id, text))
    return zero_entity, single_entity, dropped


def _split_sequence(n: int, rng: random.Random) -> list[str]:
    n_train = round(n * 0.70)
    n_val = round(n * 0.15)
    n_test = n - n_train - n_val
    sequence = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    rng.shuffle(sequence)
    return sequence


def compose_dataset(
    rng: random.Random,
    shell_pool: Sequence[tuple[str, str]],
    *,
    n_total: int = 4000,
    template_share: float = 0.48,
    wrap_share: float = 0.15,
    sentence_share: float = 0.10,
    slot_sub_share: float = 0.12,
    templates: Sequence[Template] = TEMPLATES,
) -> tuple[list[NerExample], GenerationStats]:
    """Builds the full synthetic dataset from a candidate real-message pool.
    Callers must pass only messages from Partition.shell_ids (never
    Partition.gold_ids) -- that's what makes "no gold message appears in
    training" a mechanical property rather than a hope."""
    zero_entity, single_entity, dropped = _classify_shells(shell_pool)

    n_template = round(n_total * template_share)
    n_wrap = round(n_total * wrap_share)
    n_sentence = round(n_total * sentence_share)
    n_slot_sub = min(round(n_total * slot_sub_share), len(single_entity))
    n_negative = max(n_total - n_template - n_wrap - n_sentence - n_slot_sub, 0)

    examples: list[NerExample] = []
    by_source: dict[str, int] = {}

    def record(example: NerExample) -> None:
        examples.append(example)
        by_source[example.source] = by_source.get(example.source, 0) + 1

    def sample_zero_entity() -> tuple[str, str]:
        return zero_entity[rng.randrange(len(zero_entity))]

    for i in range(n_template):
        template = rng.choice(list(templates))
        text, spans = _render_template(template, rng)
        record(
            NerExample(
                id=f"tpl:{i:06d}",
                text=text,
                entities=spans,
                source="template",
                split="",
                template_id=template.id,
            )
        )

    for i in range(n_wrap):
        if not zero_entity:
            break
        template = rng.choice(list(templates))
        rendered_text, rendered_spans = _render_template(template, rng)
        msg_id, shell_text = sample_zero_entity()
        text, spans = wrap(
            shell_text, rendered_text, rendered_spans, shell_first=rng.random() < 0.5
        )
        record(
            NerExample(
                id=f"shell:{msg_id}:wrap:{i:06d}",
                text=text,
                entities=spans,
                source="shell_wrap",
                split="",
                template_id=template.id,
            )
        )

    for i in range(n_sentence):
        if not zero_entity:
            break
        template = rng.choice(list(templates))
        rendered_text, rendered_spans = _render_template(template, rng)
        msg_id, shell_text = sample_zero_entity()
        spliced = sentence_splice(shell_text, rendered_text, rendered_spans, rng)
        if spliced is None:
            spliced = wrap(shell_text, rendered_text, rendered_spans, shell_first=True)
        text, spans = spliced
        record(
            NerExample(
                id=f"shell:{msg_id}:sentence:{i:06d}",
                text=text,
                entities=spans,
                source="shell_sentence",
                split="",
                template_id=template.id,
            )
        )

    for i in range(n_slot_sub):
        msg_id, shell_text = single_entity[i % len(single_entity)]
        target = single_entity_span(shell_text)
        assert target is not None, "single_entity bucket is guaranteed to have exactly one span"
        new_value = _SAMPLERS[target.label](rng)
        text, spans = slot_substitute(shell_text, target, new_value)
        record(
            NerExample(
                id=f"shell:{msg_id}:slotsub:{i:06d}",
                text=text,
                entities=spans,
                source="shell_slot_sub",
                split="",
                template_id=None,
            )
        )

    for i in range(n_negative):
        if not zero_entity:
            break
        msg_id, shell_text = sample_zero_entity()
        record(
            NerExample(
                id=f"shell:{msg_id}:negative:{i:06d}",
                text=shell_text,
                entities=[],
                source="negative",
                split="",
                template_id=None,
            )
        )

    splits = _split_sequence(len(examples), rng)
    examples = [
        replace(example, split=split) for example, split in zip(examples, splits, strict=True)
    ]

    stats = GenerationStats(
        n_total=len(examples),
        by_source=by_source,
        shells_considered=len(shell_pool),
        shells_dropped_not_fixed_point=dropped,
    )
    return examples, stats


def _fetch_shell_candidates(session: Session) -> list[tuple[str, str]]:
    stmt = (
        select(Message.id, Message.text_clean)
        .join(Ticket, Ticket.id == Message.ticket_id)
        .where(Ticket.source == TicketSource.TWITTER, Message.author_role == AuthorRole.CUSTOMER)
    )
    return [(str(message_id), text) for message_id, text in session.execute(stmt).all()]


def _write_manifest(stats: GenerationStats, seed: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "n_total": stats.n_total,
                "by_source": stats.by_source,
                "shells_considered": stats.shells_considered,
                "shells_dropped_not_fixed_point": stats.shells_dropped_not_fixed_point,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_partition(partition: Partition, seed: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "gold_fraction": GOLD_FRACTION,
                "shell_ids": sorted(partition.shell_ids),
                "gold_ids": sorted(partition.gold_ids),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-total", type=int, default=4000)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    settings = get_settings()
    session = SessionLocal()
    try:
        candidates = _fetch_shell_candidates(session)
    finally:
        session.close()

    partition = partition_message_ids(
        [msg_id for msg_id, _ in candidates], settings.random_seed, GOLD_FRACTION
    )
    by_id = dict(candidates)
    shell_pool = [(msg_id, by_id[msg_id]) for msg_id in partition.shell_ids]

    rng = random.Random(settings.random_seed)
    examples, stats = compose_dataset(rng, shell_pool, n_total=args.n_total)

    write_jsonl(examples, args.output)
    _write_manifest(stats, settings.random_seed, MANIFEST_PATH)
    _write_partition(partition, settings.random_seed, POOLS_PARTITION_PATH)

    print(f"wrote {stats.n_total} examples -> {args.output}")
    print(f"by source: {stats.by_source}")
    print(
        f"shells considered={stats.shells_considered} "
        f"dropped_not_fixed_point={stats.shells_dropped_not_fixed_point}"
    )
    print(f"gold pool: {len(partition.gold_ids)} message ids reserved -> {POOLS_PARTITION_PATH}")


if __name__ == "__main__":
    main()
