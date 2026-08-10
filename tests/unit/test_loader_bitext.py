import json
from pathlib import Path

from ml.data.loaders import bitext

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "bitext_sample.jsonl"


def _load_fixture_rows() -> list[bitext.BitextRow]:
    with FIXTURE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_two_messages_per_ticket() -> None:
    rows = _load_fixture_rows()
    tickets = list(bitext.iter_tickets(rows=rows))

    assert len(tickets) == len(rows)
    for ticket in tickets:
        assert len(ticket.messages) == 2
        assert ticket.messages[0].seq == 0
        assert ticket.messages[0].author_role.value == "customer"
        assert ticket.messages[1].seq == 1
        assert ticket.messages[1].author_role.value == "agent"


def test_intent_and_category_land_in_meta() -> None:
    tickets = list(bitext.iter_tickets(rows=_load_fixture_rows()))
    first = tickets[0]

    assert first.meta["intent"] == "cancel_order"
    assert first.meta["category"] == "ORDER"


def test_source_is_bitext_with_synthetic_channel() -> None:
    tickets = list(bitext.iter_tickets(rows=_load_fixture_rows()))

    assert all(t.source.value == "bitext" for t in tickets)
    assert all(t.channel == "synthetic" for t in tickets)


def test_created_at_is_none_for_bitext() -> None:
    tickets = list(bitext.iter_tickets(rows=_load_fixture_rows()))
    assert all(t.created_at is None for t in tickets)


def test_deterministic_ids_across_runs() -> None:
    rows = _load_fixture_rows()
    run1 = list(bitext.iter_tickets(rows=rows))
    run2 = list(bitext.iter_tickets(rows=rows))

    assert [t.id for t in run1] == [t.id for t in run2]
    assert [m.id for t in run1 for m in t.messages] == [m.id for t in run2 for m in t.messages]


def test_text_is_cleaned() -> None:
    rows = [
        {
            "flags": "B",
            "instruction": "check https://x.co/a now",
            "category": "X",
            "intent": "y",
            "response": "ok will do",
        }
    ]
    tickets = list(bitext.iter_tickets(rows=rows))
    assert tickets[0].messages[0].text_clean == "check <URL> now"
    assert tickets[0].messages[0].text_raw == "check https://x.co/a now"
