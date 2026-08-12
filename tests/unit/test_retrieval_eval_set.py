import uuid

from api.db.models import AuthorRole, Message, Ticket, TicketSource

from ml.data.retrieval_eval_set import (
    build_eval_set,
    eligible_tickets,
    first_customer_message,
)


def _ticket(texts_and_roles: list[tuple[str, AuthorRole]]) -> Ticket:
    ticket = Ticket(
        id=uuid.uuid4(),
        source=TicketSource.TWITTER,
        external_id=str(uuid.uuid4()),
        channel="twitter",
    )
    for seq, (text, role) in enumerate(texts_and_roles):
        ticket.messages.append(
            Message(
                id=uuid.uuid4(),
                ticket_id=ticket.id,
                seq=seq,
                author_role=role,
                text_raw=text,
                text_clean=text,
                content_hash=str(uuid.uuid4()),
                external_id=str(seq),
            )
        )
    return ticket


def test_eligible_tickets_requires_at_least_two_customer_messages() -> None:
    one_message = _ticket([("hello", AuthorRole.CUSTOMER)])
    two_messages = _ticket([("hello", AuthorRole.CUSTOMER), ("still broken", AuthorRole.CUSTOMER)])

    assert eligible_tickets([one_message, two_messages]) == [two_messages]


def test_eligible_tickets_counts_only_customer_authored_messages() -> None:
    ticket = _ticket(
        [
            ("hello", AuthorRole.CUSTOMER),
            ("hi, how can I help", AuthorRole.AGENT),
            ("still broken", AuthorRole.CUSTOMER),
            ("let me check", AuthorRole.AGENT),
        ]
    )

    assert eligible_tickets([ticket]) == [ticket]


def test_first_customer_message_returns_the_earliest_by_seq() -> None:
    ticket = _ticket(
        [
            ("first message", AuthorRole.CUSTOMER),
            ("agent reply", AuthorRole.AGENT),
            ("second message", AuthorRole.CUSTOMER),
        ]
    )

    assert first_customer_message(ticket) == "first message"


def test_build_eval_set_pairs_query_with_the_tickets_own_id() -> None:
    ticket = _ticket(
        [("order is late", AuthorRole.CUSTOMER), ("still no update", AuthorRole.CUSTOMER)]
    )

    eval_set = build_eval_set([ticket], sample_size=1)

    assert len(eval_set) == 1
    assert eval_set[0].query == "order is late"
    assert eval_set[0].relevant_ticket_id == str(ticket.id)


def test_build_eval_set_caps_at_sample_size() -> None:
    tickets = [
        _ticket([(f"issue {i}", AuthorRole.CUSTOMER), ("follow up", AuthorRole.CUSTOMER)])
        for i in range(5)
    ]

    eval_set = build_eval_set(tickets, sample_size=3)

    assert len(eval_set) == 3


def test_build_eval_set_is_deterministic_for_a_fixed_seed() -> None:
    tickets = [
        _ticket([(f"issue {i}", AuthorRole.CUSTOMER), ("follow up", AuthorRole.CUSTOMER)])
        for i in range(10)
    ]

    first = build_eval_set(tickets, sample_size=3, seed=42)
    second = build_eval_set(tickets, sample_size=3, seed=42)

    assert [q.relevant_ticket_id for q in first] == [q.relevant_ticket_id for q in second]


def test_build_eval_set_excludes_single_customer_message_tickets() -> None:
    excluded = _ticket([("only message", AuthorRole.CUSTOMER)])
    included = _ticket([("first", AuthorRole.CUSTOMER), ("second", AuthorRole.CUSTOMER)])

    eval_set = build_eval_set([excluded, included], sample_size=10)

    assert [q.relevant_ticket_id for q in eval_set] == [str(included.id)]
