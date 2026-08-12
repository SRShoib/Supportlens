import uuid

from api.db.models import AuthorRole, Message, Ticket, TicketSource

from ml.data.llm_judge_summaries import _build_prompt, _parse_scores


def test_parse_scores_extracts_both_numbers() -> None:
    faithfulness, coverage, parsed_ok = _parse_scores("faithfulness: 4\ncoverage: 2")
    assert (faithfulness, coverage, parsed_ok) == (4, 2, True)


def test_parse_scores_is_case_insensitive_and_tolerates_extra_whitespace() -> None:
    faithfulness, coverage, parsed_ok = _parse_scores("Faithfulness:  5\nCoverage:3")
    assert (faithfulness, coverage, parsed_ok) == (5, 3, True)


def test_parse_scores_tolerates_extra_surrounding_text() -> None:
    response = "Sure, here are the scores.\nfaithfulness: 1\ncoverage: 5\nThanks!"
    faithfulness, coverage, parsed_ok = _parse_scores(response)
    assert (faithfulness, coverage, parsed_ok) == (1, 5, True)


def test_parse_scores_falls_back_to_neutral_on_unparseable_response() -> None:
    faithfulness, coverage, parsed_ok = _parse_scores("I refuse to grade this.")
    assert (faithfulness, coverage, parsed_ok) == (3, 3, False)


def test_parse_scores_rejects_out_of_range_scores() -> None:
    # 0 and 6 are outside the 1-5 rubric range -- the regex shouldn't match
    # digits outside [1-5], so this falls back rather than accepting garbage.
    _faithfulness, _coverage, parsed_ok = _parse_scores("faithfulness: 0\ncoverage: 6")
    assert parsed_ok is False


def _make_ticket(texts_and_roles: list[tuple[str, AuthorRole]]) -> Ticket:
    ticket = Ticket(
        id=uuid.uuid4(), source=TicketSource.TWITTER, external_id="1", channel="twitter"
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
                content_hash=str(seq),
                external_id=str(seq),
            )
        )
    return ticket


def test_build_prompt_includes_dialogue_and_summary() -> None:
    ticket = _make_ticket(
        [("my order is late", AuthorRole.CUSTOMER), ("sorry about that", AuthorRole.AGENT)]
    )

    prompt = _build_prompt(ticket, "Customer's order was late; agent apologized.")

    assert "Customer: my order is late" in prompt
    assert "Agent: sorry about that" in prompt
    assert "Customer's order was late; agent apologized." in prompt
    assert prompt.index("Customer: my order is late") < prompt.index("Summary:")
