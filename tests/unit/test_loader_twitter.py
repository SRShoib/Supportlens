from pathlib import Path

from ml.data.loaders import twitter

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "twcs_sample.csv"


def test_conversation_grouping_counts() -> None:
    conversations = twitter.build_conversations(FIXTURE)
    counts = sorted(s.message_count for s in conversations.values())
    assert counts == [1, 1, 1, 2, 3, 3]


def test_branching_replies_merge_into_one_conversation() -> None:
    conversations = twitter.build_conversations(FIXTURE)
    branching = next(s for s in conversations.values() if "200" in s.tweet_ids)
    assert branching.tweet_ids == frozenset({"200", "201", "202"})
    assert branching.brand == "AmazonHelp"


def test_self_reply_does_not_crash_or_loop() -> None:
    conversations = twitter.build_conversations(FIXTURE)
    solo = next(s for s in conversations.values() if "300" in s.tweet_ids)
    assert solo.tweet_ids == frozenset({"300"})


def test_orphan_reply_included_as_its_own_conversation() -> None:
    conversations = twitter.build_conversations(FIXTURE)
    orphan = next(s for s in conversations.values() if "400" in s.tweet_ids)
    assert orphan.tweet_ids == frozenset({"400"})


def test_seq_ordering_follows_reply_chain_then_time() -> None:
    tickets = {t.external_id: t for t in twitter.iter_tickets(FIXTURE)}
    linear = tickets["102"]
    assert [m.external_id for m in linear.messages] == ["100", "101", "102"]
    assert [m.seq for m in linear.messages] == [0, 1, 2]

    branching = tickets["202"]
    assert branching.messages[0].external_id == "200"
    assert {m.external_id for m in branching.messages[1:]} == {"201", "202"}
    assert branching.messages[1].seq < branching.messages[2].seq


def test_author_role_from_inbound() -> None:
    tickets = {t.external_id: t for t in twitter.iter_tickets(FIXTURE)}
    linear = tickets["102"]
    assert linear.messages[0].author_role.value == "customer"
    assert linear.messages[1].author_role.value == "agent"


def test_brand_detected_from_first_outbound_author() -> None:
    tickets = {t.external_id: t for t in twitter.iter_tickets(FIXTURE)}
    assert tickets["102"].brand == "AppleSupport"
    assert tickets["202"].brand == "AmazonHelp"


def test_timestamps_parsed_as_timezone_aware() -> None:
    tickets = {t.external_id: t for t in twitter.iter_tickets(FIXTURE)}
    linear = tickets["102"]
    assert linear.created_at is not None
    assert linear.created_at.tzinfo is not None
    assert linear.messages[0].sent_at is not None
    assert linear.messages[0].sent_at.tzinfo is not None


def test_non_english_conversation_detected() -> None:
    tickets = {t.external_id: t for t in twitter.iter_tickets(FIXTURE)}
    spanish = tickets["501"]
    assert spanish.messages[0].lang == "es"


def test_mojibake_repaired() -> None:
    tickets = {t.external_id: t for t in twitter.iter_tickets(FIXTURE)}
    mojibake_ticket = tickets["600"]
    assert "It's been broken" in mojibake_ticket.messages[0].text_clean


def test_ticket_id_never_collides_with_a_message_id() -> None:
    """The union-find root_id is always drawn from a real member tweet_id (e.g.
    ticket "102" below has root "102" AND a message with external_id "102"), so
    without a type discriminator in ml.data.ids.deterministic_id, the ticket's
    id and that message's id would be identical UUIDs."""
    tickets = list(twitter.iter_tickets(FIXTURE))
    assert any(t.external_id == "102" for t in tickets), "fixture must exercise root==member case"

    for ticket in tickets:
        message_ids = {m.id for m in ticket.messages}
        assert ticket.id not in message_ids


def test_deterministic_ids_across_runs() -> None:
    run1 = list(twitter.iter_tickets(FIXTURE))
    run2 = list(twitter.iter_tickets(FIXTURE))
    ids1 = sorted(t.id for t in run1)
    ids2 = sorted(t.id for t in run2)
    assert ids1 == ids2


def test_selected_roots_filters_output() -> None:
    conversations = twitter.build_conversations(FIXTURE)
    one_root = next(iter(conversations))
    tickets = list(twitter.iter_tickets(FIXTURE, selected_roots={one_root}))
    assert len(tickets) == 1


def test_parse_created_at_invalid_returns_none() -> None:
    assert twitter.parse_created_at("not a date") is None
    assert twitter.parse_created_at(None) is None
    assert twitter.parse_created_at("") is None
