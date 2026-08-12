from ml.data.kb_generate import build_articles, render_body


def test_build_articles_totals_forty_specs() -> None:
    specs = build_articles()
    assert len(specs) == 40


def test_build_articles_has_twenty_seven_intent_and_thirteen_topic_specs() -> None:
    specs = build_articles()
    assert sum(1 for s in specs if s.source_kind == "intent") == 27
    assert sum(1 for s in specs if s.source_kind == "topic") == 13


def test_build_articles_titles_are_unique() -> None:
    specs = build_articles()
    titles = [s.title for s in specs]
    assert len(titles) == len(set(titles))


def test_build_articles_source_keys_are_unique_within_each_kind() -> None:
    specs = build_articles()
    intent_keys = [s.source_key for s in specs if s.source_kind == "intent"]
    topic_keys = [s.source_key for s in specs if s.source_kind == "topic"]
    assert len(intent_keys) == len(set(intent_keys))
    assert len(topic_keys) == len(set(topic_keys))


def test_build_articles_every_spec_has_at_least_one_step_and_tag() -> None:
    for spec in build_articles():
        assert len(spec.steps) >= 1
        assert len(spec.tags) >= 1


def test_render_body_numbers_steps_in_order_and_keeps_the_intro() -> None:
    body = render_body("Do the thing.", ["First step.", "Second step."])

    assert body.startswith("Do the thing.")
    assert "1. First step." in body
    assert "2. Second step." in body
    assert body.index("1. First step.") < body.index("2. Second step.")
