from api.db.models import Topic

from ml.data.llm_topic_labels import _build_prompt, _clean_name


def _topic(keywords: list[str]) -> Topic:
    return Topic(topic_key=0, label="placeholder", keywords=keywords, size=10, model_version="v1")


def test_build_prompt_lists_keywords_most_to_least_characteristic() -> None:
    prompt = _build_prompt(_topic(["refund", "order", "late", "delivery"]))

    assert prompt == "Keywords (most to least characteristic): refund, order, late, delivery"


def test_build_prompt_handles_empty_keywords() -> None:
    prompt = _build_prompt(_topic([]))

    assert prompt == "Keywords (most to least characteristic): "


def test_clean_name_strips_surrounding_quotes() -> None:
    assert _clean_name('"Late Refund Requests"') == "Late Refund Requests"


def test_clean_name_strips_trailing_period() -> None:
    assert _clean_name("Late Refund Requests.") == "Late Refund Requests"


def test_clean_name_strips_surrounding_whitespace() -> None:
    assert _clean_name("  Late Refund Requests  \n") == "Late Refund Requests"


def test_clean_name_handles_all_three_at_once() -> None:
    assert _clean_name('  "Late Refund Requests."  ') == "Late Refund Requests"
