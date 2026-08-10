from ml.data.masking import (
    ALL_MASK_TOKENS,
    EMOJI_TOKEN_PREFIX,
    additional_special_tokens,
    mask_entities,
)


def test_golden_token_strings_are_pinned() -> None:
    """Regression guard: these literal strings are the M2/M3 vocabulary contract."""
    assert additional_special_tokens() == ["<URL>", "<USER>", "<EMAIL>", "<PHONE>"]
    assert {"<URL>", "<USER>", "<EMAIL>", "<PHONE>"} == ALL_MASK_TOKENS
    assert EMOJI_TOKEN_PREFIX == "<EMOJI:"


def test_masks_url() -> None:
    assert mask_entities("check https://example.com/path?a=1 now") == "check <URL> now"


def test_masks_www_url() -> None:
    assert mask_entities("go to www.example.com today") == "go to <URL> today"


def test_url_trailing_punctuation_preserved_outside_token() -> None:
    assert mask_entities("see http://example.com/a.") == "see <URL>."
    assert mask_entities("(http://example.com/a)") == "(<URL>)"


def test_masks_handle_at_string_start() -> None:
    assert mask_entities("@AppleSupport my phone is broken") == "<USER> my phone is broken"


def test_masks_email_before_handle_collision() -> None:
    assert mask_entities("contact a@b.com please") == "contact <EMAIL> please"


def test_email_inside_url_is_swallowed_by_url_token() -> None:
    result = mask_entities("see http://x.com/?email=a@b.com")
    assert result == "see <URL>"


def test_masks_phone_number() -> None:
    assert mask_entities("call 555-123-4567 now") == "call <PHONE> now"


def test_order_ids_amounts_dates_survive_unmasked() -> None:
    """Masking must never destroy the M4 NER targets."""
    text = "Order #ORD-98234 for $49.99 shipped on 2024-01-15"
    result = mask_entities(text)
    assert "ORD-98234" in result
    assert "49.99" in result
    assert "2024-01-15" in result


def test_idempotent() -> None:
    text = "@user check https://x.com/a email a@b.com call 555-123-4567"
    once = mask_entities(text)
    twice = mask_entities(once)
    assert once == twice


def test_no_mask_needed_returns_unchanged() -> None:
    assert mask_entities("just plain text here") == "just plain text here"
