from ml.data.language import detect


def test_detects_english() -> None:
    result = detect("This is a perfectly ordinary English sentence about customer support.")
    assert result.lang == "en"
    assert result.confidence > 0.9


def test_detects_spanish() -> None:
    result = detect("Este es un mensaje completamente normal en español sobre soporte al cliente.")
    assert result.lang == "es"
    assert result.confidence > 0.9


def test_detects_german() -> None:
    result = detect("Dies ist eine ganz normale deutsche Nachricht ueber den Kundenservice heute.")
    assert result.lang == "de"
    assert result.confidence > 0.9


def test_detects_japanese() -> None:
    result = detect("これはカスタマーサポートに関する日本語の文章です。")
    assert result.lang == "ja"
    assert result.confidence > 0.9


def test_short_text_yields_low_confidence() -> None:
    result = detect("thx!!")
    assert result.confidence < 0.70


def test_empty_string_returns_none_lang() -> None:
    result = detect("")
    assert result.lang is None
    assert result.confidence == 0.0


def test_whitespace_only_returns_none_lang() -> None:
    result = detect("   \n\t  ")
    assert result.lang is None
    assert result.confidence == 0.0
