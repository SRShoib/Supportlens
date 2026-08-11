from pathlib import Path

import pytest

from ml.inference.token_classification import TokenClassificationPredictor

STUB_MODEL_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "models" / "stub_ner"


@pytest.fixture
def predictor() -> TokenClassificationPredictor:
    return TokenClassificationPredictor(STUB_MODEL_DIR)


def test_predict_returns_one_result_per_text(predictor: TokenClassificationPredictor) -> None:
    results = predictor.predict(["order shipped yesterday", "charged for my iphone"])
    assert len(results) == 2


def test_predict_empty_list_returns_empty(predictor: TokenClassificationPredictor) -> None:
    assert predictor.predict([]) == []


def test_every_span_offset_matches_text_on_real_model_output(
    predictor: TokenClassificationPredictor,
) -> None:
    texts = [
        "order shipped yesterday",
        "charged for my iphone",
        "no entities here at all",
        "account was debited last week since case open",
    ]
    for text, result in zip(texts, predictor.predict(texts), strict=True):
        for span in result.entities:
            assert text[span.start : span.end] == span.text


def test_every_span_label_is_a_known_entity_type(predictor: TokenClassificationPredictor) -> None:
    from ml.inference.rules_ner import ENTITY_LABELS

    results = predictor.predict(["order shipped yesterday charged for my iphone"])
    for span in results[0].entities:
        assert span.label in ENTITY_LABELS


def test_missing_export_dir_raises() -> None:
    with pytest.raises(FileNotFoundError):
        TokenClassificationPredictor(Path("does/not/exist"))


def test_truncated_flag_present_on_every_result(predictor: TokenClassificationPredictor) -> None:
    results = predictor.predict(["short text"])
    assert isinstance(results[0].truncated, bool)
