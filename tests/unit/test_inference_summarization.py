from pathlib import Path

import pytest

from ml.inference.summarization import SummarizationPredictor

STUB_MODEL_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "models" / "stub_transformer_thread_summary"
)


@pytest.fixture
def predictor() -> SummarizationPredictor:
    return SummarizationPredictor(STUB_MODEL_DIR, max_new_tokens=8, num_beams=2)


def test_predict_returns_one_result_per_dialogue(predictor: SummarizationPredictor) -> None:
    dialogues = [
        "Customer: order shipped yesterday\nAgent: please help",
        "Customer: my account was charged twice",
    ]
    results = predictor.predict(dialogues)
    assert len(results) == 2


def test_predict_empty_list_returns_empty(predictor: SummarizationPredictor) -> None:
    assert predictor.predict([]) == []


def test_predict_returns_nonempty_summary_text(predictor: SummarizationPredictor) -> None:
    [result] = predictor.predict(["Customer: order shipped yesterday\nAgent: please help"])
    assert isinstance(result.summary, str)
    assert result.summary != ""


def test_missing_export_dir_raises() -> None:
    with pytest.raises(FileNotFoundError):
        SummarizationPredictor(Path("does/not/exist"))
