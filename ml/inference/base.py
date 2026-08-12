from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

R_co = TypeVar("R_co", covariant=True)


class Predictor(Protocol[R_co]):
    """Uniform interface every task predictor implements — baseline today,
    a transformer variant added in M3 without touching the API layer
    (CLAUDE.md: "model choice... selected by config/request flag, not code
    duplication"). Generic over the per-text result type: classification
    tasks return one TaskResult per text, M4's NER task returns one
    EntityResult per text — same length-preserving list[str] -> list[R]
    shape either way, so the interface stays uniform without forcing span
    output into a label/score-shaped box.

    Return type is annotated Sequence rather than list: a covariant TypeVar
    nested in an invariant list[] return trips mypy's protocol-variance
    check (a real limitation, not stylistic) even though every concrete
    predictor still returns a plain list at runtime."""

    def predict(self, texts: list[str]) -> Sequence[R_co]: ...


@dataclass(frozen=True)
class TaskResult:
    label: str
    score: float
    probabilities: dict[str, float] | None = None


@dataclass(frozen=True)
class EntitySpan:
    """One extracted entity. start/end are character offsets into the exact
    string passed to predict() — never into a cleaned or otherwise
    transformed copy of it (see ml/inference/rules_ner.py and
    ml/inference/token_classification.py module docstrings)."""

    start: int
    end: int
    label: str
    text: str
    score: float


@dataclass(frozen=True)
class EntityResult:
    entities: list[EntitySpan] = field(default_factory=list)
    truncated: bool = False


@dataclass(frozen=True)
class SummaryResult:
    summary: str


def format_dialogue(turns: list[tuple[str, str]]) -> str:
    """Renders a ticket's message thread as one newline-joined "Speaker:
    text" string -- the shared input contract every M6 predictor takes
    (ml/inference/extractive_summary.py::ExtractiveSummaryPredictor and
    ml/inference/summarization.py::SummarizationPredictor both split back on
    "\\n" or hand the whole string to a model, never re-derive speaker turns
    themselves). `turns` is (speaker_label, text) pairs in chronological
    order, already resolved by the caller (e.g. "Customer"/"Agent") -- kept
    DB-agnostic here the same way ml/inference/sentiment_trajectory.py takes
    a plain is_customer: list[bool] instead of importing api.db.models."""
    if not turns:
        raise ValueError("format_dialogue requires at least one turn")
    return "\n".join(f"{speaker}: {text}" for speaker, text in turns)


ClassificationPredictor = Predictor[TaskResult]
EntityPredictor = Predictor[EntityResult]
SummaryPredictor = Predictor[SummaryResult]
