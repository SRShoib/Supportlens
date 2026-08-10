from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TaskResult:
    label: str
    score: float
    probabilities: dict[str, float] | None = None


class Predictor(Protocol):
    """Uniform interface every task predictor implements — baseline today,
    a transformer variant added in M3 without touching the API layer
    (CLAUDE.md: "model choice... selected by config/request flag, not code
    duplication")."""

    def predict(self, texts: list[str]) -> list[TaskResult]: ...
