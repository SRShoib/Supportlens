"""Per-ticket sentiment trajectory + a resolution-quality heuristic (SPEC M5:
"aggregate per-ticket into a trajectory... with a simple resolution-quality
heuristic (final-message sentiment x urgency)").

Trajectory: the ordered sequence of per-message *sentiment* labels across a
ticket's full message history (customer + agent). Emotion is fine-tuned and
served the same way as sentiment (POST /predict/emotion) but isn't part of
this aggregate -- tweet_eval's 3-class negative/neutral/positive maps
directly onto a -1/0/+1 sparkline in a way its 4-class emotion labels don't,
so emotion stays a secondary, independently queryable signal rather than
folded into the trajectory line.

Resolution-quality heuristic: "final message" is defined as the ticket's
*last customer message* (the customer's last word on the matter -- an
agent's closing reply isn't the customer's emotional state). "urgency" is
the urgency prediction on the ticket's *first* customer message, the same
per-message unit ml/training/splits.py::build_urgency_splits trained on --
i.e. how urgent the ticket looked when it was opened. Both definitions, and
the exact formula below, are recorded in docs/decisions.md.

Deliberately decoupled from api.db.models (no AuthorRole import): every
other ml/inference/* wrapper is DB-agnostic and unit-testable without a
SQLAlchemy session, and this module keeps that boundary -- callers pass a
plain is_customer: list[bool] instead.
"""

from dataclasses import dataclass

from ml.inference.base import TaskResult

# ml/data/weak_labels.py::UrgencyLabel is the source of truth for these three
# strings. A high-urgency ticket that ends on a negative note should score
# worst; a low-urgency ticket ending positive should score best -- so low
# urgency gets the *least* discount on the signed sentiment, not the most.
URGENCY_WEIGHT: dict[str, float] = {"low": 1.0, "medium": 0.66, "high": 0.33}


def signed_sentiment(result: TaskResult) -> float:
    """+score for positive, -score for negative, 0.0 for neutral -- one
    signed number in [-1, 1] that a sparkline can plot directly without the
    caller needing to know the label set."""
    if result.label == "positive":
        return result.score
    if result.label == "negative":
        return -result.score
    return 0.0


def compute_resolution_quality(final_sentiment: TaskResult, urgency_label: str) -> float:
    """signed_sentiment(final message) * URGENCY_WEIGHT[ticket's opening
    urgency]. Range is [-1, 1]: both factors are bounded in [0, 1] magnitude,
    and the urgency weight only ever discounts (never flips the sign of)
    the sentiment signal."""
    if urgency_label not in URGENCY_WEIGHT:
        raise ValueError(
            f"unknown urgency label: {urgency_label!r} (expected one of {sorted(URGENCY_WEIGHT)})"
        )
    return signed_sentiment(final_sentiment) * URGENCY_WEIGHT[urgency_label]


@dataclass(frozen=True)
class Trajectory:
    sequence: list[str]  # per-message sentiment labels, chronological, all messages
    scores: list[float]  # signed_sentiment per message, same order
    final_customer_label: str  # sentiment label of the last customer message
    resolution_quality: float

    def to_payload(self) -> dict[str, object]:
        """JSONB-safe shape for Prediction.payload (SPEC M5: "per-ticket
        aggregate stored as a Prediction")."""
        return {
            "sequence": self.sequence,
            "scores": self.scores,
            "final_customer_label": self.final_customer_label,
            "resolution_quality": self.resolution_quality,
        }


def build_trajectory(
    sentiment_results: list[TaskResult], is_customer: list[bool], urgency_label: str
) -> Trajectory:
    """sentiment_results is one TaskResult per message, in chronological
    (seq) order across the whole ticket; is_customer marks which of those
    are customer-authored (same length, paired). Requires at least one
    message. If a ticket somehow has no customer message at all, falls
    back to the literal last message for the resolution-quality
    calculation rather than raising -- a rare edge case, not a data error."""
    if not sentiment_results:
        raise ValueError("a ticket must have at least one message to build a trajectory")
    if len(sentiment_results) != len(is_customer):
        raise ValueError("sentiment_results and is_customer must be the same length (paired)")

    customer_results = [r for r, c in zip(sentiment_results, is_customer, strict=True) if c]
    final = customer_results[-1] if customer_results else sentiment_results[-1]

    return Trajectory(
        sequence=[r.label for r in sentiment_results],
        scores=[signed_sentiment(r) for r in sentiment_results],
        final_customer_label=final.label,
        resolution_quality=compute_resolution_quality(final, urgency_label),
    )
