from dataclasses import dataclass

from sklearn.metrics import cohen_kappa_score


@dataclass(frozen=True)
class KappaResult:
    kappa: float
    n: int


def compute_kappa(weak_labels: list[str], llm_labels: list[str]) -> KappaResult:
    """Cohen's kappa agreement between the rule-based weak urgency labels and
    the LLM-labeled seed set (SPEC §2) — reported, not used to "correct" the
    weak labels. Urgency stays a weak-supervision showcase, not ground truth."""
    if len(weak_labels) != len(llm_labels):
        raise ValueError("weak_labels and llm_labels must be the same length (paired)")
    if not weak_labels:
        return KappaResult(kappa=0.0, n=0)
    return KappaResult(kappa=float(cohen_kappa_score(weak_labels, llm_labels)), n=len(weak_labels))
