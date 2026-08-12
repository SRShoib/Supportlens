"""Lead-k extractive summarization baseline (CLAUDE.md ground rule #2:
"Never skip the classical baseline"; SPEC M6 itself only asks for the
FLAN-T5 fine-tune, but every other learned M2-M5 component gets a classical
comparison, and M6 shouldn't be the exception). The summary is simply the
ticket's first k turns, verbatim -- no learned parameters, no model file to
export, mirroring ml/inference/rules_ner.py::RulesEntityPredictor (a
baseline-only deployment never needs transformers/torch installed).

DEFAULT_K=4: a sweep of k=1..6 against the real pooled samsum+dialogsum val
split (1,318 rows) found k=4 maximizes ROUGE-1 (0.3098, vs. k=3's 0.3095 --
effectively a tie, but 4 wins by the letter of "maximize ROUGE-1"; ROUGE-2
keeps climbing past k=6 but ROUGE-1/L both peak at k=3-4). See
docs/decisions.md.

Input contract: predict() takes one already-formatted dialogue string per
ticket (ml.inference.base.format_dialogue's output), turns recovered by
splitting on "\\n" -- the same shape ml.inference.summarization.SummarizationPredictor
consumes, so POST /predict/summary can route model="baseline" and
model="transformer" through an identical request body.
"""

from ml.inference.base import SummaryResult

DEFAULT_K = 4


class ExtractiveSummaryPredictor:
    def __init__(self, k: int = DEFAULT_K) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._k = k

    def predict(self, texts: list[str]) -> list[SummaryResult]:
        results = []
        for dialogue in texts:
            turns = [line for line in dialogue.split("\n") if line.strip()]
            lead = turns[: self._k]
            results.append(SummaryResult(summary=" ".join(lead)))
        return results
