"""FLAN-T5 thread-summarization CPU serving (SPEC M6; SPEC §3: serving is
CPU-only). Same shape as ml/inference/transformer.py's classification
wrapper, but AutoModelForSeq2SeqLM + .generate() instead of a classification
head, and no label_map.json (the output is free text, not a fixed label
set).

PROMPT_TEMPLATE is FLAN-T5-specific: FLAN-T5 is instruction-tuned, so it's
prompted with a plain-English instruction rather than raw T5's terse
"summarize: " prefix convention. ml/training/train_summarization.py imports
this same constant so the fine-tune sees exactly the prompt shape this
predictor serves with at inference time -- a mismatch between train-time and
serve-time prompting would silently degrade quality without erroring.

Input contract: predict() takes one already-formatted dialogue string per
ticket (ml.inference.base.format_dialogue's output) -- the same shape
ml.inference.extractive_summary.ExtractiveSummaryPredictor consumes.
"""

from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ml.inference.base import SummaryResult

PROMPT_TEMPLATE = "Summarize the following conversation:\n{dialogue}"


class SummarizationPredictor:
    def __init__(
        self,
        export_dir: Path,
        max_source_length: int = 512,
        max_new_tokens: int = 64,
        num_beams: int = 2,
    ) -> None:
        if not (export_dir / "model.safetensors").exists():
            raise FileNotFoundError(f"no exported model at {export_dir}")

        self._tokenizer = AutoTokenizer.from_pretrained(str(export_dir))
        self._model = AutoModelForSeq2SeqLM.from_pretrained(str(export_dir))
        self._model.eval()
        self._max_source_length = max_source_length
        self._max_new_tokens = max_new_tokens
        self._num_beams = num_beams

    def predict(self, texts: list[str]) -> list[SummaryResult]:
        if not texts:
            return []

        prompts = [PROMPT_TEMPLATE.format(dialogue=text) for text in texts]
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self._max_source_length,
            # T5 doesn't accept token_type_ids -- real T5 tokenizers never
            # emit them, but a generic PreTrainedTokenizerFast (e.g. the
            # test stub, tests/fixtures/models/stub_transformer_thread_summary)
            # does by default, which .generate() then rejects outright.
            return_token_type_ids=False,
        )
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens, num_beams=self._num_beams
            )
        summaries = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return [SummaryResult(summary=s.strip()) for s in summaries]
