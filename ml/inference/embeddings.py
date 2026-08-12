"""sentence-transformers embedding wrapper (SPEC M7: "Embed the real-ticket
corpus (sentence-transformers)"). Two entry points:

- `predict()` conforms to ml/inference/base.py's Predictor protocol, one
  EmbeddingResult per input text -- so ml/evaluation/latency.py's existing
  benchmark_latency helper measures this against SPEC §3's <100 ms
  embedding budget unchanged, no bespoke benchmarking code needed.
- `encode()` is the bulk path scripts/compute_embeddings.py actually uses
  for ~36k tickets: one batched call returning a 2D numpy array, far faster
  than one predict() call per text.

Deliberately never imported by apps/api: the API reads topic assignments
back from Postgres (scripts/assign_topics.py writes them offline), it never
loads an embedding model itself (docs/decisions.md). Only this module's
callers -- scripts/compute_embeddings.py, ml/training/topic_model.py -- sit
behind the `topics` dependency group.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from ml.inference.base import EmbeddingResult

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceEmbeddingPredictor:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model = SentenceTransformer(model_name)

    def predict(self, texts: list[str]) -> list[EmbeddingResult]:
        vectors = self.encode(texts, show_progress_bar=False)
        return [EmbeddingResult(vector=vector.tolist()) for vector in vectors]

    def encode(
        self, texts: list[str], *, batch_size: int = 64, show_progress_bar: bool = True
    ) -> np.ndarray:
        result = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        return np.asarray(result)
