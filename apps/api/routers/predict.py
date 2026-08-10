from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.schemas.predict import PredictRequest, PredictResponse, TaskResultOut
from ml.inference.baseline import BaselinePredictor

router = APIRouter(prefix="/predict", tags=["predict"])

_INTENT_MODEL_PATH = Path("models/baseline_intent_v1/model.joblib")


@lru_cache
def _get_intent_predictor() -> BaselinePredictor:
    return BaselinePredictor(_INTENT_MODEL_PATH)


@router.post("/intent", response_model=PredictResponse)
def predict_intent(request: PredictRequest) -> PredictResponse:
    try:
        predictor = _get_intent_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="intent model not available; run `make train-baseline-intent` first",
        ) from exc

    results = predictor.predict(request.texts)
    return PredictResponse(
        results=[
            TaskResultOut(label=r.label, score=r.score, probabilities=r.probabilities)
            for r in results
        ]
    )
