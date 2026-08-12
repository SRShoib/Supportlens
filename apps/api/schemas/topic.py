import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class TopicOut(BaseModel):
    """The full catalog, including the topic_key=-1 "outliers" row when one
    exists -- SPEC M7's "≥ 30 coherent topics" excludes it from the count,
    but it's still useful catalog information (how much of the corpus is
    unclustered)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    topic_key: int
    label: str
    keywords: list[str]
    size: int
    model_version: str
    created_at: datetime


class TopicVolumePoint(BaseModel):
    week: str
    count: int
    share: float
    z_score: float | None
    is_emerging: bool


class TopicVolumeSeries(BaseModel):
    topic_id: int
    label: str
    points: list[TopicVolumePoint]


class TopicVolumeResponse(BaseModel):
    """weeks is the shared analysis window every series is plotted against
    (ml.evaluation.trend_metrics.select_dense_window's output) -- empty when
    there isn't enough dense weekly history yet (e.g. before
    scripts/assign_topics.py has been run)."""

    weeks: list[str]
    series: list[TopicVolumeSeries]


class EmergingIssueOut(BaseModel):
    topic_id: int
    label: str
    week: str
    count: int
    share: float
    z_score: float
