"""SPEC M7: topic catalog + weekly volume trend + emerging-issues detection.
Deliberately never loads an embedding or topic model -- every response here
is read straight from what scripts/assign_topics.py already wrote to
Postgres (the `topics` table + Prediction(task="topic") rows), the same
"API only reads durably-stored Predictions" contract GET
/tickets/{id}/predictions already uses for M5/M6 (apps/api/routers/tickets.py).

Weekly bucketing and the z-score itself both live in
ml/evaluation/trend_metrics.py (pure, DB-free, unit-tested there) -- this
module's only job is turning Postgres rows into the
dict[topic_id, dict[week, count]] shape that module expects, and turning
its output back into response schemas. The topic_key=-1 HDBSCAN outlier
cluster is excluded from trend/emerging detection entirely (both the
numerator and the total-volume denominator) -- an "outlier" ticket didn't
join any real topic, so it shouldn't dilute every real topic's share, and
"outliers spiked" isn't a meaningful emerging issue. GET /topics (the
catalog) still includes it, for transparency.
"""

from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import APIRouter
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.db.models import Prediction, Ticket, Topic
from api.deps import DbDep
from api.schemas.topic import (
    EmergingIssueOut,
    TopicOut,
    TopicVolumePoint,
    TopicVolumeResponse,
    TopicVolumeSeries,
)
from ml.evaluation.trend_metrics import compute_topic_trends, select_dense_window

router = APIRouter(prefix="/topics", tags=["topics"])

OUTLIER_TOPIC_KEY = -1


def _week_start(dt: datetime) -> str:
    monday = dt.date() - timedelta(days=dt.weekday())
    return monday.isoformat()


def _load_weekly_counts(db: Session) -> dict[int, dict[str, int]]:
    stmt = (
        select(Prediction, Ticket.created_at)
        .join(Ticket, Prediction.ticket_id == Ticket.id)
        .where(Prediction.task == "topic", Ticket.created_at.isnot(None))
    )
    weekly_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for prediction, created_at in db.execute(stmt).all():
        topic_key = prediction.payload.get("topic_key")
        if topic_key is None or topic_key == OUTLIER_TOPIC_KEY:
            continue
        weekly_counts[topic_key][_week_start(created_at)] += 1
    return weekly_counts


def _compute_topic_volume(db: Session) -> TopicVolumeResponse:
    weekly_counts = _load_weekly_counts(db)
    if not weekly_counts:
        return TopicVolumeResponse(weeks=[], series=[])

    total_by_week: dict[str, int] = defaultdict(int)
    for counts in weekly_counts.values():
        for week, count in counts.items():
            total_by_week[week] += count

    weeks = select_dense_window(total_by_week)
    if not weeks:
        return TopicVolumeResponse(weeks=[], series=[])

    labels_by_key = {t.topic_key: t.label for t in db.scalars(select(Topic)).all()}

    rows_by_topic: dict[int, list[TopicVolumePoint]] = defaultdict(list)
    for row in compute_topic_trends(weekly_counts, weeks):
        rows_by_topic[row.topic_id].append(
            TopicVolumePoint(
                week=row.week,
                count=row.count,
                share=row.share,
                z_score=row.z_score,
                is_emerging=row.is_emerging,
            )
        )

    series = [
        TopicVolumeSeries(
            topic_id=topic_id,
            label=labels_by_key.get(topic_id, "unlabeled"),
            points=points,
        )
        for topic_id, points in sorted(rows_by_topic.items())
    ]
    return TopicVolumeResponse(weeks=weeks, series=series)


@router.get("", response_model=list[TopicOut])
def list_topics(db: DbDep) -> list[Topic]:
    stmt = select(Topic).order_by(Topic.size.desc())
    return list(db.scalars(stmt).all())


@router.get("/volume", response_model=TopicVolumeResponse)
def get_topic_volume(db: DbDep) -> TopicVolumeResponse:
    return _compute_topic_volume(db)


@router.get("/emerging", response_model=list[EmergingIssueOut])
def get_emerging_topics(db: DbDep) -> list[EmergingIssueOut]:
    volume = _compute_topic_volume(db)
    return [
        EmergingIssueOut(
            topic_id=series.topic_id,
            label=series.label,
            week=point.week,
            count=point.count,
            share=point.share,
            z_score=point.z_score,  # is_emerging=True implies z_score is not None here
        )
        for series in volume.series
        for point in series.points
        if point.is_emerging
    ]
