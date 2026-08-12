"""SPEC M8's "resolved tickets" -- no explicit resolution status exists
anywhere in the canonical schema (no `status` column, no loader ever sets
one, docs/decisions.md). This module defines "resolved" from a signal M5
already computed: a Twitter ticket counts as resolved when its
sentiment_trajectory Prediction's resolution_quality
(ml/inference/sentiment_trajectory.py, stored as Prediction.score for
task="sentiment_trajectory") is strictly positive -- the customer's last
word was net positive, discounted by how urgent the ticket opened.

Restricted to TicketSource.TWITTER, same as every real-corpus module since
M7: Bitext is synthetic single-turn instruction/response pairs with no real
resolution to speak of (docs/decisions.md's M7 entries).

An earlier candidate definition -- "has at least one agent message" -- was
measured against the real corpus and rejected: 36,578 of 36,579 Twitter
tickets have an agent reply (99.997%). The twcs dataset's conversation
grouping is curated around brand-response threads, so nearly every captured
ticket includes one by construction; that definition doesn't discriminate
anything. resolution_quality > 0 does -- roughly 21% of the Twitter corpus
at the real-run scale (see docs/decisions.md).

Requires scripts/compute_sentiment_trajectories.py (SPEC M5) to have already
run -- resolved_ticket_ids returns nothing for a ticket with no
sentiment_trajectory Prediction yet, rather than erroring.
"""

from api.db.models import Prediction, Ticket, TicketSource
from sqlalchemy import select
from sqlalchemy.orm import Session

RESOLUTION_TASK = "sentiment_trajectory"


def resolved_ticket_ids(session: Session) -> list[str]:
    stmt = (
        select(Prediction.ticket_id)
        .join(Ticket, Prediction.ticket_id == Ticket.id)
        .where(
            Ticket.source == TicketSource.TWITTER,
            Prediction.task == RESOLUTION_TASK,
            Prediction.score > 0,
        )
    )
    return [str(ticket_id) for ticket_id in session.execute(stmt).scalars().all()]
