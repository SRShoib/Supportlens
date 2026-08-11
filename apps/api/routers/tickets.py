from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from api.db.models import Prediction, Ticket, TicketSource
from api.deps import DbDep
from api.schemas.prediction import PredictionOut
from api.schemas.ticket import TicketOut

router = APIRouter(prefix="/tickets", tags=["tickets"])


@router.get("", response_model=list[TicketOut])
def list_tickets(
    db: DbDep,
    source: TicketSource | None = None,
    limit: Annotated[int, Query(le=200, ge=1)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[Ticket]:
    stmt = (
        select(Ticket)
        .options(selectinload(Ticket.messages))
        .order_by(Ticket.ingested_at)
        .offset(offset)
        .limit(limit)
    )
    if source is not None:
        stmt = stmt.where(Ticket.source == source)
    return list(db.scalars(stmt).all())


@router.get("/{ticket_id}", response_model=TicketOut)
def get_ticket(ticket_id: UUID, db: DbDep) -> Ticket:
    stmt = select(Ticket).options(selectinload(Ticket.messages)).where(Ticket.id == ticket_id)
    ticket = db.scalars(stmt).first()
    if ticket is None:
        raise HTTPException(status_code=404, detail="ticket not found")
    return ticket


@router.get("/{ticket_id}/predictions", response_model=list[PredictionOut])
def list_ticket_predictions(
    ticket_id: UUID, db: DbDep, task: str | None = None
) -> list[Prediction]:
    """Reads durably-stored per-ticket Predictions (SPEC M5: "per-ticket
    aggregate stored as a Prediction") -- unlike /predict/*, which always
    computes live. `task="sentiment_trajectory"` is what
    scripts/compute_sentiment_trajectories.py writes and what the dashboard's
    sparkline reads; no `task` filter returns every persisted Prediction for
    the ticket. 404s only when the ticket itself doesn't exist, distinct
    from a ticket that exists but has no predictions yet (200, []) -- the
    backfill script may simply not have run."""
    if db.get(Ticket, ticket_id) is None:
        raise HTTPException(status_code=404, detail="ticket not found")

    stmt = select(Prediction).where(Prediction.ticket_id == ticket_id)
    if task is not None:
        stmt = stmt.where(Prediction.task == task)
    return list(db.scalars(stmt).all())
