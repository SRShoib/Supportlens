"""Stratified train/val/test splits for intent (Bitext) and urgency (Twitter,
weak-labeled) — persisted to data/splits/ so M2's baselines and M3's
transformers train on identically the same rows (SPEC M3: "same splits").

Run: uv run python -m ml.training.splits
"""

from pathlib import Path

import pandas as pd
from api.config import get_settings
from api.db.models import AuthorRole, Message, Ticket, TicketSource
from api.db.session import SessionLocal
from sklearn.model_selection import train_test_split
from sqlalchemy import select
from sqlalchemy.orm import Session

from ml.data.weak_labels import weak_label_urgency

SPLITS_DIR = Path("data/splits")
TRAIN_FRAC = 0.70


def _stratified_split(ids: list[str], labels: list[str], seed: int) -> dict[str, str]:
    """train/val/test (70/15/15), stratified by label."""
    train_ids, temp_ids, _train_labels, temp_labels = train_test_split(
        ids, labels, train_size=TRAIN_FRAC, stratify=labels, random_state=seed
    )
    val_ids, test_ids = train_test_split(
        temp_ids, train_size=0.5, stratify=temp_labels, random_state=seed
    )
    assignment = dict.fromkeys(train_ids, "train")
    assignment.update(dict.fromkeys(val_ids, "val"))
    assignment.update(dict.fromkeys(test_ids, "test"))
    return assignment


def build_intent_splits(session: Session, seed: int = 42) -> pd.DataFrame:
    """One row per Bitext ticket: the customer's instruction text -> intent
    label. Uses only the customer message — intent classifies the request,
    not the agent's canned response."""
    tickets = session.scalars(select(Ticket).where(Ticket.source == TicketSource.BITEXT)).all()

    rows = []
    for ticket in tickets:
        intent = ticket.meta.get("intent")
        customer_message = next(
            (m for m in ticket.messages if m.author_role == AuthorRole.CUSTOMER), None
        )
        if not intent or customer_message is None:
            continue
        rows.append({"id": str(ticket.id), "text": customer_message.text_clean, "label": intent})

    df = pd.DataFrame(rows)
    assignment = _stratified_split(df["id"].tolist(), df["label"].tolist(), seed)
    df["split"] = df["id"].map(assignment)
    return df


def build_urgency_splits(session: Session, seed: int = 42) -> pd.DataFrame:
    """One row per Twitter customer message -> weak_label_urgency() label."""
    stmt = (
        select(Message)
        .join(Ticket, Ticket.id == Message.ticket_id)
        .where(Ticket.source == TicketSource.TWITTER, Message.author_role == AuthorRole.CUSTOMER)
    )
    messages = session.scalars(stmt).all()

    rows = [
        {"id": str(m.id), "text": m.text_clean, "label": weak_label_urgency(m.text_clean)}
        for m in messages
    ]
    df = pd.DataFrame(rows)
    assignment = _stratified_split(df["id"].tolist(), df["label"].tolist(), seed)
    df["split"] = df["id"].map(assignment)
    return df


def save_splits(df: pd.DataFrame, name: str) -> Path:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLITS_DIR / f"{name}.parquet"
    df.to_parquet(path, engine="pyarrow", index=False)
    return path


def load_splits(name: str) -> pd.DataFrame:
    return pd.read_parquet(SPLITS_DIR / f"{name}.parquet", engine="pyarrow")


def main() -> None:
    seed = get_settings().random_seed
    session = SessionLocal()
    try:
        intent_df = build_intent_splits(session, seed)
        urgency_df = build_urgency_splits(session, seed)
    finally:
        session.close()

    intent_path = save_splits(intent_df, "intent_v1")
    urgency_path = save_splits(urgency_df, "urgency_v1")

    for name, df in [("intent", intent_df), ("urgency", urgency_df)]:
        counts = df["split"].value_counts().to_dict()
        label_counts = df["label"].value_counts().to_dict()
        print(f"{name}: {len(df)} rows -> splits={counts} labels={label_counts}")

    print(f"saved: {intent_path}, {urgency_path}")


if __name__ == "__main__":
    main()
