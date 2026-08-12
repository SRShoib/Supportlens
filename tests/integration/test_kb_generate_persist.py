import pytest
from api.db.models import KbArticle
from sqlalchemy import select
from sqlalchemy.orm import Session

from ml.data.kb_generate import ArticleSpec, persist_articles

pytestmark = pytest.mark.integration

_SPEC = ArticleSpec(
    source_kind="intent",
    source_key="cancel_order",
    title="How to Cancel an Order",
    intro="Intro text.",
    steps=["Step one."],
    tags=["cancel_order"],
)


def test_persist_articles_writes_one_row_per_spec(db_session: Session) -> None:
    written = persist_articles(db_session, [_SPEC])

    assert written == 1
    articles = db_session.scalars(select(KbArticle)).all()
    assert len(articles) == 1
    assert articles[0].title == "How to Cancel an Order"
    assert articles[0].source_kind == "intent"
    assert articles[0].source_key == "cancel_order"
    assert "1. Step one." in articles[0].body


def test_rerunning_upserts_by_deterministic_id_instead_of_duplicating(db_session: Session) -> None:
    persist_articles(db_session, [_SPEC])
    updated = ArticleSpec(
        source_kind="intent",
        source_key="cancel_order",
        title="How to Cancel an Order (Updated)",
        intro="Updated intro.",
        steps=["Updated step."],
        tags=["cancel_order", "updated"],
    )

    persist_articles(db_session, [updated])

    articles = db_session.scalars(select(KbArticle)).all()
    assert len(articles) == 1
    assert articles[0].title == "How to Cancel an Order (Updated)"
    assert articles[0].tags == ["cancel_order", "updated"]
