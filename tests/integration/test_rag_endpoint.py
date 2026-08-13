import uuid
from unittest.mock import MagicMock

import pytest
from api.config import Settings, get_settings
from api.db import session as session_module
from api.db.models import AuthorRole, Message, Ticket, TicketSource
from api.db.session import make_engine
from api.main import app
from api.routers import rag
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from ml.inference.base import EmbeddingResult
from ml.inference.llm_client import LLMClient
from ml.inference.vector_store import VectorHit

pytestmark = pytest.mark.integration


class FakeEmbedder:
    def predict(self, texts: list[str]) -> list[EmbeddingResult]:
        return [EmbeddingResult(vector=[1.0, 0.0]) for _ in texts]


class FakeStore:
    def __init__(self, hits_by_collection: dict[str, list[VectorHit]]) -> None:
        self._hits_by_collection = hits_by_collection

    def query(self, collection: str, query_embedding, n_results: int = 5, where=None):
        return self._hits_by_collection.get(collection, [])[:n_results]


class FakeReranker:
    def __init__(self, scores_by_document: dict[str, float]) -> None:
        self._scores_by_document = scores_by_document

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [self._scores_by_document.get(doc, -999.0) for doc in documents]


def _mock_openai_response(text: str):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=text))]
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    return response


def _ticket_with_customer_message(session: Session, text: str = "my order is late") -> Ticket:
    ticket = Ticket(
        id=uuid.uuid4(),
        source=TicketSource.TWITTER,
        external_id=str(uuid.uuid4()),
        channel="twitter",
    )
    ticket.messages.append(
        Message(
            id=uuid.uuid4(),
            ticket_id=ticket.id,
            seq=0,
            author_role=AuthorRole.CUSTOMER,
            text_raw=text,
            text_clean=text,
            content_hash=str(uuid.uuid4()),
            external_id="0",
        )
    )
    session.add(ticket)
    session.commit()
    return ticket


def _agent_only_ticket(session: Session) -> Ticket:
    ticket = Ticket(
        id=uuid.uuid4(),
        source=TicketSource.TWITTER,
        external_id=str(uuid.uuid4()),
        channel="twitter",
    )
    ticket.messages.append(
        Message(
            id=uuid.uuid4(),
            ticket_id=ticket.id,
            seq=0,
            author_role=AuthorRole.AGENT,
            text_raw="welcome",
            text_clean="welcome",
            content_hash=str(uuid.uuid4()),
            external_id="0",
        )
    )
    session.add(ticket)
    session.commit()
    return ticket


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, database_url: str) -> TestClient:
    engine = make_engine(database_url)
    monkeypatch.setattr(session_module, "engine", engine)
    monkeypatch.setattr(session_module, "SessionLocal", sessionmaker(bind=engine))
    return TestClient(app)


@pytest.fixture(autouse=True)
def _patch_model_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    # monkeypatch.setattr fully replaces (and auto-restores) these module
    # attributes for the duration of the test -- no lru_cache to clear
    # since the real cached loaders are never called at all here.
    monkeypatch.setattr(rag, "_get_embedding_predictor", lambda: FakeEmbedder())
    monkeypatch.setattr(rag, "_get_vector_store", lambda: FakeStore({}))
    monkeypatch.setattr(rag, "_get_reranker", lambda: FakeReranker({}))


def test_unknown_ticket_returns_404(db_session: Session, client: TestClient) -> None:
    response = client.post(f"/tickets/{uuid.uuid4()}/suggested-reply")
    assert response.status_code == 404


def test_search_disabled_returns_503_before_looking_up_the_ticket(
    db_session: Session, client: TestClient
) -> None:
    # Unknown ticket id and no fixture-created ticket -- if this reached
    # the DB lookup it'd 404, not 503. Proves the SEARCH_ENABLED=false
    # gate runs first, same guarantee test_search_endpoint.py's disabled
    # test proves for /search.
    disabled_settings = Settings(_env_file=None, search_enabled=False)  # type: ignore[call-arg]
    app.dependency_overrides[get_settings] = lambda: disabled_settings
    try:
        response = client.post(f"/tickets/{uuid.uuid4()}/suggested-reply")
    finally:
        app.dependency_overrides.pop(get_settings, None)

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Suggested replies are not available on this deployment (SEARCH_ENABLED=false)."
    )


def test_ticket_with_no_customer_message_returns_422(
    db_session: Session, client: TestClient
) -> None:
    ticket = _agent_only_ticket(db_session)

    response = client.post(f"/tickets/{ticket.id}/suggested-reply")

    assert response.status_code == 422


def test_low_confidence_refuses_without_calling_the_llm(
    db_session: Session, client: TestClient
) -> None:
    ticket = _ticket_with_customer_message(db_session)
    # store/reranker stay the default empty fakes from _patch_model_loaders
    # -- no candidates at all, so this can never clear the confidence gate.

    response = client.post(f"/tickets/{ticket.id}/suggested-reply")

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["draft"] is None
    assert body["sources"] == []


def test_confident_match_returns_a_draft_with_sources(
    db_session: Session, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticket = _ticket_with_customer_message(db_session)
    hit = VectorHit(
        id="a1",
        document="How to Track Your Order",
        metadata={"title": "How to Track Your Order"},
        similarity=0.9,
    )
    monkeypatch.setattr(rag, "_get_vector_store", lambda: FakeStore({"kb_articles": [hit]}))
    monkeypatch.setattr(rag, "_get_reranker", lambda: FakeReranker({hit.document: 5.0}))

    openai_client = MagicMock()
    openai_client.chat.completions.create.return_value = _mock_openai_response(
        "Track it here, see [1]."
    )
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, llm_enabled=True, llm_budget_usd=5.0, openai_api_key="test-key"
    )
    monkeypatch.setattr(
        rag,
        "_build_llm_client",
        lambda db, _settings: LLMClient(db, settings, openai_client),
    )

    response = client.post(f"/tickets/{ticket.id}/suggested-reply")

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is False
    assert body["draft"] == "Track it here, see [1]."
    assert body["cited_indices"] == [1]
    assert len(body["sources"]) == 1
    assert body["sources"][0]["title"] == "How to Track Your Order"


def test_llm_disabled_returns_503(
    db_session: Session, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticket = _ticket_with_customer_message(db_session)
    hit = VectorHit(id="a1", document="track it", metadata={"title": "Tracking"}, similarity=0.9)
    monkeypatch.setattr(rag, "_get_vector_store", lambda: FakeStore({"kb_articles": [hit]}))
    monkeypatch.setattr(rag, "_get_reranker", lambda: FakeReranker({hit.document: 5.0}))

    disabled_settings = Settings(_env_file=None, llm_enabled=False)  # type: ignore[call-arg]
    monkeypatch.setattr(
        rag,
        "_build_llm_client",
        lambda db, _settings: LLMClient(db, disabled_settings, MagicMock()),
    )

    response = client.post(f"/tickets/{ticket.id}/suggested-reply")

    assert response.status_code == 503
