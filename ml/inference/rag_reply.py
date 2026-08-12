"""RAG suggested-reply drafting (SPEC M8: "retrieve top-k similar resolved
cases + KB articles -> OpenAI drafts a reply with citations to the
retrieved sources"). Built on ml/inference/retrieval.py's shared pipeline --
always reranked here regardless of any /search UI toggle, since the
confidence gate below is calibrated against cross-encoder scores
specifically, not raw cosine similarity (see docs/decisions.md for the
real-corpus measurement that motivated this).

Confidence gate: refuses to call the LLM at all when the best retrieved
source scores below MIN_CONFIDENCE -- SPEC M8's "no-answer behavior",
covered by tests/integration/test_rag_no_answer.py's injected
out-of-domain query. This doubles as a budget guard: a query with nothing
relevant indexed would otherwise still burn a paid call on a reply nobody
should trust.
"""

import re
from dataclasses import dataclass

from ml.inference.base import EmbeddingPredictor
from ml.inference.llm_client import LLMClient
from ml.inference.reranker import Reranker
from ml.inference.retrieval import RankedHit, retrieve
from ml.inference.vector_store import ChromaVectorStore

PURPOSE = "rag_reply_draft"
POOL_SIZE = 8
MAX_SOURCES = 4
# Cross-encoder (ms-marco-MiniLM) raw logit score, not a probability -- 0 is
# the natural "more relevant than not" cutoff for this checkpoint. Measured
# against the real indexed corpus (docs/decisions.md): 5 realistic support
# queries scored [0.93, 9.02] on their best source; 5 clearly off-topic
# queries (trivia, small talk, gibberish) scored [-11.17, -3.33] -- a wide,
# clean gap either side of 0.
MIN_CONFIDENCE = 0.0

_SYSTEM_PROMPT = (
    "You are drafting a suggested reply for a customer support agent to send to a customer. "
    "You will be given the customer's issue and a numbered list of sources: similar past "
    "cases that were resolved, and knowledge-base articles. Write a concise, professional "
    "reply that resolves the customer's issue, using ONLY information from the sources. "
    "Cite every claim you make with the matching [N] marker. If the sources don't actually "
    "cover the customer's issue, say so plainly instead of guessing."
)


@dataclass(frozen=True)
class RagSource:
    index: int  # 1-based, matches the [N] citation markers in the prompt/draft
    kind: str  # "ticket" | "kb_article"
    id: str
    title: str | None
    text: str
    score: float


@dataclass(frozen=True)
class RagDraftResult:
    refused: bool
    refusal_reason: str | None
    draft: str | None
    sources: list[RagSource]
    cited_indices: list[int]
    cached: bool
    cost_usd: float


def _source_text(ranked: RankedHit) -> str:
    if ranked.source == "ticket":
        # The full thread (including the agent's resolution) rides in
        # metadata -- the embedded/matched document is customer-problem
        # text only (ml/inference/vector_store.py's module docstring).
        return str(ranked.hit.metadata.get("thread_text", ranked.hit.document))
    return ranked.hit.document


def to_rag_sources(ranked: list[RankedHit]) -> list[RagSource]:
    sources = []
    for i, r in enumerate(ranked[:MAX_SOURCES], start=1):
        title = r.hit.metadata.get("title") if r.source == "kb_article" else None
        sources.append(
            RagSource(
                index=i,
                kind=r.source,
                id=r.hit.id,
                title=str(title) if title is not None else None,
                text=_source_text(r),
                score=r.score,
            )
        )
    return sources


def build_prompt(customer_issue: str, sources: list[RagSource]) -> str:
    lines = [f"Customer's issue:\n{customer_issue}\n", "Sources:"]
    for s in sources:
        label = s.title if s.title else f"Similar resolved case ({s.id})"
        lines.append(f"[{s.index}] {label}\n{s.text}\n")
    lines.append("Write the reply now, citing sources as [1], [2], etc.")
    return "\n".join(lines)


def extract_cited_indices(draft: str) -> list[int]:
    return sorted({int(m) for m in re.findall(r"\[(\d+)\]", draft)})


def draft_reply(
    llm_client: LLMClient,
    *,
    customer_issue: str,
    embedder: EmbeddingPredictor,
    store: ChromaVectorStore,
    reranker: Reranker,
    max_tokens: int,
    exclude_ticket_id: str | None = None,
) -> RagDraftResult:
    exclude_ids = frozenset({exclude_ticket_id}) if exclude_ticket_id else frozenset()
    ranked = retrieve(
        customer_issue,
        embedder=embedder,
        store=store,
        reranker=reranker,
        pool_size=POOL_SIZE,
        exclude_ids=exclude_ids,
    )
    if not ranked or ranked[0].score < MIN_CONFIDENCE:
        return RagDraftResult(
            refused=True,
            refusal_reason=(
                "No sufficiently similar resolved case or KB article was found for this issue."
            ),
            draft=None,
            sources=[],
            cited_indices=[],
            cached=False,
            cost_usd=0.0,
        )

    sources = to_rag_sources(ranked)
    prompt = build_prompt(customer_issue, sources)
    result = llm_client.complete(
        purpose=PURPOSE, prompt=prompt, system=_SYSTEM_PROMPT, max_tokens=max_tokens
    )
    return RagDraftResult(
        refused=False,
        refusal_reason=None,
        draft=result.response,
        sources=sources,
        cited_indices=extract_cited_indices(result.response),
        cached=result.cached,
        cost_usd=result.cost_usd,
    )
