from collections.abc import Iterator
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from api.db.models import AuthorRole, TicketSource
from api.schemas.ticket import CanonicalMessage, CanonicalTicket

from ml.data.cleaning import clean_text
from ml.data.dedup import content_hash
from ml.data.ids import deterministic_id
from ml.data.language import detect

TWCS_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"

_STRUCTURE_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "response_tweet_id",
    "in_response_to_tweet_id",
]
_FULL_COLUMNS = [*_STRUCTURE_COLUMNS, "text"]
_MIN_DT = datetime.min.replace(tzinfo=UTC)


def parse_created_at(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return datetime.strptime(raw.strip(), TWCS_DATE_FORMAT)
    except ValueError:
        return None


def _parse_response_ids(raw: Any) -> list[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _is_inbound(raw: Any) -> bool:
    return str(raw).strip().lower() == "true"


class _UnionFind:
    """Path-compressed union-find over tweet ids. Reply edges (including a tweet
    replying to itself, or to an id absent from the file) merge branching reply
    chains into single conversations."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        self._parent.setdefault(a, a)
        self._parent.setdefault(b, b)
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_a] = root_b


@dataclass
class ConversationSummary:
    root_id: str
    brand: str | None
    start_time: datetime | None
    message_count: int
    tweet_ids: frozenset[str]


def _read_structure_chunks(csv_path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(csv_path, usecols=_STRUCTURE_COLUMNS, chunksize=chunksize, dtype=str)


def build_conversations(csv_path: Path, chunksize: int = 200_000) -> dict[str, ConversationSummary]:
    """Pass A: group tweets into conversations via reply-chain union-find. Only
    structure columns are read — message text never touches memory here."""
    union_find = _UnionFind()
    rows: dict[str, dict[str, Any]] = {}

    for chunk in _read_structure_chunks(csv_path, chunksize):
        for row in chunk.itertuples(index=False):
            tweet_id = str(row.tweet_id)
            union_find.find(tweet_id)
            rows[tweet_id] = {
                "author_id": row.author_id,
                "inbound": _is_inbound(row.inbound),
                "created_at": parse_created_at(row.created_at),
            }
            in_reply_to = row.in_response_to_tweet_id
            if isinstance(in_reply_to, str) and in_reply_to.strip():
                union_find.union(tweet_id, in_reply_to.strip())
            for reply_id in _parse_response_ids(row.response_tweet_id):
                union_find.union(tweet_id, reply_id)

    members: dict[str, list[str]] = {}
    for tweet_id in rows:
        root = union_find.find(tweet_id)
        members.setdefault(root, []).append(tweet_id)

    summaries: dict[str, ConversationSummary] = {}
    for root, tweet_ids in members.items():
        brand = next(
            (rows[tid]["author_id"] for tid in tweet_ids if not rows[tid]["inbound"]), None
        )
        start_times = [
            rows[tid]["created_at"] for tid in tweet_ids if rows[tid]["created_at"] is not None
        ]
        summaries[root] = ConversationSummary(
            root_id=root,
            brand=brand,
            start_time=min(start_times) if start_times else None,
            message_count=len(tweet_ids),
            tweet_ids=frozenset(tweet_ids),
        )
    return summaries


def _sort_key(message: dict[str, Any]) -> tuple[datetime, str]:
    return (message["created_at"] or _MIN_DT, message["tweet_id"])


def _order_conversation(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reply-chain order (parent before child) with timestamp breaking ties among
    concurrent branches; any message unreachable from a root (cycles, malformed
    refs) is appended by timestamp so nothing is silently dropped."""
    by_id = {m["tweet_id"]: m for m in messages}
    children: dict[str, list[str]] = {}
    for m in messages:
        parent = m["in_response_to"]
        if parent and parent in by_id and parent != m["tweet_id"]:
            children.setdefault(parent, []).append(m["tweet_id"])

    child_ids = {c for kids in children.values() for c in kids}
    queue = sorted((m for m in messages if m["tweet_id"] not in child_ids), key=_sort_key)

    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    while queue:
        queue.sort(key=_sort_key)
        current = queue.pop(0)
        if current["tweet_id"] in seen:
            continue
        seen.add(current["tweet_id"])
        ordered.append(current)
        for child_id in children.get(current["tweet_id"], []):
            if child_id not in seen:
                queue.append(by_id[child_id])

    for m in sorted(messages, key=_sort_key):
        if m["tweet_id"] not in seen:
            ordered.append(m)
            seen.add(m["tweet_id"])

    return ordered


def _build_ticket(summary: ConversationSummary, ordered: list[dict[str, Any]]) -> CanonicalTicket:
    ticket_id = deterministic_id(TicketSource.TWITTER, summary.root_id)
    messages: list[CanonicalMessage] = []
    for seq, m in enumerate(ordered):
        raw_text = m["text"] or ""
        cleaned = clean_text(raw_text)
        lang_result = detect(cleaned)
        messages.append(
            CanonicalMessage(
                id=deterministic_id(TicketSource.TWITTER, m["tweet_id"]),
                seq=seq,
                author_role=AuthorRole.CUSTOMER if m["inbound"] else AuthorRole.AGENT,
                text_raw=raw_text,
                text_clean=cleaned,
                sent_at=m["created_at"],
                lang=lang_result.lang,
                lang_confidence=lang_result.confidence,
                content_hash=content_hash(cleaned),
                external_id=m["tweet_id"],
            )
        )

    return CanonicalTicket(
        id=ticket_id,
        source=TicketSource.TWITTER,
        external_id=summary.root_id,
        created_at=summary.start_time,
        channel="twitter",
        customer_id=None,
        brand=summary.brand,
        lang=messages[0].lang if messages else None,
        meta={"message_count": summary.message_count},
        messages=messages,
    )


def iter_tickets(
    csv_path: Path,
    *,
    selected_roots: AbstractSet[str] | None = None,
    chunksize: int = 200_000,
) -> Iterator[CanonicalTicket]:
    """Pass B: materialize CanonicalTickets for the given conversation roots (all
    conversations if selected_roots is None — used for small/full-file loads)."""
    conversations = build_conversations(csv_path, chunksize=chunksize)
    if selected_roots is not None:
        conversations = {
            root: summary for root, summary in conversations.items() if root in selected_roots
        }

    root_by_tweet: dict[str, str] = {
        tweet_id: root for root, summary in conversations.items() for tweet_id in summary.tweet_ids
    }
    buffers: dict[str, list[dict[str, Any]]] = {root: [] for root in conversations}

    for chunk in pd.read_csv(csv_path, usecols=_FULL_COLUMNS, chunksize=chunksize, dtype=str):
        for row in chunk.itertuples(index=False):
            tweet_id = str(row.tweet_id)
            root = root_by_tweet.get(tweet_id)
            if root is None:
                continue
            in_reply_to = row.in_response_to_tweet_id
            buffers[root].append(
                {
                    "tweet_id": tweet_id,
                    "inbound": _is_inbound(row.inbound),
                    "created_at": parse_created_at(row.created_at),
                    "in_response_to": in_reply_to.strip() if isinstance(in_reply_to, str) else None,
                    "text": row.text,
                }
            )

    for root, summary in conversations.items():
        buffered = buffers[root]
        if not buffered:
            continue
        yield _build_ticket(summary, _order_conversation(buffered))
