import hashlib
from collections.abc import Iterable, Iterator
from typing import Protocol, TypeVar

from ml.data.cleaning import normalize_for_hash


def content_hash(text: str) -> str:
    return hashlib.sha1(normalize_for_hash(text).encode("utf-8"), usedforsecurity=False).hexdigest()


class _HasText(Protocol):
    text_clean: str


T = TypeVar("T", bound=_HasText)


def dedup_messages(messages: Iterable[T]) -> Iterator[T]:
    """First-wins dedup on exact/normalization-equivalent duplicate text."""
    seen: set[str] = set()
    for message in messages:
        h = content_hash(message.text_clean)
        if h in seen:
            continue
        seen.add(h)
        yield message
