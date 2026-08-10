from collections.abc import Iterator
from typing import Protocol

from api.schemas.ticket import CanonicalTicket


class TicketLoader(Protocol):
    """Loader -> DB contract. Loaders never touch the database themselves;
    ml.data.persist consumes whatever they yield."""

    def iter_tickets(self) -> Iterator[CanonicalTicket]: ...
