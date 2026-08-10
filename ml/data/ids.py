import uuid

NAMESPACE = uuid.UUID("6f6a6d1e-6b0a-4b1a-9c1a-2f6b8d5c4a10")


def deterministic_id(*parts: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE, ":".join(parts))
