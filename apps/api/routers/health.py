from fastapi import APIRouter
from sqlalchemy import text

from api.deps import DbDep
from api.version import __version__

router = APIRouter()


@router.get("/healthz")
def healthz(db: DbDep) -> dict[str, str]:
    db.execute(text("SELECT 1"))
    return {"status": "ok", "db": "ok", "version": __version__}
