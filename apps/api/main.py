from fastapi import FastAPI

from api.routers import health, tickets
from api.version import __version__

app = FastAPI(title="supportlens", version=__version__)
app.include_router(health.router)
app.include_router(tickets.router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "hello from supportlens", "version": __version__}
