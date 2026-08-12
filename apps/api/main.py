from fastapi import FastAPI

from api.routers import health, predict, rag, search, tickets, topics
from api.version import __version__

app = FastAPI(title="supportlens", version=__version__)
app.include_router(health.router)
app.include_router(tickets.router)
app.include_router(predict.router)
app.include_router(topics.router)
app.include_router(search.router)
app.include_router(rag.router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "hello from supportlens", "version": __version__}
