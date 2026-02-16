from __future__ import annotations

from fastapi import Depends, FastAPI

from shared.constants import ensure_dirs

from .security import get_or_create_token, require_token
from .state import ServerState

app = FastAPI(title="bitnetd", version="0.1.0-phase1-1")
state = ServerState()


@app.on_event("startup")
def on_startup() -> None:
    ensure_dirs()
    get_or_create_token()


@app.get("/health")
def health() -> dict:
    return state.health().model_dump()


@app.api_route("/clients/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
def clients_placeholder(path: str, _: str = Depends(require_token)) -> dict:
    return {"detail": "not_implemented", "path": path}


@app.api_route("/generate", methods=["POST"])
def generate_placeholder(_: str = Depends(require_token)) -> dict:
    return {"detail": "not_implemented"}
