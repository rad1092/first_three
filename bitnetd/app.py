from __future__ import annotations

import asyncio
from contextlib import suppress
from uuid import UUID

from fastapi import BackgroundTasks, Depends, FastAPI
from pydantic import BaseModel

from shared.constants import ensure_dirs

from .security import get_or_create_token, require_token
from .state import AllowedAppName, ServerState

PRUNE_INTERVAL_SECONDS = 1.0
CLIENT_TTL_SECONDS = 15

app = FastAPI(title="bitnetd", version="0.2.0-phase1-2")
state = ServerState()
_prune_task: asyncio.Task[None] | None = None


class RegisterClientRequest(BaseModel):
    client_id: UUID
    app_name: AllowedAppName


class ClientHeartbeatRequest(BaseModel):
    client_id: UUID


class ClientResponse(BaseModel):
    ok: bool
    client_count: int
    error: str | None = None


async def _prune_loop() -> None:
    while True:
        await asyncio.sleep(PRUNE_INTERVAL_SECONDS)
        _, client_count = await state.prune_expired_clients(ttl_seconds=CLIENT_TTL_SECONDS)
        if client_count == 0 and await state.should_exit_now():
            state.exit_now()


@app.on_event("startup")
async def on_startup() -> None:
    global _prune_task

    ensure_dirs()
    get_or_create_token()
    _prune_task = asyncio.create_task(_prune_loop())


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global _prune_task

    if _prune_task is not None:
        _prune_task.cancel()
        with suppress(asyncio.CancelledError):
            await _prune_task
        _prune_task = None


@app.get("/health")
async def health() -> dict:
    return (await state.health()).model_dump()


@app.post("/clients/register", response_model=ClientResponse)
async def register_client(
    payload: RegisterClientRequest,
    _: str = Depends(require_token),
) -> ClientResponse:
    client_count = await state.register_client(
        client_id=str(payload.client_id),
        app_name=payload.app_name,
    )
    return ClientResponse(ok=True, client_count=client_count)


@app.post("/clients/heartbeat", response_model=ClientResponse)
async def heartbeat_client(
    payload: ClientHeartbeatRequest,
    _: str = Depends(require_token),
) -> ClientResponse:
    ok, client_count, error = await state.heartbeat_client(client_id=str(payload.client_id))
    return ClientResponse(ok=ok, client_count=client_count, error=error)


@app.post("/clients/unregister", response_model=ClientResponse)
async def unregister_client(
    payload: ClientHeartbeatRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(require_token),
) -> ClientResponse:
    ok, client_count, error = await state.unregister_client(client_id=str(payload.client_id))

    if client_count == 0 and await state.should_exit_now():
        background_tasks.add_task(state.exit_now)

    return ClientResponse(ok=ok, client_count=client_count, error=error)


@app.api_route("/generate", methods=["POST"])
def generate_placeholder(_: str = Depends(require_token)) -> dict:
    return {"detail": "not_implemented"}
