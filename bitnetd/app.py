from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from uuid import UUID

from fastapi import BackgroundTasks, Depends, FastAPI
from pydantic import BaseModel

from shared.constants import ensure_dirs

from .llm import BitNetModelService
from .model_store import DEFAULT_MODEL_ID
from .security import get_or_create_token, require_token
from .state import AllowedAppName, ServerState

logger = logging.getLogger(__name__)

PRUNE_INTERVAL_SECONDS = 1.0
CLIENT_TTL_SECONDS = 15

app = FastAPI(title="bitnetd", version="0.2.0-phase8.1")
state = ServerState()
model_service = BitNetModelService()
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
        removed_count, before_count, after_count = await state.prune_expired_clients(
            ttl_seconds=CLIENT_TTL_SECONDS
        )
        if removed_count > 0 and state.should_exit_on_transition_to_zero(before_count, after_count):
            state.exit_now()


@app.on_event("startup")
async def on_startup() -> None:
    global _prune_task

    ensure_dirs()
    get_or_create_token()
    state.status = "starting"
    state.reasons = ["model_not_loaded"]
    _prune_task = asyncio.create_task(_prune_loop())

    try:
        await asyncio.to_thread(model_service.load_if_needed, DEFAULT_MODEL_ID, "main")
        state.status = "ready"
        state.reasons = []
    except Exception as exc:
        logger.error("BitNet model load failed at startup: %s", exc)
        state.status = "error"
        state.reasons = ["model_load_failed"]


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
    if model_service.is_loaded:
        state.status = "ready"
        state.reasons = []
    elif state.status == "error":
        if "model_load_failed" not in state.reasons:
            state.reasons = ["model_load_failed"]
    else:
        state.status = "starting"
        state.reasons = ["model_not_loaded"]
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
    ok, before_count, after_count, error = await state.unregister_client(
        client_id=str(payload.client_id)
    )

    if state.should_exit_on_transition_to_zero(before_count, after_count):
        background_tasks.add_task(state.exit_now)

    return ClientResponse(ok=ok, client_count=after_count, error=error)


@app.api_route("/generate", methods=["POST"])
def generate_placeholder(_: str = Depends(require_token)) -> dict:
    return {"detail": "not_implemented"}
