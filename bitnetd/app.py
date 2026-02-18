from __future__ import annotations

import asyncio
import json
import threading
from contextlib import suppress
from datetime import datetime, timezone
from time import monotonic
from typing import Any
from uuid import UUID, uuid4

import torch
from fastapi import BackgroundTasks, Depends, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from shared.constants import ensure_dirs

from .llm import BitNetModelService, build_streamer, seed_torch
from .model_store import DEFAULT_MODEL_ID
from .security import get_or_create_token, require_token
from .state import AllowedAppName, ServerState

PRUNE_INTERVAL_SECONDS = 1.0
CLIENT_TTL_SECONDS = 15

DEFAULT_GENERATE_OPTIONS = {
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.95,
    "seed": None,
    "repeat_penalty": 1.1,
    "stop": [],
    "timeout_ms": 60_000,
}

app = FastAPI(title="bitnetd", version="0.2.0-phase8")
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


class GenerateRequest(BaseModel):
    prompt: str
    stream: bool = True
    max_tokens: int = Field(default=DEFAULT_GENERATE_OPTIONS["max_tokens"], ge=1)
    temperature: float = Field(default=DEFAULT_GENERATE_OPTIONS["temperature"], ge=0)
    top_p: float = Field(default=DEFAULT_GENERATE_OPTIONS["top_p"], gt=0, le=1)
    seed: int | None = DEFAULT_GENERATE_OPTIONS["seed"]
    repeat_penalty: float = Field(default=DEFAULT_GENERATE_OPTIONS["repeat_penalty"], gt=0)
    stop: list[str] = Field(default_factory=list)
    timeout_ms: int | None = Field(default=DEFAULT_GENERATE_OPTIONS["timeout_ms"], ge=1)


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


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global _prune_task

    if _prune_task is not None:
        _prune_task.cancel()
        with suppress(asyncio.CancelledError):
            await _prune_task
        _prune_task = None


def _find_stop_position(text: str, stop: list[str]) -> int | None:
    best: int | None = None
    for marker in stop:
        if not marker:
            continue
        idx = text.find(marker)
        if idx >= 0 and (best is None or idx < best):
            best = idx
    return best


def _build_done_meta(
    *,
    req: GenerateRequest,
    loaded_model_id: str,
    elapsed_ms: int,
    text: str,
    stop_reason: str,
) -> dict[str, Any]:
    tokens_out = 0
    if text:
        tokens_out = len(model_service.loaded.tokenizer.encode(text, add_special_tokens=False))

    params_applied = {
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p if req.temperature > 0 else None,
        "seed": req.seed,
        "repeat_penalty": req.repeat_penalty,
        "stop": req.stop,
        "timeout_ms": req.timeout_ms,
    }
    return {
        "model": loaded_model_id,
        "elapsed_ms": elapsed_ms,
        "tokens_out": tokens_out,
        "stop_reason": stop_reason,
        "params_applied": params_applied,
    }


async def _mark_generation_start() -> None:
    async with state.lock:
        state.active_generations += 1


async def _mark_generation_end() -> None:
    should_exit = False
    async with state.lock:
        state.active_generations = max(0, state.active_generations - 1)
        should_exit = state.active_generations == 0 and state.exit_pending and len(state.clients) == 0
    if should_exit:
        state.exit_now()


@app.get("/health")
async def health() -> dict:
    if model_service.is_loaded:
        state.status = "ready"
        state.reasons = []
    elif state.status == "error":
        pass
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


@app.post("/generate")
async def generate(
    payload: GenerateRequest,
    _: str = Depends(require_token),
):
    await _mark_generation_start()

    try:
        loaded = await asyncio.to_thread(model_service.load_if_needed, DEFAULT_MODEL_ID, "main")
        state.status = "ready"
        state.reasons = []

        seed_torch(payload.seed)
        started = monotonic()
        request_id = str(uuid4())

        tokenizer = loaded.tokenizer
        model = loaded.model
        inputs = tokenizer(payload.prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(loaded.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(loaded.device)

        generate_kwargs = model_service.prepare_generation_kwargs(
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
            repeat_penalty=payload.repeat_penalty,
            timeout_ms=payload.timeout_ms,
        )

        if payload.stream:

            async def event_generator():
                full_text = ""
                stop_reason = "length"
                timeout_seconds = (payload.timeout_ms / 1000.0) if payload.timeout_ms else None
                meta_event = {
                    "model": loaded.model_id,
                    "params_applied": {
                        "max_tokens": payload.max_tokens,
                        "temperature": payload.temperature,
                        "top_p": payload.top_p if payload.temperature > 0 else None,
                        "seed": payload.seed,
                        "repeat_penalty": payload.repeat_penalty,
                        "stop": payload.stop,
                        "timeout_ms": payload.timeout_ms,
                    },
                    "request_id": request_id,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                yield {"event": "meta", "data": json.dumps(meta_event, ensure_ascii=False)}

                streamer = build_streamer(tokenizer)
                worker_exception: list[Exception] = []

                def _run_generation() -> None:
                    try:
                        with torch.inference_mode():
                            model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                streamer=streamer,
                                **generate_kwargs,
                            )
                    except Exception as exc:  # pragma: no cover
                        worker_exception.append(exc)

                thread = threading.Thread(target=_run_generation, daemon=True)
                thread.start()

                timed_out = False
                try:
                    while True:
                        if timeout_seconds and (monotonic() - started) > timeout_seconds:
                            timed_out = True
                            stop_reason = "timeout"
                            break

                        chunk = await asyncio.to_thread(next, streamer, None)
                        if chunk is None:
                            break

                        full_text += chunk
                        cut_pos = _find_stop_position(full_text, payload.stop)
                        if cut_pos is not None:
                            full_text = full_text[:cut_pos]
                            stop_reason = "stop"
                            break

                        yield {
                            "event": "delta",
                            "data": json.dumps({"delta": chunk}, ensure_ascii=False),
                        }

                    thread.join(timeout=0.1)

                    if worker_exception:
                        raise worker_exception[0]

                    if timed_out:
                        pass
                    elif stop_reason != "stop":
                        generated_tokens = len(
                            tokenizer.encode(full_text, add_special_tokens=False)
                        )
                        if generated_tokens >= payload.max_tokens:
                            stop_reason = "length"

                    elapsed_ms = int((monotonic() - started) * 1000)
                    done_meta = _build_done_meta(
                        req=payload,
                        loaded_model_id=loaded.model_id,
                        elapsed_ms=elapsed_ms,
                        text=full_text,
                        stop_reason=stop_reason,
                    )
                    yield {
                        "event": "done",
                        "data": json.dumps(
                            {
                                "text": full_text,
                                "meta": done_meta,
                            },
                            ensure_ascii=False,
                        ),
                    }
                except Exception as exc:
                    elapsed_ms = int((monotonic() - started) * 1000)
                    error_meta = _build_done_meta(
                        req=payload,
                        loaded_model_id=loaded.model_id,
                        elapsed_ms=elapsed_ms,
                        text=full_text,
                        stop_reason="error",
                    )
                    yield {
                        "event": "error",
                        "data": json.dumps(
                            {"message": str(exc), "meta": error_meta},
                            ensure_ascii=False,
                        ),
                    }

            return EventSourceResponse(event_generator())

        def _run_non_stream() -> tuple[str, str]:
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generate_kwargs,
                )

            generated_ids = outputs[0][input_ids.shape[1] :]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            stop_reason = "length"
            cut_pos = _find_stop_position(text, payload.stop)
            if cut_pos is not None:
                text = text[:cut_pos]
                stop_reason = "stop"
            elif payload.timeout_ms and (monotonic() - started) * 1000 >= payload.timeout_ms:
                stop_reason = "timeout"
            return text, stop_reason

        text, stop_reason = await asyncio.to_thread(_run_non_stream)
        elapsed_ms = int((monotonic() - started) * 1000)
        meta = _build_done_meta(
            req=payload,
            loaded_model_id=loaded.model_id,
            elapsed_ms=elapsed_ms,
            text=text,
            stop_reason=stop_reason,
        )
        return JSONResponse(content={"text": text, "meta": meta})
    except Exception as exc:
        state.status = "error"
        state.reasons = ["model_load_failed"]
        return JSONResponse(
            status_code=500,
            content={
                "text": "",
                "meta": {
                    "model": DEFAULT_MODEL_ID,
                    "elapsed_ms": 0,
                    "tokens_out": 0,
                    "stop_reason": "error",
                    "params_applied": {
                        "max_tokens": payload.max_tokens,
                        "temperature": payload.temperature,
                        "top_p": payload.top_p if payload.temperature > 0 else None,
                        "seed": payload.seed,
                        "repeat_penalty": payload.repeat_penalty,
                        "stop": payload.stop,
                        "timeout_ms": payload.timeout_ms,
                    },
                },
                "error": str(exc),
            },
        )
    finally:
        await _mark_generation_end()
