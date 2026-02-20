from __future__ import annotations

import asyncio
import json
import logging
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

from shared.constants import bitnet_home, ensure_dirs

from .config import load_config
from .engines.base import EngineError, GenerateParams, VocabMismatchError
from .engines.factory import create_engine
from .engines.torch_engine import TorchEngine, is_cuda_runtime_error, summarize_exc
from .llm import CompilerMissingError, seed_torch
from .security import get_or_create_token, require_token
from .state import AllowedAppName, ServerState

logger = logging.getLogger(__name__)

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

app = FastAPI(title="bitnetd", version="0.2.0-phase8.2")
state = ServerState()
engine_config = load_config()
engine = create_engine(engine_config)
_prune_task: asyncio.Task[None] | None = None
_gpu_disabled_reason: str | None = None
_gpu_disabled_detail: str | None = None


def _gpu_is_disabled() -> bool:
    return _gpu_disabled_reason is not None


def _disable_gpu(reason: str, detail: str | None = None) -> None:
    global _gpu_disabled_reason, _gpu_disabled_detail
    _gpu_disabled_reason = reason
    _gpu_disabled_detail = detail


def _resolve_snapshot_path() -> str:
    manifest = bitnet_home() / "config" / "manifest.json"
    if manifest.exists():
        with suppress(Exception):
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            snapshot_path = str(raw.get("snapshot_path", "")).strip()
            if snapshot_path:
                return snapshot_path
    return str(bitnet_home() / "models")


def _engine_model_label() -> str:
    with suppress(Exception):
        return str(engine.model_label())
    return "unknown"


def _tokens_out_for_text(text: str) -> int:
    if not text:
        return 0
    if isinstance(engine, TorchEngine):
        with suppress(Exception):
            return len(engine.loaded.tokenizer.encode(text, add_special_tokens=False))
    return 0


def _output_progress_units(text: str) -> int:
    if isinstance(engine, TorchEngine):
        return _tokens_out_for_text(text)
    return len(text)


def _build_meta(
    *,
    req: "GenerateRequest",
    model: str,
    elapsed_ms: int,
    text: str,
    stop_reason: str,
    tokens_out: int | None = None,
) -> dict[str, Any]:
    output_tokens = int(tokens_out) if tokens_out is not None else 0
    return {
        "model": model,
        "elapsed_ms": elapsed_ms,
        "tokens_out": output_tokens,
        "stop_reason": stop_reason,
        "params_applied": {
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p if req.temperature > 0 else None,
            "seed": req.seed,
            "repeat_penalty": req.repeat_penalty,
            "stop": req.stop,
            "timeout_ms": req.timeout_ms,
        },
    }


def _gpu_disabled_response(payload: "GenerateRequest") -> JSONResponse:
    meta = _build_meta(
        req=payload,
        model=_engine_model_label(),
        elapsed_ms=0,
        text="",
        stop_reason="error",
        tokens_out=0,
    )
    return JSONResponse(
        status_code=503,
        content={
            "text": "",
            "meta": meta,
            "error": "GPU가 비활성화된 상태입니다. bitnetd 재시작이 필요합니다.",
            "detail": _gpu_disabled_detail or _gpu_disabled_reason,
        },
    )


def _engine_error_response(payload: "GenerateRequest", exc: EngineError) -> JSONResponse:
    meta = _build_meta(
        req=payload,
        model=_engine_model_label(),
        elapsed_ms=0,
        text="",
        stop_reason="error",
        tokens_out=0,
    )
    content: dict[str, Any] = {"text": "", "meta": meta, "error": exc.error}
    if exc.detail:
        content["detail"] = exc.detail
    return JSONResponse(status_code=exc.status_code, content=content)


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


def _sync_state_from_engine() -> None:
    ready, reasons = engine.is_ready()
    state.engine = getattr(engine, "engine_id", "unknown")
    state.model = _engine_model_label()
    if ready:
        state.status = "ready"
        state.reasons = []
    else:
        state.status = "starting" if "model_not_loaded" in reasons else "not_ready"
        state.reasons = reasons or ["model_not_loaded"]

    if _gpu_is_disabled():
        if "gpu_disabled" not in state.reasons:
            state.reasons.append("gpu_disabled")
        if _gpu_disabled_reason and _gpu_disabled_reason not in state.reasons:
            state.reasons.append(_gpu_disabled_reason)


@app.on_event("startup")
async def on_startup() -> None:
    global _prune_task

    ensure_dirs()
    get_or_create_token()
    state.status = "starting"
    state.reasons = ["model_not_loaded"]
    _prune_task = asyncio.create_task(_prune_loop())

    try:
        snapshot_path = _resolve_snapshot_path()
        await asyncio.to_thread(engine.ensure_loaded, snapshot_path)
        _sync_state_from_engine()
    except Exception as exc:  # noqa: BLE001
        logger.error("Engine load failed at startup: %s", exc)
        state.status = "error"
        reasons = ["model_load_failed"]
        if (
            isinstance(exc, CompilerMissingError)
            or "cl.exe" in str(exc).lower()
            or "cl is not found" in str(exc).lower()
        ):
            reasons.append("compiler_missing")
        state.reasons = reasons
        state.engine = getattr(engine, "engine_id", "unknown")
        state.model = _engine_model_label()


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
    _sync_state_from_engine()
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
async def generate(payload: GenerateRequest, _: str = Depends(require_token)):
    await _mark_generation_start()
    stream_manages_finally = False

    if _gpu_is_disabled():
        if payload.stream:
            stream_manages_finally = True

            async def _guard_stream():
                try:
                    yield {
                        "event": "error",
                        "data": json.dumps(
                            {
                                "message": "GPU가 비활성화된 상태입니다. bitnetd 재시작이 필요합니다.",
                                "detail": _gpu_disabled_detail or _gpu_disabled_reason,
                            },
                            ensure_ascii=False,
                        ),
                    }
                finally:
                    await _mark_generation_end()

            return EventSourceResponse(_guard_stream(), status_code=503)
        return _gpu_disabled_response(payload)

    try:
        snapshot_path = _resolve_snapshot_path()
        await asyncio.to_thread(engine.ensure_loaded, snapshot_path)
        _sync_state_from_engine()

        params = GenerateParams(
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
            repeat_penalty=payload.repeat_penalty,
            stop=payload.stop,
            timeout_ms=payload.timeout_ms,
            seed=payload.seed,
        )

        if isinstance(engine, TorchEngine):
            use_cuda = engine.loaded.device == "cuda" and not _gpu_is_disabled()
            seed_torch(payload.seed, use_cuda=use_cuda)

        started = monotonic()
        request_id = str(uuid4())

        if payload.stream:
            stream_manages_finally = True
            try:
                chunk_iterator = await asyncio.to_thread(engine.generate_stream, payload.prompt, params=params)
            except VocabMismatchError as exc:
                return _engine_error_response(payload, exc)
            except EngineError as exc:
                return _engine_error_response(payload, exc)

            async def event_generator():
                text = ""
                stop_reason = "length"
                timeout_s = payload.timeout_ms / 1000.0 if payload.timeout_ms else None
                meta_event = {
                    "model": _engine_model_label(),
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

                try:
                    timed_out = False
                    while True:
                        if timeout_s and (monotonic() - started) > timeout_s:
                            timed_out = True
                            stop_reason = "timeout"
                            break

                        chunk = await asyncio.to_thread(next, chunk_iterator, None)
                        if chunk is None:
                            break

                        candidate = text + chunk
                        cut_pos = _find_stop_position(candidate, payload.stop)
                        if cut_pos is not None:
                            safe_part = candidate[:cut_pos]
                            delta = safe_part[len(text) :]
                            if delta:
                                yield {"event": "delta", "data": json.dumps({"delta": delta}, ensure_ascii=False)}
                            text = safe_part
                            stop_reason = "stop"
                            break

                        text = candidate
                        yield {"event": "delta", "data": json.dumps({"delta": chunk}, ensure_ascii=False)}

                    if timed_out:
                        stop_reason = "timeout"
                    elif stop_reason != "stop":
                        produced = _output_progress_units(text)
                        stop_reason = "length" if produced >= payload.max_tokens else "stop"

                    elapsed_ms = int((monotonic() - started) * 1000)
                    done_meta = _build_meta(
                        req=payload,
                        model=_engine_model_label(),
                        elapsed_ms=elapsed_ms,
                        text=text,
                        stop_reason=stop_reason,
                        tokens_out=_tokens_out_for_text(text),
                    )
                    yield {"event": "done", "data": json.dumps({"text": text, "meta": done_meta}, ensure_ascii=False)}
                except Exception as exc:  # noqa: BLE001
                    elapsed_ms = int((monotonic() - started) * 1000)
                    error_meta = _build_meta(
                        req=payload,
                        model=_engine_model_label(),
                        elapsed_ms=elapsed_ms,
                        text=text,
                        stop_reason="error",
                        tokens_out=0,
                    )
                    if is_cuda_runtime_error(exc):
                        _disable_gpu("cuda_runtime_error", detail=summarize_exc(exc))
                        logger.error("CUDA runtime error detected; GPU latched/disabled; restart required")
                        with suppress(Exception):
                            torch.cuda.synchronize()
                        yield {
                            "event": "error",
                            "data": json.dumps(
                                {
                                    "message": "GPU 추론 오류로 GPU가 비활성화되었습니다. bitnetd 재시작이 필요합니다.",
                                    "detail": _gpu_disabled_detail or _gpu_disabled_reason,
                                    "meta": error_meta,
                                },
                                ensure_ascii=False,
                            ),
                        }
                    elif isinstance(exc, EngineError):
                        yield {
                            "event": "error",
                            "data": json.dumps(
                                {
                                    "message": exc.error,
                                    "detail": exc.detail,
                                    "meta": error_meta,
                                },
                                ensure_ascii=False,
                            ),
                        }
                    else:
                        yield {
                            "event": "error",
                            "data": json.dumps({"message": str(exc), "meta": error_meta}, ensure_ascii=False),
                        }
                finally:
                    await _mark_generation_end()

            return EventSourceResponse(event_generator())

        try:
            text, stop_reason, tokens_out = await asyncio.to_thread(
                engine.generate_nonstream,
                payload.prompt,
                params=params,
            )
        except VocabMismatchError as exc:
            return _engine_error_response(payload, exc)
        except Exception as exc:  # noqa: BLE001
            if is_cuda_runtime_error(exc):
                _disable_gpu("cuda_runtime_error", detail=summarize_exc(exc))
                logger.error("CUDA runtime error detected; GPU latched/disabled; restart required")
                with suppress(Exception):
                    torch.cuda.synchronize()
                return _gpu_disabled_response(payload)
            if isinstance(exc, EngineError):
                return _engine_error_response(payload, exc)
            raise

        elapsed_ms = int((monotonic() - started) * 1000)
        meta = _build_meta(
            req=payload,
            model=_engine_model_label(),
            elapsed_ms=elapsed_ms,
            text=text,
            stop_reason=stop_reason,
            tokens_out=tokens_out,
        )
        return JSONResponse(content={"text": text, "meta": meta})
    except Exception as exc:  # noqa: BLE001
        state.status = "error"
        state.reasons = ["model_load_failed"]
        error_meta = _build_meta(
            req=payload,
            model=_engine_model_label(),
            elapsed_ms=0,
            text="",
            stop_reason="error",
            tokens_out=0,
        )
        return JSONResponse(status_code=500, content={"text": "", "meta": error_meta, "error": str(exc)})
    finally:
        if not stream_manages_finally:
            await _mark_generation_end()
