from __future__ import annotations

import logging
import threading
from contextlib import suppress
from time import monotonic
from typing import Any, Iterator

import torch

from ..llm import BitNetModelService, build_streamer, prepare_generation_kwargs
from .base import EngineError, GenerateParams, VocabMismatchError

logger = logging.getLogger(__name__)


def _find_stop_position(text: str, stop: list[str]) -> int | None:
    best: int | None = None
    for marker in stop:
        if not marker:
            continue
        idx = text.find(marker)
        if idx >= 0 and (best is None or idx < best):
            best = idx
    return best


class TorchEngine:
    engine_id = "torch"
    stream_supported = True

    def __init__(self) -> None:
        self._model_service = BitNetModelService()

    @property
    def loaded(self):
        return self._model_service.loaded

    def model_label(self) -> str:
        if self._model_service.is_loaded:
            return self._model_service.loaded.model_id
        return "unknown"

    def is_ready(self) -> tuple[bool, list[str]]:
        if self._model_service.is_loaded:
            return True, []
        return False, ["model_not_loaded"]

    def ensure_loaded(self, snapshot_path: str) -> None:
        self._model_service.load_if_needed(snapshot_path)

    def _prepare_inputs(self, prompt: str, params: GenerateParams) -> tuple[Any, Any, Any, dict[str, Any]]:
        loaded = self.loaded
        tokenizer = loaded.tokenizer
        model_input = tokenizer(prompt, return_tensors="pt")
        input_ids = model_input["input_ids"]
        attention_mask = model_input.get("attention_mask")

        vocab_size = self._vocab_size_for_model(loaded.model)
        max_id = int(input_ids.max().item()) if input_ids.numel() > 0 else 0
        if vocab_size is not None and max_id >= vocab_size:
            logger.error("vocab mismatch: max_id=%s, vocab_size=%s", max_id, vocab_size)
            raise VocabMismatchError(detail=f"max_id={max_id}, vocab_size={vocab_size}")

        generation_kwargs = prepare_generation_kwargs(
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            repeat_penalty=params.repeat_penalty,
        )
        return tokenizer, input_ids, attention_mask, generation_kwargs

    def _vocab_size_for_model(self, model: Any) -> int | None:
        with suppress(Exception):
            emb = model.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                return int(emb.weight.shape[0])
        with suppress(Exception):
            return int(getattr(model.config, "vocab_size"))
        return None

    def generate_nonstream(self, prompt: str, *, params: GenerateParams) -> tuple[str, str, int]:
        loaded = self.loaded
        tokenizer, input_ids, attention_mask, generation_kwargs = self._prepare_inputs(prompt, params)
        started = monotonic()

        with torch.inference_mode():
            outputs = loaded.model.generate(
                input_ids=input_ids.to(loaded.device),
                attention_mask=attention_mask.to(loaded.device) if attention_mask is not None else None,
                **generation_kwargs,
            )

        generated_ids = outputs[0][input_ids.shape[1] :]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        reason = "length"
        cut_pos = _find_stop_position(text, params.stop)
        if cut_pos is not None:
            text = text[:cut_pos]
            reason = "stop"
        elif params.timeout_ms and (monotonic() - started) * 1000 > params.timeout_ms:
            reason = "timeout"
        elif len(generated_ids) < params.max_tokens:
            reason = "stop"

        tokens_out = len(tokenizer.encode(text, add_special_tokens=False)) if text else 0
        return text, reason, tokens_out

    def generate_stream(self, prompt: str, *, params: GenerateParams) -> Iterator[str]:
        loaded = self.loaded
        tokenizer, input_ids, attention_mask, generation_kwargs = self._prepare_inputs(prompt, params)
        streamer = build_streamer(tokenizer)
        worker_errors: list[Exception] = []

        def _run_generate_stream() -> None:
            try:
                with torch.inference_mode():
                    loaded.model.generate(
                        input_ids=input_ids.to(loaded.device),
                        attention_mask=attention_mask.to(loaded.device) if attention_mask is not None else None,
                        streamer=streamer,
                        **generation_kwargs,
                    )
            except Exception as exc:  # noqa: BLE001
                worker_errors.append(exc)

        def _iterator() -> Iterator[str]:
            thread = threading.Thread(target=_run_generate_stream, daemon=True)
            thread.start()

            while True:
                chunk = next(streamer, None)
                if chunk is None:
                    break
                yield chunk

            thread.join(timeout=0.1)
            if worker_errors:
                raise worker_errors[0]

        return _iterator()


def is_cuda_runtime_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "device-side assert" in text or "cuda error" in text


def summarize_exc(exc: Exception, limit: int = 180) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    summary = lines[0] if lines else str(exc).strip()
    if len(summary) > limit:
        return f"{summary[:limit].rstrip()}…"
    return summary
