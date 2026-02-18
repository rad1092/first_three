from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from .model_store import DEFAULT_MODEL_ID, DEFAULT_REVISION, resolve_model_snapshot

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LoadedModel:
    model_id: str
    revision: str
    snapshot_path: str
    tokenizer: Any
    model: Any
    device: str


class BitNetModelService:
    def __init__(self) -> None:
        self._loaded: LoadedModel | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    @property
    def loaded(self) -> LoadedModel:
        if self._loaded is None:
            raise RuntimeError("model_not_loaded")
        return self._loaded

    def load_if_needed(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_REVISION,
    ) -> LoadedModel:
        if self._loaded is not None:
            return self._loaded

        with self._lock:
            if self._loaded is not None:
                return self._loaded

            snapshot_path = resolve_model_snapshot(model_id=model_id, revision=revision)
            tokenizer = AutoTokenizer.from_pretrained(
                str(snapshot_path),
                trust_remote_code=True,
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    str(snapshot_path),
                    trust_remote_code=True,
                )
                model = model.to(device)
            except Exception as exc:
                if device != "cpu":
                    logger.warning("CUDA load failed, falling back to CPU: %s", exc)
                    device = "cpu"
                    model = AutoModelForCausalLM.from_pretrained(
                        str(snapshot_path),
                        trust_remote_code=True,
                    )
                    model = model.to(device)
                else:
                    raise

            model.eval()
            self._loaded = LoadedModel(
                model_id=model_id,
                revision=revision,
                snapshot_path=str(snapshot_path),
                tokenizer=tokenizer,
                model=model,
                device=device,
            )
            return self._loaded

    def prepare_generation_kwargs(
        self,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repeat_penalty: float,
        timeout_ms: int | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repeat_penalty,
        }

        if temperature <= 0:
            kwargs["do_sample"] = False
        else:
            kwargs["do_sample"] = True
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p

        if timeout_ms and timeout_ms > 0:
            kwargs["max_time"] = timeout_ms / 1000.0

        return kwargs


def seed_torch(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_streamer(tokenizer: Any) -> TextIteratorStreamer:
    return TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
