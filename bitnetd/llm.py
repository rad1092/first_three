from __future__ import annotations

import inspect
import logging
import os
import shutil
import threading
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_WINDOWS_MISSING_CL = os.name == "nt" and shutil.which("cl") is None
if _WINDOWS_MISSING_CL:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import torch
import transformers
from transformers import AutoTokenizer, TextIteratorStreamer

from .model_store import DEFAULT_MODEL_ID, DEFAULT_REVISION, resolve_model_snapshot


class CompilerMissingError(RuntimeError):
    pass


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

    def _load_tokenizer(self, snapshot_path: str) -> Any:
        tokenizer_kwargs: dict[str, Any] = {}
        signature = inspect.signature(AutoTokenizer.from_pretrained)
        if "fix_mistral_regex" in signature.parameters:
            tokenizer_kwargs["fix_mistral_regex"] = True
        return AutoTokenizer.from_pretrained(snapshot_path, **tokenizer_kwargs)

    def _resolve_bitnet_model_class(self) -> Any:
        model_class = getattr(transformers, "BitNetForCausalLM", None)
        if model_class is None:
            raise RuntimeError(
                "BitNet model class is unavailable in installed transformers. "
                "Upgrade transformers to a BitNet-supporting version (e.g. >=4.48.0)."
            )
        return model_class

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

            if _WINDOWS_MISSING_CL:
                logger.warning(
                    "Windows detected; cl.exe not found; disabling torch compile/inductor for compatibility."
                )
                with suppress(Exception):
                    import torch._dynamo

                    torch._dynamo.config.disable = True

            logger.info("Starting model load model_id=%s revision=%s", model_id, revision)
            snapshot_path = resolve_model_snapshot(model_id=model_id, revision=revision)
            tokenizer = self._load_tokenizer(str(snapshot_path))
            model_class = self._resolve_bitnet_model_class()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Attempting BitNet model load on device=%s", device)
            try:
                model = model_class.from_pretrained(str(snapshot_path))
                model = model.to(device)
            except Exception as exc:
                if _WINDOWS_MISSING_CL and (
                    "cl is not found" in str(exc).lower()
                    or "cpp_builder" in str(exc).lower()
                    or "inductor" in str(exc).lower()
                ):
                    raise CompilerMissingError(
                        "Visual Studio Build Tools (C++ compiler cl.exe) is required OR torch compile "
                        "must be disabled; currently cl.exe not found."
                    ) from exc

                if device != "cpu":
                    logger.warning("CUDA load failed, falling back to CPU: %s", exc)
                    device = "cpu"
                    model = model_class.from_pretrained(str(snapshot_path))
                    model = model.to(device)
                else:
                    logger.error("CPU model load failed: %s", exc)
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
            logger.info(
                "Model load completed model_id=%s revision=%s device=%s snapshot_path=%s",
                model_id,
                revision,
                device,
                snapshot_path,
            )
            return self._loaded


def seed_torch(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def prepare_generation_kwargs(
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
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
    return kwargs


def build_streamer(tokenizer: Any) -> TextIteratorStreamer:
    return TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
