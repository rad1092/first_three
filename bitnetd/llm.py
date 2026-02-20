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

DEFAULT_MODEL_ID = "microsoft/bitnet-b1.58-2B-4T"
DEFAULT_REVISION = "main"


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
        self._loaded_cpu: LoadedModel | None = None
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
        kwargs: dict[str, Any] = {}
        sig = inspect.signature(AutoTokenizer.from_pretrained)
        if "fix_mistral_regex" in sig.parameters:
            kwargs["fix_mistral_regex"] = True
        return AutoTokenizer.from_pretrained(snapshot_path, **kwargs)

    def _resolve_model_class(self) -> Any:
        model_class = getattr(transformers, "BitNetForCausalLM", None)
        if model_class is None:
            raise RuntimeError(
                "BitNet model class is unavailable in installed transformers. "
                "Upgrade transformers to a BitNet-supporting version."
            )
        return model_class

    def _first_param_device(self, model: Any) -> str:
        with suppress(Exception):
            return str(next(model.parameters()).device)
        return "unknown"

    def _cuda_direct_load(self, model_class: Any, snapshot_path: str) -> Any:
        kwargs = {
            "device_map": "cuda",
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }
        try:
            return model_class.from_pretrained(snapshot_path, **kwargs)
        except TypeError:
            logger.warning("CUDA direct-load kwargs unsupported; falling back to classic load")
            model = model_class.from_pretrained(snapshot_path)
            return model.to("cuda")

    def load_if_needed(
        self,
        snapshot_path: str,
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

            tokenizer = self._load_tokenizer(snapshot_path)
            model_class = self._resolve_model_class()

            intended_device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Attempting BitNet model load on device=%s", intended_device)

            try:
                if intended_device == "cuda":
                    model = self._cuda_direct_load(model_class, snapshot_path)
                else:
                    model = model_class.from_pretrained(snapshot_path)
                    model = model.to("cpu")
            except Exception as exc:
                if _WINDOWS_MISSING_CL and "cl is not found" in str(exc).lower():
                    raise CompilerMissingError(
                        "Visual Studio Build Tools (C++ compiler cl.exe) is required OR torch compile must be disabled; currently cl.exe not found."
                    ) from exc
                if intended_device == "cuda":
                    logger.warning("CUDA load failed, falling back to CPU: %s", exc)
                    model = model_class.from_pretrained(snapshot_path)
                    model = model.to("cpu")
                    intended_device = "cpu"
                else:
                    raise

            first_param_device = self._first_param_device(model)
            if intended_device == "cuda" and not first_param_device.startswith("cuda"):
                logger.warning("GPU available but model stayed on %s; forcing model.to('cuda')", first_param_device)
                with suppress(Exception):
                    model = model.to("cuda")
                first_param_device = self._first_param_device(model)
                if not first_param_device.startswith("cuda"):
                    logger.warning("GPU move failed; keeping CPU fallback")
                    with suppress(Exception):
                        model = model.to("cpu")
                    intended_device = "cpu"

            model.eval()
            logger.info("First param device=%s", first_param_device)
            logger.info(
                "Model load completed model_id=%s revision=%s device=%s snapshot_path=%s",
                model_id,
                revision,
                intended_device,
                snapshot_path,
            )

            self._loaded = LoadedModel(
                model_id=model_id,
                revision=revision,
                snapshot_path=snapshot_path,
                tokenizer=tokenizer,
                model=model,
                device=intended_device,
            )
            return self._loaded


    def load_cpu_fallback(
        self,
        snapshot_path: str,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_REVISION,
    ) -> LoadedModel:
        if self._loaded_cpu is not None:
            return self._loaded_cpu

        with self._lock:
            if self._loaded_cpu is not None:
                return self._loaded_cpu

            tokenizer = self._load_tokenizer(snapshot_path)
            model_class = self._resolve_model_class()
            model = model_class.from_pretrained(snapshot_path)
            model = model.to("cpu")
            model.eval()

            loaded = LoadedModel(
                model_id=model_id,
                revision=revision,
                snapshot_path=snapshot_path,
                tokenizer=tokenizer,
                model=model,
                device="cpu",
            )
            self._loaded_cpu = loaded
            return loaded


def seed_torch(seed: int | None, *, use_cuda: bool) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if use_cuda:
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
