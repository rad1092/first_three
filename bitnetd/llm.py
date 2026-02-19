from __future__ import annotations

import logging
import os
import shutil
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WINDOWS_MISSING_CL = os.name == "nt" and shutil.which("cl") is None
if _WINDOWS_MISSING_CL:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import torch
import transformers

DEFAULT_MODEL_ID = "microsoft/bitnet-b1.58-2B-4T"
DEFAULT_REVISION = "main"


class CompilerMissingError(RuntimeError):
    pass


class BitNetModelService:
    def __init__(self) -> None:
        self._loaded: dict[str, Any] | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    @property
    def loaded(self) -> dict[str, Any]:
        if self._loaded is None:
            raise RuntimeError("model_not_loaded")
        return self._loaded

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
            first_param = next(model.parameters())
            return str(first_param.device)
        return "unknown"

    def _load_on_cuda(self, model_class: Any, snapshot_path: str) -> Any:
        kwargs = {
            "device_map": "cuda",
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }
        try:
            return model_class.from_pretrained(snapshot_path, **kwargs)
        except TypeError:
            logger.warning("CUDA direct-load kwargs not supported; falling back to classic load")
            model = model_class.from_pretrained(snapshot_path)
            return model.to("cuda")

    def load_if_needed(self, snapshot_path: str, model_id: str = DEFAULT_MODEL_ID, revision: str = DEFAULT_REVISION) -> dict[str, Any]:
        if self._loaded is not None:
            return self._loaded

        with self._lock:
            if self._loaded is not None:
                return self._loaded

            model_class = self._resolve_model_class()

            if _WINDOWS_MISSING_CL:
                logger.warning(
                    "Windows detected; cl.exe not found; disabling torch compile/inductor for compatibility."
                )
                with suppress(Exception):
                    import torch._dynamo

                    torch._dynamo.config.disable = True

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Attempting BitNet model load on device=%s", device)

            try:
                if device == "cuda":
                    model = self._load_on_cuda(model_class, snapshot_path)
                else:
                    model = model_class.from_pretrained(snapshot_path)
                    model = model.to("cpu")
            except Exception as exc:
                if _WINDOWS_MISSING_CL and "cl is not found" in str(exc).lower():
                    raise CompilerMissingError(
                        "Visual Studio Build Tools (C++ compiler cl.exe) is required OR torch compile must be disabled; currently cl.exe not found."
                    ) from exc
                if device == "cuda":
                    logger.warning("CUDA load failed, falling back to CPU: %s", exc)
                    model = model_class.from_pretrained(snapshot_path)
                    model = model.to("cpu")
                    device = "cpu"
                else:
                    raise

            first_param_device = self._first_param_device(model)
            if device == "cuda" and not first_param_device.startswith("cuda"):
                logger.warning("GPU available but model stayed on %s; forcing model.to('cuda')", first_param_device)
                with suppress(Exception):
                    model = model.to("cuda")
                first_param_device = self._first_param_device(model)
                if not first_param_device.startswith("cuda"):
                    logger.warning("GPU move failed; keeping CPU fallback")
                    with suppress(Exception):
                        model = model.to("cpu")
                    device = "cpu"

            model.eval()
            logger.info("First param device=%s", first_param_device)
            logger.info(
                "Model load completed model_id=%s revision=%s device=%s snapshot_path=%s",
                model_id,
                revision,
                device,
                snapshot_path,
            )

            self._loaded = {
                "model_id": model_id,
                "revision": revision,
                "snapshot_path": str(Path(snapshot_path)),
                "model": model,
                "device": device,
            }
            return self._loaded
