from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

            logger.info("Starting model load model_id=%s revision=%s", model_id, revision)
            snapshot_path = resolve_model_snapshot(model_id=model_id, revision=revision)
            tokenizer = AutoTokenizer.from_pretrained(str(snapshot_path), trust_remote_code=True)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Attempting model load on device=%s", device)
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
