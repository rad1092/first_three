from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(slots=True)
class GenerateParams:
    max_tokens: int
    temperature: float
    top_p: float
    repeat_penalty: float
    stop: list[str] = field(default_factory=list)
    timeout_ms: int | None = None
    seed: int | None = None


class EngineError(RuntimeError):
    def __init__(self, *, error: str, detail: str | None = None, status_code: int = 500) -> None:
        super().__init__(detail or error)
        self.error = error
        self.detail = detail
        self.status_code = status_code


class VocabMismatchError(EngineError):
    def __init__(self, *, detail: str) -> None:
        super().__init__(error="tokenizer/model vocab mismatch", detail=detail, status_code=400)


class Engine(Protocol):
    engine_id: str
    stream_supported: bool

    def model_label(self) -> str: ...

    def is_ready(self) -> tuple[bool, list[str]]: ...

    def ensure_loaded(self, snapshot_path: str) -> None: ...

    def generate_nonstream(
        self,
        prompt: str,
        *,
        params: GenerateParams,
    ) -> tuple[str, str, int]: ...

    def generate_stream(self, prompt: str, *, params: GenerateParams): ...
