from __future__ import annotations

from .base import EngineError, GenerateParams
from .torch_engine import TorchEngine


class StubBitnetCppEngine:
    engine_id = "bitnetcpp"
    stream_supported = False

    def model_label(self) -> str:
        return "bitnetcpp-unavailable"

    def is_ready(self) -> tuple[bool, list[str]]:
        return False, ["engine_not_implemented"]

    def ensure_loaded(self, snapshot_path: str) -> None:
        return

    def generate_nonstream(self, prompt: str, *, params: GenerateParams) -> tuple[str, str, int]:
        raise EngineError(
            error="선택한 엔진(bitnetcpp)은 아직 구현되지 않았습니다.",
            detail="engine_not_implemented",
            status_code=503,
        )

    def generate_stream(self, prompt: str, *, params: GenerateParams):
        raise EngineError(
            error="선택한 엔진(bitnetcpp)은 아직 구현되지 않았습니다.",
            detail="engine_not_implemented",
            status_code=503,
        )


def create_engine(config) -> object:
    if getattr(config, "engine", "torch") == "bitnetcpp":
        return StubBitnetCppEngine()
    return TorchEngine()
