from __future__ import annotations

from ..config import BitnetdConfig
from .bitnetcpp_engine import BitnetCppEngine
from .torch_engine import TorchEngine


def create_engine(config: BitnetdConfig) -> object:
    if getattr(config, "engine", "torch") == "bitnetcpp":
        return BitnetCppEngine(config.bitnetcpp)
    return TorchEngine()
