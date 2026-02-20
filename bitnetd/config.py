from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from shared.constants import bitnet_home


@dataclass(slots=True)
class BitnetdConfig:
    engine: str = "torch"
    bitnetcpp: dict = field(default_factory=dict)


def _config_path() -> Path:
    return bitnet_home() / "config" / "bitnetd.json"


def load_config() -> BitnetdConfig:
    raw: dict = {}
    cfg_path = _config_path()
    if cfg_path.exists():
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            raw = {}

    engine = str(raw.get("engine", "torch")).strip().lower() or "torch"
    env_engine = os.getenv("BITNETD_ENGINE", "").strip().lower()
    if env_engine in {"torch", "bitnetcpp"}:
        engine = env_engine

    return BitnetdConfig(engine=engine, bitnetcpp=raw.get("bitnetcpp", {}) if isinstance(raw.get("bitnetcpp", {}), dict) else {})
