from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from shared.constants import bitnet_home


@dataclass(slots=True)
class BitnetCppConfig:
    mode: str = "script"
    python_exe: str = sys.executable
    script_path: str = ""
    exe_path: str = ""
    model_path: str = ""
    ctx_size: int = 2048
    threads: int = max(1, (os.cpu_count() or 2) // 2)
    extra_args: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BitnetdConfig:
    engine: str = "torch"
    bitnetcpp: BitnetCppConfig = field(default_factory=BitnetCppConfig)


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

    cpp_raw = raw.get("bitnetcpp", {}) if isinstance(raw.get("bitnetcpp", {}), dict) else {}
    mode = str(cpp_raw.get("mode", "script")).strip().lower() or "script"
    python_exe = str(cpp_raw.get("python_exe", "")).strip() or sys.executable
    script_path = str(cpp_raw.get("script_path", "")).strip()
    exe_path = str(cpp_raw.get("exe_path", "")).strip()
    model_path = str(cpp_raw.get("model_path", "")).strip()

    try:
        ctx_size = int(cpp_raw.get("ctx_size", 2048))
    except Exception:
        ctx_size = 2048
    ctx_size = max(256, ctx_size)

    try:
        threads = int(cpp_raw.get("threads", max(1, (os.cpu_count() or 2) // 2)))
    except Exception:
        threads = max(1, (os.cpu_count() or 2) // 2)
    threads = max(1, threads)

    extra_args_raw = cpp_raw.get("extra_args", [])
    extra_args = [str(v) for v in extra_args_raw] if isinstance(extra_args_raw, list) else []

    return BitnetdConfig(
        engine=engine,
        bitnetcpp=BitnetCppConfig(
            mode=mode,
            python_exe=python_exe,
            script_path=script_path,
            exe_path=exe_path,
            model_path=model_path,
            ctx_size=ctx_size,
            threads=threads,
            extra_args=extra_args,
        ),
    )
