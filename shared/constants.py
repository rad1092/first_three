from __future__ import annotations

import os
from pathlib import Path

BITNETD_BASE_URL = "http://127.0.0.1:11435"


def get_localappdata() -> Path:
    """Return LOCALAPPDATA path; fallback to user home-based AppData/Local."""
    env_path = os.getenv("LOCALAPPDATA")
    if env_path:
        return Path(env_path)
    return Path.home() / "AppData" / "Local"


def bitnet_home() -> Path:
    """%LOCALAPPDATA%\\BitNet\\"""
    return get_localappdata() / "BitNet"


def analyzer_home() -> Path:
    """%LOCALAPPDATA%\\AnalyzerApp\\"""
    return get_localappdata() / "AnalyzerApp"


def ensure_dirs() -> None:
    """Create required runtime directories for Phase 0 baseline."""
    bitnet_root = bitnet_home()
    analyzer_root = analyzer_home()

    for rel in ("bin", "models", "config", "cache", "logs"):
        (bitnet_root / rel).mkdir(parents=True, exist_ok=True)

    for rel in ("results", "logs"):
        (analyzer_root / rel).mkdir(parents=True, exist_ok=True)
