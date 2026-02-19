from __future__ import annotations

import hmac
import secrets
from pathlib import Path

from fastapi import Header, HTTPException, status

from shared.constants import bitnet_home, ensure_dirs

TOKEN_HEADER_NAME = "X-Local-Token"
_TOKEN_FILE_NAME = "token.txt"


def token_file_path() -> Path:
    """Return `%LOCALAPPDATA%\\BitNet\\config\\token.txt` path."""
    return bitnet_home() / "config" / _TOKEN_FILE_NAME


def _read_token(token_path: Path) -> str:
    return token_path.read_text(encoding="utf-8").strip()


def get_or_create_token() -> str:
    """Load local token from disk, creating it if missing or empty."""
    ensure_dirs()
    token_path = token_file_path()

    if token_path.exists():
        token = _read_token(token_path)
        if token:
            return token

    token = secrets.token_urlsafe(32)
    token_path.write_text(token + "\n", encoding="utf-8")
    return token


def verify_token(provided_token: str) -> bool:
    """Constant-time token comparison against persisted local token."""
    expected = get_or_create_token()
    return hmac.compare_digest(provided_token, expected)


async def require_token(x_local_token: str | None = Header(default=None)) -> str:
    """FastAPI dependency for token-protected endpoints."""
    if not x_local_token or not verify_token(x_local_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_or_missing_token",
        )
    return x_local_token
