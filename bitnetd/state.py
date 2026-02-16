from __future__ import annotations

from time import monotonic
from typing import Literal

from pydantic import BaseModel, Field

from shared.constants import BITNETD_BASE_URL

HealthStatus = Literal["ready", "starting", "not_ready", "shutting_down", "error"]


class HealthResponse(BaseModel):
    status: HealthStatus
    reasons: list[str] = Field(default_factory=list)
    client_count: int = 0
    uptime_ms: int
    token_enabled: bool = True
    base_url: str = BITNETD_BASE_URL


class ServerState:
    """In-memory state for health reporting in Phase 1-1."""

    def __init__(self) -> None:
        self.started_at = monotonic()
        self.status: HealthStatus = "not_ready"
        self.reasons: list[str] = ["weights_missing"]
        self.client_count: int = 0
        self.token_enabled: bool = True

    def uptime_ms(self) -> int:
        return int((monotonic() - self.started_at) * 1000)

    def health(self) -> HealthResponse:
        return HealthResponse(
            status=self.status,
            reasons=self.reasons,
            client_count=self.client_count,
            uptime_ms=self.uptime_ms(),
            token_enabled=self.token_enabled,
            base_url=BITNETD_BASE_URL,
        )
