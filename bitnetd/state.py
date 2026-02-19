from __future__ import annotations

import asyncio
import logging
import os
from time import monotonic
from typing import Literal

from pydantic import BaseModel, Field

from shared.constants import BITNETD_BASE_URL

logger = logging.getLogger(__name__)

HealthStatus = Literal["ready", "starting", "not_ready", "shutting_down", "error"]
AllowedAppName = Literal["analyzer", "chat", "agent"]


class HealthResponse(BaseModel):
    status: HealthStatus
    reasons: list[str] = Field(default_factory=list)
    client_count: int = 0
    uptime_ms: int
    token_enabled: bool = True
    base_url: str = BITNETD_BASE_URL


class ClientInfo(BaseModel):
    app_name: AllowedAppName
    last_seen: float


class ServerState:
    """In-memory state for health, client lifecycle, and shutdown logic."""

    def __init__(self) -> None:
        self.started_at = monotonic()
        self.status: HealthStatus = "not_ready"
        self.reasons: list[str] = ["weights_missing"]
        self.token_enabled: bool = True

        self.clients: dict[str, ClientInfo] = {}
        self.active_generations: int = 0
        self.exit_pending: bool = False
        self.has_ever_had_client: bool = False
        self.last_client_count: int = 0
        self.lock = asyncio.Lock()

    def uptime_ms(self) -> int:
        return int((monotonic() - self.started_at) * 1000)

    def _update_last_count(self, new_count: int) -> None:
        self.last_client_count = new_count

    def should_exit_on_transition_to_zero(self, prev_count: int, new_count: int) -> bool:
        transitioned_to_zero = self.has_ever_had_client and prev_count > 0 and new_count == 0
        if transitioned_to_zero:
            self.exit_pending = True

        return transitioned_to_zero and self.active_generations == 0

    async def register_client(self, client_id: str, app_name: AllowedAppName) -> int:
        async with self.lock:
            self.clients[client_id] = ClientInfo(app_name=app_name, last_seen=monotonic())
            self.exit_pending = False
            self.has_ever_had_client = True
            new_count = len(self.clients)
            self._update_last_count(new_count)
            return new_count

    async def heartbeat_client(self, client_id: str) -> tuple[bool, int, str | None]:
        async with self.lock:
            info = self.clients.get(client_id)
            if info is None:
                return False, len(self.clients), "client_not_registered"

            info.last_seen = monotonic()
            self.clients[client_id] = info
            return True, len(self.clients), None

    async def unregister_client(self, client_id: str) -> tuple[bool, int, int, str | None]:
        async with self.lock:
            before_count = len(self.clients)
            removed = self.clients.pop(client_id, None)
            after_count = len(self.clients)
            self._update_last_count(after_count)

            if removed is None:
                return False, before_count, after_count, "client_not_registered"

            return True, before_count, after_count, None

    async def prune_expired_clients(self, ttl_seconds: int = 15) -> tuple[int, int, int]:
        now = monotonic()
        async with self.lock:
            before_count = len(self.clients)
            expired: list[tuple[str, float]] = []

            for client_id, info in self.clients.items():
                age_seconds = now - info.last_seen
                if age_seconds > ttl_seconds:
                    expired.append((client_id, age_seconds))

            for client_id, age_seconds in expired:
                self.clients.pop(client_id, None)
                logger.debug(
                    "Pruned expired client id=%s age_seconds=%.3f ttl_seconds=%s",
                    client_id,
                    age_seconds,
                    ttl_seconds,
                )

            after_count = len(self.clients)
            self._update_last_count(after_count)
            return len(expired), before_count, after_count

    async def health(self) -> HealthResponse:
        async with self.lock:
            client_count = len(self.clients)

        return HealthResponse(
            status=self.status,
            reasons=self.reasons,
            client_count=client_count,
            uptime_ms=self.uptime_ms(),
            token_enabled=self.token_enabled,
            base_url=BITNETD_BASE_URL,
        )

    def exit_now(self) -> None:
        os._exit(0)
