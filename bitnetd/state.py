from __future__ import annotations

import asyncio
import os
from time import monotonic
from typing import Literal

from pydantic import BaseModel, Field

from shared.constants import BITNETD_BASE_URL

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
        self.lock = asyncio.Lock()

    def uptime_ms(self) -> int:
        return int((monotonic() - self.started_at) * 1000)

    async def register_client(self, client_id: str, app_name: AllowedAppName) -> int:
        async with self.lock:
            self.clients[client_id] = ClientInfo(app_name=app_name, last_seen=monotonic())
            self.exit_pending = False
            self.has_ever_had_client = True
            return len(self.clients)

    async def heartbeat_client(self, client_id: str) -> tuple[bool, int, str | None]:
        async with self.lock:
            info = self.clients.get(client_id)
            if info is None:
                return False, len(self.clients), "client_not_registered"

            info.last_seen = monotonic()
            self.clients[client_id] = info
            return True, len(self.clients), None

    async def unregister_client(self, client_id: str) -> tuple[bool, int, str | None]:
        async with self.lock:
            removed = self.clients.pop(client_id, None)
            if removed is None:
                return False, len(self.clients), "client_not_registered"

            return True, len(self.clients), None

    async def prune_expired_clients(self, ttl_seconds: int = 15) -> tuple[int, int]:
        now = monotonic()
        async with self.lock:
            expired = [
                client_id
                for client_id, info in self.clients.items()
                if (now - info.last_seen) > ttl_seconds
            ]
            for client_id in expired:
                self.clients.pop(client_id, None)

            return len(expired), len(self.clients)

    async def should_exit_now(self) -> bool:
        async with self.lock:
            should_exit = (
                self.has_ever_had_client
                and len(self.clients) == 0
                and self.active_generations == 0
            )
            if should_exit:
                self.exit_pending = True
            return should_exit

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
