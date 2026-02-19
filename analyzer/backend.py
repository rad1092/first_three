from __future__ import annotations

import atexit
import threading
from pathlib import Path
from uuid import uuid4

import requests

from shared.constants import BITNETD_BASE_URL, bitnet_home

HEALTH_URL = f"{BITNETD_BASE_URL}/health"
REQUEST_TIMEOUT_SECONDS = 1.0
GENERATE_TIMEOUT_SECONDS = 45.0
CLIENT_HEARTBEAT_SECONDS = 5.0


def _token_file_path() -> Path:
    return bitnet_home() / "config" / "token.txt"


def _read_token() -> str:
    return _token_file_path().read_text(encoding="utf-8").strip()


class BitnetClient:
    def __init__(self) -> None:
        self.client_id = str(uuid4())
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    def _headers(self) -> dict[str, str]:
        token = _read_token()
        return {"X-Local-Token": token, "Content-Type": "application/json"}

    def _post_clients(self, path: str, payload: dict) -> bool:
        url = f"{BITNETD_BASE_URL}{path}"
        try:
            response = requests.post(
                url,
                json=payload,
                headers=self._headers(),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def register(self) -> None:
        ok = self._post_clients(
            "/clients/register",
            {"client_id": self.client_id, "app_name": "analyzer"},
        )
        if not ok:
            return

        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(CLIENT_HEARTBEAT_SECONDS):
            self._post_clients("/clients/heartbeat", {"client_id": self.client_id})

    def unregister(self) -> None:
        self._heartbeat_stop.set()
        self._post_clients("/clients/unregister", {"client_id": self.client_id})

    def generate_text(
        self,
        *,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.5,
        top_p: float = 0.9,
        timeout_ms: int = 25000,
    ) -> tuple[bool, str]:
        body = {
            "prompt": prompt,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "timeout_ms": timeout_ms,
        }
        try:
            response = requests.post(
                f"{BITNETD_BASE_URL}/generate",
                json=body,
                headers=self._headers(),
                timeout=GENERATE_TIMEOUT_SECONDS,
            )
            if response.status_code != 200:
                return False, "엔진 연결이 필요해요. 우측 상단 연결 상태를 확인해 주세요."
            data = response.json()
            text = str(data.get("text", "")).strip()
            if not text:
                return False, "엔진 연결이 필요해요. 우측 상단 연결 상태를 확인해 주세요."
            return True, text
        except (requests.RequestException, ValueError):
            return False, "엔진 연결이 필요해요. 우측 상단 연결 상태를 확인해 주세요."


_bitnet_client = BitnetClient()
atexit.register(_bitnet_client.unregister)


def get_bitnet_client() -> BitnetClient:
    return _bitnet_client


def check_bitnetd_health() -> tuple[bool, str]:
    """Check bitnetd health endpoint and return a user-facing connection summary."""
    try:
        response = requests.get(HEALTH_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 200:
            return True, "연결 성공"
        return False, "연결 실패"
    except requests.RequestException:
        return False, "연결 실패"
