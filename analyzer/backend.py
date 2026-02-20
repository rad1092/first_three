from __future__ import annotations

import atexit
import threading
from pathlib import Path
from uuid import uuid4

import requests

from shared.constants import BITNETD_BASE_URL, bitnet_home

HEALTH_URL = f"{BITNETD_BASE_URL}/health"
REQUEST_TIMEOUT_SECONDS = 1.0
GENERATE_TIMEOUT_SECONDS = 100.0
CLIENT_HEARTBEAT_SECONDS = 5.0
ERROR_TEXT_MAX_LENGTH = 240


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

    def _clip_text(self, value: object, limit: int = ERROR_TEXT_MAX_LENGTH) -> str:
        text_value = str(value or "").strip()
        if len(text_value) <= limit:
            return text_value
        return f"{text_value[:limit].rstrip()}…"

    def _build_error_message(self, response: requests.Response) -> str:
        if response.status_code == 401:
            return "토큰 인증에 실패했어요. /docs Authorize 또는 token.txt 값을 확인해 주세요."

        message = ""
        try:
            data = response.json()
        except ValueError:
            data = None

        if isinstance(data, dict):
            meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
            stop_reason = str(meta.get("stop_reason", "")).strip().lower()
            if stop_reason == "timeout":
                return "응답 시간이 초과됐어요. 잠시 기다리거나 max_tokens를 줄여 다시 시도해 주세요."

            primary = self._clip_text(data.get("error") or data.get("message"))
            detail = self._clip_text(data.get("detail"))

            if primary:
                message = primary
                if detail and detail not in message:
                    message = f"{message} ({detail})"
            elif detail:
                message = detail

        if message:
            return message
        if response.status_code >= 500:
            return "엔진 내부 오류가 발생했어요. 잠시 후 다시 시도해 주세요."
        return "요청 처리에 실패했어요. 엔진 연결 상태와 입력값을 확인해 주세요."

    def _post_generate(
        self,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        timeout_ms: int,
        stop: list[str] | None,
    ) -> tuple[bool, str]:
        body = {
            "prompt": prompt,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "timeout_ms": timeout_ms,
            "stop": stop,
        }

        try:
            headers = self._headers()
        except FileNotFoundError:
            return False, "토큰 파일을 찾을 수 없어요. %LOCALAPPDATA%\\BitNet\\config\\token.txt 를 확인해 주세요."
        except OSError:
            return False, "토큰 파일을 읽지 못했어요. 파일 권한과 내용을 확인해 주세요."

        try:
            response = requests.post(
                f"{BITNETD_BASE_URL}/generate",
                json=body,
                headers=headers,
                timeout=GENERATE_TIMEOUT_SECONDS,
            )
        except requests.Timeout:
            return False, "응답이 느려요. 잠시 기다리거나 생성 길이(max_tokens)를 줄여서 다시 시도해 주세요."
        except requests.RequestException:
            return False, "엔진 연결이 필요해요. 우측 상단 연결 상태를 확인해 주세요."

        if response.status_code != 200:
            return False, self._build_error_message(response)

        try:
            data = response.json()
        except ValueError:
            return False, "엔진 응답을 해석하지 못했어요. 잠시 후 다시 시도해 주세요."

        text = str(data.get("text", "")).strip() if isinstance(data, dict) else ""
        if text:
            return True, text

        meta = data.get("meta") if isinstance(data, dict) else {}
        stop_reason = str((meta or {}).get("stop_reason", "")).strip().lower()
        if stop_reason == "timeout":
            return False, "응답 시간이 초과됐어요. 잠시 기다리거나 max_tokens를 줄여 다시 시도해 주세요."
        return False, "엔진이 빈 응답을 반환했어요. 잠시 후 다시 시도해 주세요."

    def generate_text(
        self,
        *,
        prompt: str,
        max_tokens: int = 72,
        temperature: float = 0.4,
        top_p: float = 0.85,
        timeout_ms: int = 90000,
        stop: list[str] | None = None,
    ) -> tuple[bool, str]:
        stop_list = stop or [
            "```",
            "```python",
            "def ",
            "import ",
            "OUTPUT",
            "desired_result",
            "System:",
            "User:",
            "AI:",
            "NAME:",
            "Desired Result",
        ]
        return self._post_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout_ms=timeout_ms,
            stop=stop_list,
        )

    def generate_raw(
        self,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        timeout_ms: int,
        stop: list[str] | None = None,
    ) -> tuple[bool, str]:
        return self._post_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout_ms=timeout_ms,
            stop=stop,
        )


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
