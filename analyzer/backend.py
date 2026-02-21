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
        self._last_error: dict[str, str | int] | None = None

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

    def _build_error_message(self, response: requests.Response) -> tuple[str, str]:
        if response.status_code == 401:
            return "토큰 인증에 실패했어요. /docs Authorize 또는 token.txt 값을 확인해 주세요.", ""

        message = ""
        try:
            data = response.json()
        except ValueError:
            data = None

        detail_raw = ""
        if isinstance(data, dict):
            meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
            stop_reason = str(meta.get("stop_reason", "")).strip().lower()
            if stop_reason == "timeout":
                return "응답 시간이 초과됐어요. 잠시 기다리거나 max_tokens를 줄여 다시 시도해 주세요.", str(data.get("detail") or "")

            detail_raw = str(data.get("detail") or "").strip()
            primary_raw = str(data.get("error") or data.get("message") or "").strip()
            primary = self._clip_text(primary_raw)
            detail = self._clip_text(detail_raw)

            if response.status_code == 422:
                return (
                    "클라이언트 요청 바디 문제(422)입니다. 입력 형식/필드를 확인해 주세요.",
                    detail_raw or detail,
                )

            if response.status_code == 503 and "0xC0000409" in detail_raw.upper():
                message = "엔진 크래시(0xC0000409)로 보입니다. 안전모드 재시도가 수행되었는지 확인해 주세요."
                return message, detail_raw or detail

            if primary:
                message = primary
                if detail and detail not in message:
                    message = f"{message} ({detail})"
            elif detail:
                message = detail

        if message:
            return message, detail_raw
        if response.status_code >= 500:
            return "엔진 내부 오류가 발생했어요. 잠시 후 다시 시도해 주세요.", detail_raw
        return "요청 처리에 실패했어요. 엔진 연결 상태와 입력값을 확인해 주세요.", detail_raw

    def _set_last_error(self, *, status_code: int, summary: str, detail: str = "") -> None:
        self._last_error = {
            "status_code": int(status_code),
            "summary": summary,
            "detail": detail,
        }

    def pop_last_error(self) -> dict[str, str | int] | None:
        data = self._last_error
        self._last_error = None
        return data

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
        self._last_error = None
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
            summary = "토큰 파일을 찾을 수 없어요. %LOCALAPPDATA%\\BitNet\\config\\token.txt 를 확인해 주세요."
            self._set_last_error(status_code=401, summary=summary)
            return False, summary
        except OSError:
            summary = "토큰 파일을 읽지 못했어요. 파일 권한과 내용을 확인해 주세요."
            self._set_last_error(status_code=401, summary=summary)
            return False, summary

        try:
            response = requests.post(
                f"{BITNETD_BASE_URL}/generate",
                json=body,
                headers=headers,
                timeout=GENERATE_TIMEOUT_SECONDS,
            )
        except requests.Timeout:
            summary = "응답이 느려요. 잠시 기다리거나 생성 길이(max_tokens)를 줄여서 다시 시도해 주세요."
            self._set_last_error(status_code=504, summary=summary)
            return False, summary
        except requests.RequestException:
            summary = "엔진 연결이 필요해요. 우측 상단 연결 상태를 확인해 주세요."
            self._set_last_error(status_code=503, summary=summary)
            return False, summary

        if response.status_code != 200:
            summary, detail = self._build_error_message(response)
            self._set_last_error(status_code=response.status_code, summary=summary, detail=detail)
            return False, summary

        try:
            data = response.json()
        except ValueError:
            summary = "엔진 응답을 해석하지 못했어요. 잠시 후 다시 시도해 주세요."
            self._set_last_error(status_code=502, summary=summary)
            return False, summary

        text = str(data.get("text", "")).strip() if isinstance(data, dict) else ""
        if text:
            return True, text

        meta = data.get("meta") if isinstance(data, dict) else {}
        stop_reason = str((meta or {}).get("stop_reason", "")).strip().lower()
        if stop_reason == "timeout":
            summary = "응답 시간이 초과됐어요. 잠시 기다리거나 max_tokens를 줄여 다시 시도해 주세요."
            self._set_last_error(status_code=504, summary=summary)
            return False, summary
        summary = "엔진이 빈 응답을 반환했어요. 잠시 후 다시 시도해 주세요."
        self._set_last_error(status_code=502, summary=summary)
        return False, summary

    def run_smoke_test(
        self,
        *,
        name: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        timeout_ms: int,
    ) -> dict[str, object]:
        ok, text = self.generate_raw(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout_ms=timeout_ms,
            stop=[],
        )
        result: dict[str, object] = {"name": name, "ok": ok, "text": text if ok else ""}
        if not ok:
            result["error"] = text
            result["detail"] = (self._last_error or {}).get("detail", "")
            result["status_code"] = (self._last_error or {}).get("status_code", 0)
        return result

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


def fetch_bitnetd_health() -> dict[str, object]:
    try:
        response = requests.get(HEALTH_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        return {
            "ok": False,
            "status_code": 0,
            "status": "unreachable",
            "engine": "unknown",
            "model": "unknown",
            "reasons": [f"request_error:{exc.__class__.__name__}"],
            "raw": "",
        }

    payload: dict | None = None
    try:
        parsed = response.json()
        if isinstance(parsed, dict):
            payload = parsed
    except ValueError:
        payload = None

    return {
        "ok": response.status_code == 200,
        "status_code": response.status_code,
        "status": str((payload or {}).get("status") or "unknown"),
        "engine": str((payload or {}).get("engine") or "unknown"),
        "model": str((payload or {}).get("model") or "unknown"),
        "reasons": (payload or {}).get("reasons") or [],
        "raw": payload or response.text,
    }
