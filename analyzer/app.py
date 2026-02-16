from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import webview

from .backend import check_bitnetd_health
from .datasets import DatasetMeta, DatasetRegistry
from .history import append_event, ensure_history_file, load_sessions


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_title_from_text(text: str, max_len: int = 20) -> str:
    sentence = re.split(r"[\n.!?]", text.strip())[0].strip()
    if not sentence:
        return "새 채팅"
    return sentence[:max_len]


def _dataset_summary_lines(metas: list[DatasetMeta]) -> str:
    lines = ["첨부 파일 자동 로드 결과:"]
    for meta in metas:
        status = "✅" if meta.loaded_ok else "❌"
        details = f"{meta.row_count}행 x {meta.col_count}열"
        extras: list[str] = []
        if meta.encoding:
            extras.append(f"encoding={meta.encoding}")
        if meta.sheet_name:
            extras.append(f"sheet={meta.sheet_name}")
        if meta.notes:
            extras.append(meta.notes)
        suffix = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"{status} {meta.name}: {details}{suffix}")

    return "\n".join(lines)


class AnalyzerApi:
    def __init__(self) -> None:
        ensure_history_file()
        self.current_session_id: str | None = None
        self.registry = DatasetRegistry()

    def _state(self) -> dict:
        data = load_sessions()
        sessions = data["sessions"]
        messages_by_session = data["messages_by_session"]

        if self.current_session_id not in messages_by_session:
            self.current_session_id = sessions[0]["session_id"] if sessions else None

        return {
            "sessions": sessions,
            "current_session_id": self.current_session_id,
            "messages": messages_by_session.get(self.current_session_id or "", []),
            **self.registry.as_state(),
        }

    def _ensure_session(self) -> None:
        if self.current_session_id:
            return
        self.new_session()

    def get_initial_state(self) -> dict:
        ok, status_text = check_bitnetd_health()
        return {
            "connection_ok": ok,
            "connection_text": status_text,
            **self._state(),
        }

    def list_sessions(self) -> dict:
        return self._state()

    def switch_session(self, session_id: str) -> dict:
        self.current_session_id = session_id
        return self._state()

    def set_active_dataset(self, dataset_id: str) -> dict:
        self.registry.set_active(dataset_id)
        return self._state()

    def new_session(self) -> dict:
        current = load_sessions()
        next_index = len(current["sessions"]) + 1
        session_id = str(uuid4())
        now = _utc_now_iso()

        append_event(
            {
                "type": "session_created",
                "session_id": session_id,
                "title": f"새 채팅 {next_index}",
                "created_at": now,
            }
        )

        self.current_session_id = session_id
        return self._state()

    def delete_session(self, session_id: str) -> dict:
        append_event(
            {
                "type": "session_deleted",
                "session_id": session_id,
                "deleted_at": _utc_now_iso(),
            }
        )

        if self.current_session_id == session_id:
            self.current_session_id = None

        return self._state()

    def attach_files(self) -> dict:
        self._ensure_session()
        assert self.current_session_id is not None

        window = webview.windows[0]
        selected = window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=True,
            file_types=("Data files (*.csv;*.xlsx)", "All files (*.*)"),
        )

        if not selected:
            return self._state()

        file_paths = [str(p) for p in selected]
        metas = self.registry.register_files(file_paths)

        files_event = []
        for m in metas:
            p = Path(m.path)
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            files_event.append({"path": m.path, "name": m.name, "size_bytes": size})

        append_event(
            {
                "type": "files_attached",
                "session_id": self.current_session_id,
                "files": files_event,
                "created_at": _utc_now_iso(),
            }
        )

        append_event(
            {
                "type": "message_added",
                "session_id": self.current_session_id,
                "role": "assistant",
                "text": _dataset_summary_lines(metas),
                "created_at": _utc_now_iso(),
            }
        )

        return self._state()

    def send_message(self, text: str) -> dict:
        trimmed = text.strip()
        if not trimmed:
            return self._state()

        self._ensure_session()
        assert self.current_session_id is not None

        before = load_sessions()
        before_messages = before["messages_by_session"].get(self.current_session_id, [])
        user_message_count = len([m for m in before_messages if m.get("role") == "user"])

        now = _utc_now_iso()
        append_event(
            {
                "type": "message_added",
                "session_id": self.current_session_id,
                "role": "user",
                "text": trimmed,
                "created_at": now,
            }
        )

        if user_message_count == 0:
            append_event(
                {
                    "type": "session_title_updated",
                    "session_id": self.current_session_id,
                    "title": _extract_title_from_text(trimmed),
                    "created_at": _utc_now_iso(),
                }
            )

        append_event(
            {
                "type": "message_added",
                "session_id": self.current_session_id,
                "role": "assistant",
                "text": "Phase 6에서 응답이 제공됩니다.",
                "created_at": _utc_now_iso(),
            }
        )

        return self._state()

    def install_build_engine(self) -> str:
        return "not implemented yet"

    def download_model(self) -> str:
        return "not implemented yet"


def _render_html() -> str:
    assets_dir = Path(__file__).parent / "assets"
    return (assets_dir / "index.html").read_text(encoding="utf-8")


def run_app() -> None:
    html = _render_html()

    api = AnalyzerApi()
    webview.create_window(
        "Analyzer",
        html=html,
        js_api=api,
        width=1180,
        height=760,
    )
    webview.start(debug=False)


if __name__ == "__main__":
    run_app()
