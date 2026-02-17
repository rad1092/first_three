from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import webview

from .backend import check_bitnetd_health
from .datasets import DatasetMeta, DatasetRegistry
from .history import append_event, ensure_history_file, load_sessions


@dataclass
class SessionDatasetState:
    session_id: str
    attached_files: list[dict] = field(default_factory=list)
    registry: DatasetRegistry = field(default_factory=DatasetRegistry)
    active_dataset_id: str | None = None
    hydrated: bool = False
    file_signature: tuple[str, ...] = field(default_factory=tuple)


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
        self._session_states: dict[str, SessionDatasetState] = {}

    def _state_for(self, session_id: str) -> SessionDatasetState:
        return self._session_states.setdefault(session_id, SessionDatasetState(session_id=session_id))

    def _sync_session_states_from_history(self, files_by_session: dict[str, list[dict]]) -> None:
        active_ids = set(files_by_session.keys())
        for sid in list(self._session_states.keys()):
            if sid not in active_ids:
                del self._session_states[sid]

        for sid, files in files_by_session.items():
            st = self._state_for(sid)
            signature = tuple(f"{f.get('path','')}|{f.get('size_bytes',0)}" for f in files)
            if signature != st.file_signature:
                st.attached_files = files
                st.file_signature = signature
                st.hydrated = False
                st.registry.clear()
                st.active_dataset_id = None

    def _hydrate_session_datasets(self, session_id: str) -> None:
        st = self._state_for(session_id)
        if st.hydrated:
            return

        st.registry.clear()
        metas: list[DatasetMeta] = []

        for file_ref in st.attached_files:
            p = Path(str(file_ref.get("path", "")))
            if p.exists():
                metas.extend(st.registry.register_files([str(p)]))
            else:
                kind = "xlsx" if p.suffix.lower() in {".xlsx", ".xlsm", ".xltx", ".xltm"} else "csv"
                meta = DatasetMeta(
                    dataset_id=str(uuid4()),
                    path=str(p),
                    name=file_ref.get("name") or p.name or "unknown",
                    kind=kind,
                    loaded_ok=False,
                    row_count=0,
                    col_count=0,
                    notes="파일을 찾을 수 없어 재로드에 실패했습니다. 파일 위치를 확인해 주세요.",
                )
                st.registry.add_dataset_meta(meta)
                metas.append(meta)

        if metas:
            st.active_dataset_id = metas[-1].dataset_id
            st.registry.set_active(st.active_dataset_id)
        else:
            st.active_dataset_id = None

        st.hydrated = True

    def _state(self) -> dict:
        data = load_sessions()
        sessions = data["sessions"]
        messages_by_session = data["messages_by_session"]
        files_by_session = data.get("files_by_session", {})

        self._sync_session_states_from_history(files_by_session)

        if self.current_session_id not in messages_by_session:
            self.current_session_id = sessions[0]["session_id"] if sessions else None

        datasets_state = {"datasets": [], "active_dataset": None}
        if self.current_session_id:
            self._hydrate_session_datasets(self.current_session_id)
            st = self._state_for(self.current_session_id)
            datasets_state = st.registry.as_state()

        return {
            "sessions": sessions,
            "current_session_id": self.current_session_id,
            "messages": messages_by_session.get(self.current_session_id or "", []),
            **datasets_state,
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
        if self.current_session_id:
            st = self._state_for(self.current_session_id)
            if st.registry.set_active(dataset_id):
                st.active_dataset_id = dataset_id
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
        self._session_states[session_id] = SessionDatasetState(session_id=session_id)
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
        self._session_states.pop(session_id, None)

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
        st = self._state_for(self.current_session_id)
        metas = st.registry.register_files(file_paths)

        files_event = []
        for m in metas:
            p = Path(m.path)
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            files_event.append({"path": m.path, "name": m.name, "size_bytes": size})

        st.attached_files.extend(files_event)
        st.file_signature = tuple(f"{f.get('path','')}|{f.get('size_bytes',0)}" for f in st.attached_files)
        st.hydrated = True
        if metas:
            st.active_dataset_id = metas[-1].dataset_id
            st.registry.set_active(st.active_dataset_id)

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

        append_event(
            {
                "type": "message_added",
                "session_id": self.current_session_id,
                "role": "user",
                "text": trimmed,
                "created_at": _utc_now_iso(),
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
