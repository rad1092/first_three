from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.constants import analyzer_home, ensure_dirs

HISTORY_FILENAME = "history.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def history_file_path() -> Path:
    return analyzer_home() / HISTORY_FILENAME


def ensure_history_file() -> Path:
    ensure_dirs()
    path = history_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    return path


def append_event(event: dict[str, Any]) -> None:
    path = ensure_history_file()
    line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _default_session(session_id: str, created_at: str | None = None, title: str | None = None) -> dict[str, Any]:
    now = _utc_now_iso()
    return {
        "session_id": session_id,
        "title": title or "새 채팅",
        "created_at": created_at or now,
        "last_updated_at": created_at or now,
        "deleted": False,
        "messages": [],
        "files": [],
    }


def load_sessions() -> dict[str, Any]:
    ensure_history_file()
    sessions: dict[str, dict[str, Any]] = {}
    pending_by_session: dict[str, dict[str, Any] | None] = {}

    with history_file_path().open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")
            session_id = event.get("session_id")

            if event_type == "session_created" and session_id:
                sessions.setdefault(
                    session_id,
                    _default_session(
                        session_id=session_id,
                        created_at=event.get("created_at"),
                        title=event.get("title"),
                    ),
                )
                sessions[session_id]["title"] = event.get("title") or sessions[session_id]["title"]
                sessions[session_id]["created_at"] = event.get("created_at") or sessions[session_id]["created_at"]
                sessions[session_id]["last_updated_at"] = event.get("created_at") or sessions[session_id][
                    "last_updated_at"
                ]
                continue

            if not session_id:
                continue

            session = sessions.setdefault(session_id, _default_session(session_id=session_id))

            if event_type == "message_added":
                msg = {
                    "role": event.get("role", "assistant"),
                    "text": event.get("text", ""),
                    "created_at": event.get("created_at") or _utc_now_iso(),
                }
                session["messages"].append(msg)
                session["last_updated_at"] = msg["created_at"]
            elif event_type == "files_attached":
                files = event.get("files", [])
                if isinstance(files, list):
                    session["files"].extend(files)
                session["last_updated_at"] = event.get("created_at") or session["last_updated_at"]
            elif event_type == "session_deleted":
                session["deleted"] = True
                session["deleted_at"] = event.get("deleted_at") or _utc_now_iso()
                session["last_updated_at"] = session["deleted_at"]
            elif event_type == "session_title_updated":
                new_title = event.get("title")
                if isinstance(new_title, str) and new_title.strip():
                    session["title"] = new_title.strip()
                session["last_updated_at"] = event.get("created_at") or session["last_updated_at"]
            elif event_type == "clarification_requested":
                pending_by_session[session_id] = {
                    "intent": event.get("intent"),
                    "kind": event.get("kind"),
                    "question": event.get("question"),
                    "candidates": event.get("candidates", []),
                    "context": event.get("context", {}),
                    "stage": event.get("stage", "initial"),
                }
            elif event_type in {"clarification_resolved", "clarification_cleared"}:
                pending_by_session[session_id] = None

    visible_sessions = [s for s in sessions.values() if not s.get("deleted", False)]
    visible_sessions.sort(key=lambda s: s.get("last_updated_at", ""), reverse=True)

    return {
        "sessions": [
            {
                "session_id": s["session_id"],
                "title": s["title"],
                "created_at": s["created_at"],
                "last_updated_at": s.get("last_updated_at", s["created_at"]),
                "message_count": len(s.get("messages", [])),
            }
            for s in visible_sessions
        ],
        "messages_by_session": {s["session_id"]: s.get("messages", []) for s in visible_sessions},
        "files_by_session": {s["session_id"]: s.get("files", []) for s in visible_sessions},
        "pending_by_session": {
            sid: pending_by_session.get(sid) for sid in [s["session_id"] for s in visible_sessions]
        },
    }
