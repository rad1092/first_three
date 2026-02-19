from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from uuid import uuid4

import webview

from .backend import check_bitnetd_health, get_bitnet_client
from .cards import render_cards_to_html
from .datasets import DatasetMeta, DatasetRegistry
from .executor import ExecutionTrace, execute_actions
from .history import append_event, ensure_history_file, load_sessions
from .router import route


@dataclass
class SessionDatasetState:
    session_id: str
    attached_files: list[dict] = field(default_factory=list)
    registry: DatasetRegistry = field(default_factory=DatasetRegistry)
    active_dataset_id: str | None = None
    hydrated: bool = False
    file_signature: tuple[str, ...] = field(default_factory=tuple)
    traces: list[ExecutionTrace] = field(default_factory=list)
    card_trace_ids: list[str] = field(default_factory=list)


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


def _router_actions_to_text(actions: list[dict]) -> str:
    lines = ["Router 결과 (Phase 6에서 실행 예정):"]
    for idx, action in enumerate(actions, start=1):
        targets = action.get("targets", {})
        lines.append(f"{idx}) intent={action.get('intent')}")
        lines.append(f"   datasets={targets.get('datasets', [])}")
        lines.append(f"   columns={targets.get('columns', [])}")
        lines.append(f"   args={action.get('args', {})}")
        if action.get("needs_clarification"):
            clarify = action.get("clarify") or {}
            lines.append(f"   clarify={clarify.get('question', '')}")
    return "\n".join(lines)


class AnalyzerApi:
    def __init__(self) -> None:
        ensure_history_file()
        self.current_session_id: str | None = None
        self._session_states: dict[str, SessionDatasetState] = {}
        self._pending_by_session: dict[str, dict | None] = {}
        self._bitnet_client = get_bitnet_client()
        self._bitnet_client.register()
        self._chat_inflight: bool = False
        self._last_error_text: str = ""
        self._last_error_at: float = 0.0

    def _append_assistant(self, text: str) -> None:
        assert self.current_session_id is not None
        append_event(
            {
                "type": "message_added",
                "session_id": self.current_session_id,
                "role": "assistant",
                "text": text,
                "created_at": _utc_now_iso(),
            }
        )

    def _append_assistant_dedup(self, text: str, *, window_seconds: float = 5.0) -> None:
        now = monotonic()
        if text == self._last_error_text and (now - self._last_error_at) <= window_seconds:
            return
        self._append_assistant(text)
        self._last_error_text = text
        self._last_error_at = now

    def _session_runtime_state(self) -> dict:
        assert self.current_session_id is not None
        ui_state = self._state()
        st = self._state_for(self.current_session_id)
        return {
            "session_id": self.current_session_id,
            "datasets": ui_state.get("datasets", []),
            "active_dataset": ui_state.get("active_dataset"),
            "registry": st.registry,
        }

    def _execute_and_append(self, actions: list[dict]) -> None:
        assert self.current_session_id is not None
        st = self._state_for(self.current_session_id)
        cards, traces, card_trace_ids = execute_actions(actions, self._session_runtime_state())

        for trace in traces:
            st.traces.append(trace)
            if len(st.traces) > 20:
                st.traces = st.traces[-20:]
            append_event(
                {
                    "type": "execution_trace",
                    "session_id": self.current_session_id,
                    "trace_id": trace.get("trace_id"),
                    "intent": trace.get("intent"),
                    "datasets": trace.get("datasets", []),
                    "columns": trace.get("columns", []),
                    "args": trace.get("args", {}),
                    "steps": trace.get("steps", []),
                    "code": trace.get("code", ""),
                    "created_at": trace.get("created_at") or _utc_now_iso(),
                }
            )

        st.card_trace_ids.extend([t for t in card_trace_ids if t])
        if len(st.card_trace_ids) > 200:
            st.card_trace_ids = st.card_trace_ids[-200:]

        for idx, card in enumerate(cards):
            trace_id = card_trace_ids[idx] if idx < len(card_trace_ids) else None
            if not trace_id:
                continue
            meta = card.get("meta") or {}
            if isinstance(meta, dict):
                meta["trace_id"] = trace_id
                card["meta"] = meta

        self._append_assistant(render_cards_to_html(cards))

    def _state_for(self, session_id: str) -> SessionDatasetState:
        return self._session_states.setdefault(session_id, SessionDatasetState(session_id=session_id))

    def _sync_session_states_from_history(
        self, files_by_session: dict[str, list[dict]], traces_by_session: dict[str, list[dict]]
    ) -> None:
        active_ids = set(files_by_session.keys())
        for sid in list(self._session_states.keys()):
            if sid not in active_ids:
                del self._session_states[sid]

        for sid, files in files_by_session.items():
            st = self._state_for(sid)
            signature = tuple(f"{f.get('dataset_id','')}|{f.get('path','')}|{f.get('size_bytes',0)}" for f in files)
            if signature != st.file_signature:
                st.attached_files = files
                st.file_signature = signature
                st.hydrated = False
                st.registry.clear()
                st.active_dataset_id = None
                st.traces = []
                st.card_trace_ids = []

            loaded_traces = traces_by_session.get(sid, [])
            if len(loaded_traces) >= len(st.traces):
                st.traces = loaded_traces[-20:]

    def _sync_pending_from_history(self, pending_by_session: dict[str, dict | None]) -> None:
        self._pending_by_session = {k: v for k, v in pending_by_session.items()}

    def _set_pending(self, pending: dict) -> None:
        assert self.current_session_id is not None
        self._pending_by_session[self.current_session_id] = pending
        append_event(
            {
                "type": "clarification_requested",
                "session_id": self.current_session_id,
                "kind": pending.get("kind"),
                "question": pending.get("question"),
                "candidates": pending.get("candidates", []),
                "context": pending.get("context", {}),
                "intent": pending.get("intent"),
                "stage": pending.get("stage", "initial"),
                "created_at": _utc_now_iso(),
            }
        )

    def _clear_pending(self) -> None:
        assert self.current_session_id is not None
        self._pending_by_session[self.current_session_id] = None
        append_event(
            {
                "type": "clarification_cleared",
                "session_id": self.current_session_id,
                "cleared_at": _utc_now_iso(),
            }
        )

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
                    columns=[],
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

        self._sync_session_states_from_history(files_by_session, data.get("traces_by_session", {}))
        self._sync_pending_from_history(data.get("pending_by_session", {}))

        if self.current_session_id not in messages_by_session:
            self.current_session_id = sessions[0]["session_id"] if sessions else None

        datasets_state = {"datasets": [], "active_dataset": None}
        pending = None
        if self.current_session_id:
            self._hydrate_session_datasets(self.current_session_id)
            st = self._state_for(self.current_session_id)
            datasets_state = st.registry.as_state()
            pending = self._pending_by_session.get(self.current_session_id)

        return {
            "sessions": sessions,
            "current_session_id": self.current_session_id,
            "messages": messages_by_session.get(self.current_session_id or "", []),
            "pending_clarification": pending,
            **datasets_state,
        }

    def _ensure_session(self) -> None:
        if self.current_session_id:
            return
        self.new_session()

    def _parse_choice(self, text: str, max_n: int) -> int | None:
        m = re.search(r"(\d+)", text)
        if not m:
            return None
        n = int(m.group(1))
        return n if 1 <= n <= max_n else None

    def _is_reject(self, text: str) -> bool:
        t = text.strip().lower()
        return t in {"아니다", "다른거", "none", "no", "아니요", "x"}

    def _is_yes(self, text: str) -> bool:
        return text.strip().lower() in {"y", "yes", "응", "네", "예"}

    def _is_no(self, text: str) -> bool:
        return text.strip().lower() in {"n", "no", "아니", "아니요"}

    def _resolve_pending_with_selection(self, pending: dict, candidate: dict) -> None:
        assert self.current_session_id is not None
        context = pending.get("context", {})
        candidate_id = candidate.get("id")
        if isinstance(candidate_id, list):
            columns = [str(c) for c in candidate_id if str(c)]
        elif isinstance(candidate_id, str) and candidate_id:
            columns = [candidate_id]
        else:
            columns = []

        dataset_targets: list[str] = []
        if pending.get("kind") == "dataset":
            if isinstance(candidate_id, list):
                dataset_targets = [str(v) for v in candidate_id if str(v)]
            elif isinstance(candidate_id, str) and candidate_id:
                dataset_targets = [candidate_id]
            all_ids = context.get("all_dataset_ids") or []
            if len(dataset_targets) == 1 and len(all_ids) >= 2:
                # compare 확정 시 단일 선택이면 해당 파일 + 다른 첫 파일로 보정
                other = [d for d in all_ids if d != dataset_targets[0]]
                if other:
                    dataset_targets.append(other[0])

        action = {
            "intent": context.get("intent") or pending.get("intent") or "summary",
            "targets": {
                "datasets": dataset_targets or ([context.get("dataset_id")] if context.get("dataset_id") else []),
                "sheets": [],
                "columns": [] if pending.get("kind") == "dataset" else columns,
            },
            "args": {},
            "needs_clarification": False,
            "clarify": None,
        }
        append_event(
            {
                "type": "clarification_resolved",
                "session_id": self.current_session_id,
                "selected": candidate,
                "resolved_at": _utc_now_iso(),
            }
        )
        self._clear_pending()
        self._execute_and_append([action])

    def get_initial_state(self) -> dict:
        ok, status_text = check_bitnetd_health()
        return {
            "connection_ok": ok,
            "connection_text": status_text,
            **self._state(),
        }

    def list_sessions(self) -> dict:
        return self._state()

    def get_connection_status(self) -> dict:
        ok, status_text = check_bitnetd_health()
        return {
            "connection_ok": ok,
            "connection_text": status_text,
        }

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

        append_event(
            {
                "type": "session_created",
                "session_id": session_id,
                "title": f"새 채팅 {next_index}",
                "created_at": _utc_now_iso(),
            }
        )

        self.current_session_id = session_id
        self._session_states[session_id] = SessionDatasetState(session_id=session_id)
        self._pending_by_session[session_id] = None
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
        self._pending_by_session.pop(session_id, None)

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
            files_event.append({"dataset_id": m.dataset_id, "path": m.path, "name": m.name, "size_bytes": size})

        st.attached_files.extend(files_event)
        st.file_signature = tuple(f"{f.get('dataset_id','')}|{f.get('path','')}|{f.get('size_bytes',0)}" for f in st.attached_files)
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
        self._append_assistant(_dataset_summary_lines(metas))
        return self._state()

    def detach_dataset(self, dataset_id: str) -> dict:
        self._ensure_session()
        assert self.current_session_id is not None

        st = self._state_for(self.current_session_id)
        meta = st.registry.get_dataset(dataset_id)
        if not meta:
            return self._state()

        st.registry.remove_dataset(dataset_id)
        st.attached_files = [f for f in st.attached_files if f.get("dataset_id") != dataset_id and f.get("path") != meta.path]
        st.file_signature = tuple(
            f"{f.get('dataset_id','')}|{f.get('path','')}|{f.get('size_bytes',0)}" for f in st.attached_files
        )
        st.hydrated = True

        active = st.registry.get_active()
        st.active_dataset_id = active.dataset_id if active else None

        append_event(
            {
                "type": "files_detached",
                "session_id": self.current_session_id,
                "dataset_id": dataset_id,
                "path": meta.path,
                "name": meta.name,
                "detached_at": _utc_now_iso(),
            }
        )
        self._append_assistant(f"세션에서 첨부 제거: {meta.name}")
        return self._state()

    def _extract_trace_request(self, text: str) -> tuple[str, int | None] | None:
        t = text.strip().lower()
        if not t:
            return None
        code_match = re.search(r"(\d+)번\s*(코드|로직)", t)
        if code_match:
            return code_match.group(2), int(code_match.group(1))
        if any(k in t for k in ["코드 보여줘", "방금 코드"]):
            return "코드", None
        if any(k in t for k in ["로직 보여줘", "방금 로직", "방금 로직 보여줘"]):
            return "로직", None
        return None

    def _find_trace_by_number(self, n: int | None) -> ExecutionTrace | None:
        assert self.current_session_id is not None
        st = self._state_for(self.current_session_id)
        if not st.traces:
            return None
        if n is None:
            return st.traces[-1]

        if 1 <= n <= len(st.card_trace_ids):
            trace_id = st.card_trace_ids[n - 1]
            for trace in reversed(st.traces):
                if trace.get("trace_id") == trace_id:
                    return trace

        if 1 <= n <= len(st.traces):
            return st.traces[n - 1]
        return None

    def _render_trace_card_text(self, trace: ExecutionTrace) -> str:
        steps = trace.get("steps", [])
        code = str(trace.get("code", "")).strip()
        lines = ["[로직]"]
        for idx, step in enumerate(steps, start=1):
            lines.append(f"{idx}. {step}")
        lines.append("")
        lines.append("[재현 코드]")
        lines.append("```python")
        lines.append(code)
        lines.append("```")
        return "\n".join(lines)



    def _is_cancel_pending(self, text: str) -> bool:
        return text.strip().lower() in {"취소", "cancel", "그만", "나가기"}

    def _has_analysis_keyword(self, text: str) -> bool:
        t = text.lower()
        keywords = [
            "요약", "스키마", "미리보기", "상위", "중복", "결측", "검증", "비교", "그래프", "차트", "시각화",
            "filter", "aggregate", "compare", "plot",
        ]
        return any(k in t for k in keywords)

    def _is_numeric_only_choice(self, text: str) -> bool:
        return re.fullmatch(r"\s*\d+\s*", text) is not None

    def _is_new_question_while_pending(self, text: str) -> bool:
        if self._is_numeric_only_choice(text):
            return False
        if self._is_yes(text) or self._is_no(text) or self._is_reject(text):
            return False
        return self._has_analysis_keyword(text)

    def _build_dataset_missing_prompt(self, user_text: str) -> str:
        return (
            "반드시 한국어만 사용하세요. 영어 단어/문장 금지. "
            "코드, 마크다운, ``` , def , import , JSON 출력 금지. "
            "AI:, System:, User: 같은 접두 금지. "
            "총 2~4문장(최대 6문장)으로만 답하세요. "
            "첫 문장은 파일 첨부가 필요한 이유를 1문장으로 설명하고, 이어서 1) 2) 3) 형식으로 다음 행동을 제시하세요. "
            f"사용자 입력: {user_text}"
        )

    def _build_general_chat_prompt(self, user_text: str) -> str:
        return (
            "반드시 한국어만 사용하세요. 영어 단어/문장 금지. "
            "코드, 마크다운, ``` , def , import , JSON 출력 금지. "
            "AI:, System:, User: 같은 접두 금지. "
            "짧고 친절하게 2~4문장(최대 6문장)으로만 답하세요. "
            "불필요한 예시/목록/반복 없이, 마지막 문장은 다음 질문을 자연스럽게 유도하세요. "
            f"사용자 입력: {user_text}"
        )

    def _respond_via_bitnet_chat(self, *, user_text: str, guidance_mode: bool) -> bool:
        prompt = (
            self._build_dataset_missing_prompt(user_text)
            if guidance_mode
            else self._build_general_chat_prompt(user_text)
        )
        ok, reply = self._bitnet_client.generate_text(
            prompt=prompt,
            max_tokens=72,
            temperature=0.35 if guidance_mode else 0.45,
            top_p=0.85,
            timeout_ms=90000,
            stop=[
                "```",
                "```python",
                "def ",
                "import ",
                "OUTPUT",
                "desired_result",
                "System:",
                "User:",
                "AI:",
            ],
        )
        if ok:
            self._append_assistant(reply)
            self._last_error_text = ""
            self._last_error_at = 0.0
        else:
            self._append_assistant_dedup(reply)
        return ok

    def _route_and_respond(self, trimmed: str) -> dict:
        session_state = self._state()
        actions = route(trimmed, session_state)
        first = actions[0]
        if first.get("needs_clarification"):
            clarify = first.get("clarify") or {}
            pending_new = {
                "intent": first.get("intent"),
                "kind": clarify.get("kind", "column"),
                "question": clarify.get("question", "대상을 선택해 주세요."),
                "candidates": clarify.get("candidates", []),
                "context": clarify.get("context", {}),
                "stage": "initial",
            }
            self._set_pending(pending_new)
            labels = [f"{i+1}. {c['label']} (score={c.get('score',0)})" for i, c in enumerate(pending_new["candidates"])]
            self._append_assistant(pending_new["question"] + "\n" + "\n".join(labels) + "\n1/2/3으로 선택하거나 '아니다'라고 입력해 주세요.")
            return self._state()

        self._execute_and_append(actions)
        return self._state()

    def send_message(self, text: str) -> dict:
        trimmed = text.strip()
        if not trimmed:
            return self._state()

        self._ensure_session()
        assert self.current_session_id is not None

        if self._chat_inflight:
            return self._state()

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

        trace_req = self._extract_trace_request(trimmed)
        if trace_req:
            _, number = trace_req
            trace = self._find_trace_by_number(number)
            if not trace:
                self._append_assistant("코드/로직을 보여줄 실행 이력이 없습니다. 먼저 요약/검증/비교/plot을 실행해 주세요.")
                return self._state()
            text_card = render_cards_to_html([
                {
                    "type": "text",
                    "title": f"코드/로직 보기 · intent={trace.get('intent', 'unknown')}",
                    "text": self._render_trace_card_text(trace),
                    "meta": {"trace_id": trace.get("trace_id")},
                }
            ])
            self._append_assistant(text_card)
            return self._state()

        pending = self._pending_by_session.get(self.current_session_id)
        if pending:
            if self._is_cancel_pending(trimmed):
                self._clear_pending()
                self._append_assistant("좋아요. 방금 선택 단계는 취소했어요. 새 질문을 입력해 주세요.")
                return self._state()
            if self._is_new_question_while_pending(trimmed):
                self._clear_pending()
                self._append_assistant("이전 선택 단계를 취소하고 새 질문으로 넘어갈게요.")
                return self._route_and_respond(trimmed)

            stage = pending.get("stage", "initial")
            candidates = pending.get("candidates", [])

            if stage in {"initial", "choose10"}:
                choice = self._parse_choice(trimmed, len(candidates))
                if choice:
                    self._resolve_pending_with_selection(pending, candidates[choice - 1])
                    return self._state()
                if self._is_reject(trimmed):
                    if pending.get("kind") == "dataset":
                        self._append_assistant("비교 대상은 번호로 선택해 주세요. (예: 1 또는 2)")
                        return self._state()
                    menu = {
                        **pending,
                        "stage": "second_menu",
                        "question": "2차 좁히기: 1) 키워드 더 정확히 2) 관련 후보 10개 3) 유사 전부로 진행(y/n)",
                    }
                    self._set_pending(menu)
                    self._append_assistant(menu["question"])
                    return self._state()

            elif stage == "second_menu":
                choice = self._parse_choice(trimmed, 3)
                if choice == 1:
                    nxt = {**pending, "stage": "keyword", "question": "키워드를 더 정확히 입력해 주세요. 예: '위도(좌표)'"}
                    self._set_pending(nxt)
                    self._append_assistant(nxt["question"])
                    return self._state()
                if choice == 2:
                    all_candidates = pending.get("context", {}).get("all_candidates", [])[:10]
                    nxt = {**pending, "stage": "choose10", "candidates": all_candidates, "question": "관련 후보 10개입니다. 번호로 선택해 주세요."}
                    self._set_pending(nxt)
                    labels = [f"{i+1}. {c['label']}" for i, c in enumerate(all_candidates)]
                    self._append_assistant(nxt["question"] + "\n" + "\n".join(labels))
                    return self._state()
                if choice == 3:
                    nxt = {**pending, "stage": "all_confirm", "question": "유사 전부로 진행할까요? (y/n)"}
                    self._set_pending(nxt)
                    self._append_assistant(nxt["question"])
                    return self._state()

            elif stage == "all_confirm":
                if self._is_yes(trimmed):
                    all_candidates = pending.get("context", {}).get("all_candidates", [])
                    selected_ids = [c.get("id") for c in all_candidates if c.get("id")]
                    selected = {
                        "id": selected_ids,
                        "label": "유사 후보 전체",
                        "score": 1.0,
                    }
                    self._resolve_pending_with_selection(pending, selected)
                    return self._state()
                if self._is_no(trimmed):
                    self._clear_pending()
                    self._append_assistant(
                        "좋아요. 새 질문으로 좁혀볼게요. 예시: 1) 위도 컬럼 결측치 확인 2) 시도별 평균 인구 3) timestamp 기준 추세"
                    )
                    return self._state()

            elif stage == "keyword":
                session_state = self._state()
                actions = route(trimmed, session_state)
                a0 = actions[0]
                if a0.get("needs_clarification"):
                    clarify = a0.get("clarify")
                    pending2 = {
                        "intent": a0.get("intent"),
                        "kind": clarify.get("kind"),
                        "question": clarify.get("question"),
                        "candidates": clarify.get("candidates", []),
                        "context": clarify.get("context", {}),
                        "stage": "initial",
                    }
                    self._set_pending(pending2)
                    labels = [f"{i+1}. {c['label']}" for i, c in enumerate(pending2["candidates"])]
                    self._append_assistant(pending2["question"] + "\n" + "\n".join(labels) + "\n(아니면 '아니다')")
                    return self._state()
                self._clear_pending()
                self._execute_and_append(actions)
                return self._state()

            self._append_assistant(
                "아직 확정하지 못했어요. 번호(1/2/3)로 고르거나 '아니다'라고 입력해 주세요.\n"
                "필요하면 새 질문 예시: 1) 위도 컬럼 결측 2) 시도별 평균 3) 상위 20행 미리보기"
            )
            return self._state()

        session_state = self._state()
        if session_state.get("active_dataset") is None:
            self._chat_inflight = True
            try:
                self._respond_via_bitnet_chat(
                    user_text=trimmed,
                    guidance_mode=self._has_analysis_keyword(trimmed),
                )
                return self._state()
            finally:
                self._chat_inflight = False

        actions = route(trimmed, session_state)
        first = actions[0]
        if first.get("needs_clarification"):
            clarify = first.get("clarify") or {}
            pending_new = {
                "intent": first.get("intent"),
                "kind": clarify.get("kind", "column"),
                "question": clarify.get("question", "대상을 선택해 주세요."),
                "candidates": clarify.get("candidates", []),
                "context": clarify.get("context", {}),
                "stage": "initial",
            }
            self._set_pending(pending_new)
            labels = [f"{i+1}. {c['label']} (score={c.get('score',0)})" for i, c in enumerate(pending_new["candidates"])]
            self._append_assistant(pending_new["question"] + "\n" + "\n".join(labels) + "\n1/2/3으로 선택하거나 '아니다'라고 입력해 주세요.")
            return self._state()

        self._execute_and_append(actions)
        return self._state()

    def install_build_engine(self) -> str:
        return "not implemented yet"

    def download_model(self) -> str:
        return "not implemented yet"


def _assets_index_uri() -> str:
    return (Path(__file__).parent / "assets" / "index.html").resolve().as_uri()


def run_app() -> None:
    api = AnalyzerApi()
    webview.create_window(
        "Analyzer",
        url=_assets_index_uri(),
        js_api=api,
        width=1180,
        height=760,
    )
    debug = os.getenv("ANALYZER_WEBVIEW_DEBUG", "0") == "1"
    try:
        webview.start(debug=debug, http_server=True)
    except TypeError:
        webview.start(debug=debug)


if __name__ == "__main__":
    run_app()
