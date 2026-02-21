from __future__ import annotations

import json
from typing import Any

from .backend import BitnetClient
from .router import route

ALLOWED_INTENTS = {
    "summary",
    "schema",
    "validate",
    "preview",
    "columns",
    "plot",
    "compare",
    "filter",
    "aggregate",
}


def _trim_columns(columns: list[Any], limit: int = 80) -> list[str]:
    clean = [str(col) for col in columns if str(col)]
    if len(clean) <= limit:
        return clean
    return [*clean[:limit], "…"]


def _session_brief(session_state: dict[str, Any]) -> dict[str, Any]:
    datasets = session_state.get("datasets", [])
    active = session_state.get("active_dataset") or {}
    brief_datasets = []
    for ds in datasets:
        brief_datasets.append(
            {
                "dataset_id": ds.get("dataset_id"),
                "name": ds.get("name"),
                "kind": ds.get("kind"),
                "columns": _trim_columns(ds.get("columns", []), limit=80),
            }
        )

    brief: dict[str, Any] = {"datasets": brief_datasets}
    if active.get("dataset_id"):
        brief["active_dataset"] = {
            "dataset_id": active.get("dataset_id"),
            "name": active.get("name"),
        }
    return brief


def _build_prompt(text: str, session_state: dict[str, Any]) -> str:
    context = _session_brief(session_state)
    context_json = json.dumps(context, ensure_ascii=False)
    return (
        "너는 데이터 분석 액션 라우터다. 반드시 JSON만 출력한다. 설명 문장, 마크다운, 코드블록, 주석, 접두어를 절대 출력하지 마라. "
        "반드시 JSON만 출력한다. ``` 금지. 반드시 JSON만 출력한다.\n"
        "출력 스키마는 action 객체 배열이다. 각 객체는 다음 키를 반드시 포함한다: "
        "intent, targets, args, needs_clarification, clarify.\n"
        "intent 허용값: summary|schema|validate|preview|columns|plot|compare|filter|aggregate\n"
        "targets 스키마: {\"datasets\":[dataset_id],\"columns\":[column],\"sheets\":[]}\n"
        "clarify는 null 또는 {\"kind\":\"column|dataset\",\"question\":str,\"candidates\":[{\"id\":...,\"label\":...,\"score\":0.0}],\"context\":{}}\n"
        "예시 출력(JSON만):"
        "[{\"intent\":\"preview\",\"targets\":{\"datasets\":[\"ds1\"],\"columns\":[],\"sheets\":[]},\"args\":{\"limit\":20},\"needs_clarification\":false,\"clarify\":null}]\n"
        "규칙: 비교(compare)는 datasets 2개 이상 필요. 부족하면 needs_clarification=true와 kind=dataset으로 후보를 제시.\n"
        "규칙: plot에서 컬럼을 정하지 못하면 needs_clarification=true와 kind=column으로 후보를 제시.\n"
        f"세션 데이터셋 컨텍스트: {context_json}\n"
        f"사용자 요청: {text}\n"
        "지금 JSON만 출력하라."
    )


def _extract_json_payload(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    stack: list[str] = []
    start = -1
    for idx, ch in enumerate(text):
        if ch in "[{":
            if not stack:
                start = idx
            stack.append(ch)
        elif ch in "]}":
            if not stack:
                continue
            open_ch = stack.pop()
            if (open_ch == "[" and ch != "]") or (open_ch == "{" and ch != "}"):
                stack.clear()
                start = -1
                continue
            if not stack and start >= 0:
                candidate = text[start : idx + 1]
                return json.loads(candidate)
    return json.loads(text)


def _dataset_map(session_state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(d.get("dataset_id")): d for d in session_state.get("datasets", []) if d.get("dataset_id")}


def _build_dataset_clarification(session_state: dict[str, Any], question: str) -> dict[str, Any]:
    datasets = session_state.get("datasets", [])
    candidates = [
        {"id": d.get("dataset_id"), "label": d.get("name", d.get("dataset_id")), "score": 1.0}
        for d in datasets[:10]
    ]
    return {
        "kind": "dataset",
        "question": question,
        "candidates": candidates,
        "context": {
            "intent": "compare",
            "all_dataset_ids": [d.get("dataset_id") for d in datasets if d.get("dataset_id")],
            "all_candidates": candidates,
        },
    }


def _validate_actions(actions: Any, session_state: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(actions, dict):
        if isinstance(actions.get("actions"), list):
            actions = actions["actions"]
        else:
            actions = [actions]
    if not isinstance(actions, list) or not actions:
        return []

    datasets_by_id = _dataset_map(session_state)
    active = session_state.get("active_dataset") or {}
    active_id = active.get("dataset_id")
    validated: list[dict[str, Any]] = []

    for raw in actions[:3]:
        if not isinstance(raw, dict):
            return []
        intent = str(raw.get("intent", "")).strip().lower()
        if intent not in ALLOWED_INTENTS:
            return []

        targets = raw.get("targets") if isinstance(raw.get("targets"), dict) else {}
        ds_ids = [str(d) for d in targets.get("datasets", []) if str(d) in datasets_by_id]
        if not ds_ids and active_id and active_id in datasets_by_id:
            ds_ids = [str(active_id)]
        if not ds_ids and datasets_by_id:
            ds_ids = [next(iter(datasets_by_id.keys()))]

        columns = [str(c) for c in targets.get("columns", []) if str(c)]
        if ds_ids:
            allowed_cols: set[str] = set()
            for did in ds_ids:
                allowed_cols.update(str(c) for c in (datasets_by_id.get(did, {}).get("columns") or []))
            columns = [c for c in columns if c in allowed_cols]

        args = raw.get("args") if isinstance(raw.get("args"), dict) else {}
        needs_clarification = bool(raw.get("needs_clarification"))
        clarify = raw.get("clarify") if isinstance(raw.get("clarify"), dict) else None

        if intent == "preview":
            limit = args.get("limit", 20)
            try:
                limit = int(limit)
            except Exception:  # noqa: BLE001
                limit = 20
            args = {"limit": max(1, min(limit, 200))}

        if intent == "plot":
            mode = str(args.get("plot_mode", "auto")).strip().lower()
            if mode not in {"auto", "missing", "category", "hist", "timeseries"}:
                mode = "auto"
            top_n = args.get("top_n", 20)
            try:
                top_n = int(top_n)
            except Exception:  # noqa: BLE001
                top_n = 20
            args = {"plot_mode": mode, "top_n": max(1, min(top_n, 200)), "plot_columns": columns}
            if mode in {"category", "hist"} and not columns and not needs_clarification and ds_ids:
                heuristic = route("그래프", session_state)[0]
                if heuristic.get("needs_clarification"):
                    needs_clarification = True
                    clarify = heuristic.get("clarify")

        if intent == "aggregate":
            metric = str(args.get("metric", "mean")).strip().lower()
            if metric not in {"mean", "count", "sum"}:
                metric = "mean"
            group_by = args.get("group_by")
            if group_by is not None:
                group_by = str(group_by)
                if group_by not in columns:
                    group_by = None
            args = {"metric": metric, "group_by": group_by}

        if intent == "filter":
            op = str(args.get("op", "==")).strip()
            if op not in {"<", "<=", "==", ">=", ">"}:
                op = "=="
            args = {"op": op, "value": str(args.get("value", ""))}

        if intent == "compare" and len(ds_ids) < 2:
            needs_clarification = True
            clarify = _build_dataset_clarification(session_state, "비교할 파일을 선택해 주세요. (2개 이상 필요)")

        if intent in {"validate", "aggregate", "filter", "plot"} and not columns and not needs_clarification:
            heuristic = route("컬럼 선택", session_state)[0]
            if heuristic.get("needs_clarification"):
                needs_clarification = True
                clarify = heuristic.get("clarify")

        if needs_clarification and not clarify:
            return []

        validated.append(
            {
                "intent": intent,
                "targets": {
                    "datasets": ds_ids,
                    "columns": columns,
                    "sheets": [],
                },
                "args": args,
                "needs_clarification": needs_clarification,
                "clarify": clarify,
            }
        )

    return validated


def route_with_llm(text: str, session_state: dict, bitnet_client: BitnetClient) -> tuple[bool, list[dict], str]:
    if not text.strip():
        return False, [], "empty_text"
    if not session_state.get("datasets"):
        return False, [], "no_datasets"

    prompt = _build_prompt(text, session_state)
    ok, reply = bitnet_client.generate_raw(
        prompt=prompt,
        max_tokens=320,
        temperature=0.1,
        top_p=0.9,
        timeout_ms=45000,
        stop=[],
    )
    if not ok:
        return False, [], f"engine_error:{reply[:80]}"

    try:
        parsed = _extract_json_payload(reply)
    except Exception:  # noqa: BLE001
        return False, [], "json_parse_failed"

    actions = _validate_actions(parsed, session_state)
    if not actions:
        return False, [], "action_validation_failed"
    return True, actions, "ok"
