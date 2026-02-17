from __future__ import annotations

import re
from typing import Any

from .normalize import find_column_candidates, normalize_text

INTENTS = [
    "summary",
    "schema",
    "filter",
    "aggregate",
    "compare",
    "plot",
    "export",
    "help",
    "columns",
    "preview",
    "validate",
]

INTENT_KEYWORDS: dict[str, list[str]] = {
    "summary": ["요약", "summary", "개요"],
    "schema": ["스키마", "schema", "타입", "구조"],
    "filter": ["필터", "조건", "where", "이상", "이하"],
    "aggregate": ["평균", "합계", "group", "집계", "건수", "count", "mean"],
    "compare": ["비교", "compare", "차이"],
    "plot": ["그래프", "시각화", "plot", "차트"],
    "export": ["저장", "내보내기", "export", "다운로드"],
    "help": ["도움", "help", "어떻게"],
    "columns": ["컬럼", "열", "column", "columns"],
    "preview": ["미리보기", "preview", "샘플", "상위"],
    "validate": ["검증", "결측", "이상치", "validate", "missing"],
}


def _detect_intent(text: str) -> str:
    norm = normalize_text(text)
    best_intent = "summary"
    best_score = 0

    for intent, kws in INTENT_KEYWORDS.items():
        score = 0
        for kw in kws:
            if normalize_text(kw) in norm:
                score += 1
        if score > best_score:
            best_score = score
            best_intent = intent

    return best_intent


def _detect_dataset_targets(text: str, session_state: dict[str, Any]) -> list[str]:
    datasets: list[dict[str, Any]] = session_state.get("datasets", [])
    active = session_state.get("active_dataset")
    t = text.lower()

    if "전체" in text or "all" in t or "첨부한" in text:
        return [d["dataset_id"] for d in datasets]

    if "csv" in t:
        return [d["dataset_id"] for d in datasets if d.get("kind") == "csv"]

    if "엑셀" in text or "xlsx" in t:
        return [d["dataset_id"] for d in datasets if d.get("kind") == "xlsx"]

    m = re.search(r"첨부한\s*(\d+)개", text)
    if m:
        n = int(m.group(1))
        return [d["dataset_id"] for d in datasets[:n]]

    if active:
        return [active["dataset_id"]]

    return [datasets[0]["dataset_id"]] if datasets else []


def _collect_columns_for_targets(target_ids: list[str], session_state: dict[str, Any]) -> list[str]:
    by_id = {d["dataset_id"]: d for d in session_state.get("datasets", [])}
    cols: list[str] = []
    for tid in target_ids:
        d = by_id.get(tid)
        if not d:
            continue
        cols.extend(d.get("columns", []))
    # unique preserve order
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def route(text: str, session_state: dict[str, Any]) -> list[dict[str, Any]]:
    intent = _detect_intent(text)
    dataset_targets = _detect_dataset_targets(text, session_state)
    all_columns = _collect_columns_for_targets(dataset_targets, session_state)
    column_candidates = find_column_candidates(text, all_columns, top_n=10)

    needs_clarification = False
    clarify = None

    if any(token in text for token in ["위도", "경도", "lat", "lon", "id", "시간", "date"]):
        if not column_candidates:
            needs_clarification = True
            clarify = {
                "question": "요청한 컬럼을 정확히 찾지 못했습니다. 아래 후보 중에서 선택해 주세요.",
                "candidates": all_columns[:10],
            }
        elif len(column_candidates) > 3:
            needs_clarification = True
            clarify = {
                "question": "컬럼 후보가 여러 개입니다. 우선 상위 3개 중 선택해 주세요.",
                "candidates": column_candidates[:3],
            }

    args: dict[str, Any] = {}
    if intent == "preview":
        args["limit"] = 20
    if intent == "aggregate":
        args["metric"] = "mean"

    action = {
        "intent": intent,
        "targets": {
            "datasets": dataset_targets,
            "sheets": [],
            "columns": column_candidates[:3] if needs_clarification and column_candidates else column_candidates,
        },
        "args": args,
        "needs_clarification": needs_clarification,
        "clarify": clarify,
    }

    return [action]
