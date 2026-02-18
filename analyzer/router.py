from __future__ import annotations

import re
from typing import Any

from .normalize import normalize_text, rank_column_candidates

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
    "compare": ["비교", "compare", "차이", "다라", "달라", "차이점"],
    "plot": ["그래프", "시각화", "plot", "차트", "시계열", "추세", "trend", "timeseries", "히스토", "분포"],
    "export": ["저장", "내보내기", "export", "다운로드"],
    "help": ["도움", "help", "어떻게"],
    "columns": ["컬럼", "column", "columns"],
    "preview": ["미리보기", "preview", "샘플", "상위"],
    "validate": ["검증", "결측", "이상치", "validate", "missing"],
}


def _detect_intent(text: str) -> str:
    norm = normalize_text(text)
    compare_terms = ["비교", "차이", "차이점", "달라", "다른점", "어떤게달라", "어떤게다른"]
    if any(normalize_text(t) in norm for t in compare_terms):
        return "compare"
    plot_terms = ["그래프", "시각화", "plot", "차트", "시계열", "추세", "trend", "timeseries", "히스토", "분포"]
    if any(normalize_text(t) in norm for t in plot_terms):
        return "plot"

    best_intent = "summary"
    best_score = 0
    for intent, kws in INTENT_KEYWORDS.items():
        score = sum(1 for kw in kws if normalize_text(kw) in norm)
        if score > best_score:
            best_score = score
            best_intent = intent
    return best_intent


def _detect_dataset_targets(text: str, session_state: dict[str, Any], intent: str) -> list[str]:
    datasets: list[dict[str, Any]] = session_state.get("datasets", [])
    active = session_state.get("active_dataset")
    t = text.lower()

    all_ids = [d["dataset_id"] for d in datasets]
    if not datasets:
        return []

    if any(k in text for k in ["둘 다", "두개 다", "두 개 다"]) or "both" in t:
        return all_ids[:2] if len(all_ids) >= 2 else all_ids

    if any(k in text for k in ["전부", "전체", "첨부한 파일 전부"]) or "all" in t:
        return all_ids

    m = re.search(r"첨부한\s*(\d+)\s*개", text)
    if m:
        n = int(m.group(1))
        return all_ids[:n]

    if "csv" in t:
        return [d["dataset_id"] for d in datasets if d.get("kind") == "csv"]
    if "엑셀" in text or "xlsx" in t:
        return [d["dataset_id"] for d in datasets if d.get("kind") == "xlsx"]

    if intent == "compare":
        return all_ids[:2] if len(all_ids) >= 2 else all_ids

    if active:
        return [active["dataset_id"]]
    return [all_ids[0]]


def _collect_columns_for_targets(target_ids: list[str], session_state: dict[str, Any]) -> list[str]:
    by_id = {d["dataset_id"]: d for d in session_state.get("datasets", [])}
    cols: list[str] = []
    for tid in target_ids:
        d = by_id.get(tid)
        if d:
            cols.extend(d.get("columns", []))

    seen = set()
    unique = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _is_column_ambiguous(ranked: list[tuple[str, float]], intent: str) -> bool:
    if len(ranked) < 2:
        return False
    top1 = ranked[0][1]
    top2 = ranked[1][1]
    if intent in {"validate", "aggregate", "filter"} and top2 >= 0.5:
        return True
    return top1 < 0.80 or (top1 - top2) < 0.12




def _detect_plot_mode(text: str) -> str:
    norm = normalize_text(text)
    if any(k in norm for k in ["결측", "결측률", "누락", "missing", "null"]):
        return "missing"
    if any(k in norm for k in ["분포", "상위", "top", "빈도", "bar", "카테고리", "범주"]):
        return "category"
    if any(k in norm for k in ["히스토그램", "히스토", "hist", "frequency"]):
        return "hist"
    if any(k in norm for k in ["시계열", "추세", "trend", "timeseries"]):
        return "timeseries"
    return "auto"


def _looks_like_plot_column_request(text: str) -> bool:
    cleaned = normalize_text(text)
    for k in ["그래프", "시각화", "plot", "차트", "분포", "top", "상위", "빈도", "bar", "히스토그램", "히스토", "hist", "시계열", "추세", "trend", "missing", "null", "결측", "결측률", "누락", "show", "보기", "보여줘", "그려줘", "top20", "top10"]:
        cleaned = cleaned.replace(normalize_text(k), "")
    cleaned = re.sub(r"[0-9\s]+", "", cleaned)
    return len(cleaned) >= 2


def _plot_column_resolution(text: str, all_columns: list[str]) -> tuple[list[str], dict[str, Any] | None, bool]:
    if not all_columns:
        return [], None, False
    ranked = rank_column_candidates(text, all_columns)
    strong = [(c, s) for c, s in ranked if s >= 0.45]
    if not strong:
        return [], None, False

    top_col, top_score = strong[0]
    top2 = strong[1][1] if len(strong) > 1 else 0.0
    explicit = top_score >= 0.75 or (top_score >= 0.6 and (top_score - top2) >= 0.15)
    ambiguous = len(strong) >= 2 and (top_score - top2) < 0.12

    if explicit and not ambiguous:
        return [top_col], None, False

    clarify = {
        "kind": "column",
        "question": "요청한 컬럼을 정확히 찾지 못했어요. 후보(1/2/3)를 골라 주세요.",
        "candidates": [{"id": col, "label": col, "score": round(score, 4)} for col, score in strong[:3]],
        "context": {
            "intent": "plot",
            "raw_query": text,
            "all_candidates": [{"id": col, "label": col, "score": round(score, 4)} for col, score in strong[:10]],
        },
    }
    return [c["id"] for c in clarify["candidates"]], clarify, True


def _compare_clarify(session_state: dict[str, Any], text: str, dataset_targets: list[str]) -> dict[str, Any] | None:
    datasets: list[dict[str, Any]] = session_state.get("datasets", [])
    if len(dataset_targets) >= 2:
        return None

    candidates = [{"id": d["dataset_id"], "label": d.get("name", d["dataset_id"]), "score": 1.0} for d in datasets[:10]]
    return {
        "kind": "dataset",
        "question": "비교할 파일을 선택해 주세요. (2개 이상 필요)",
        "candidates": candidates,
        "context": {
            "intent": "compare",
            "raw_query": text,
            "all_dataset_ids": [d["dataset_id"] for d in datasets],
            "all_candidates": candidates,
        },
    }


def route(text: str, session_state: dict[str, Any]) -> list[dict[str, Any]]:
    intent = _detect_intent(text)
    dataset_targets = _detect_dataset_targets(text, session_state, intent)

    needs_clarification = False
    clarify = None
    selected_columns: list[str] = []
    all_columns = _collect_columns_for_targets(dataset_targets, session_state)

    if intent == "compare":
        clarify = _compare_clarify(session_state, text, dataset_targets)
        needs_clarification = clarify is not None
    else:
        ranked = rank_column_candidates(text, all_columns)
        strong = [(c, s) for c, s in ranked if s >= 0.45]
        requires_column_target = intent in {"validate", "filter", "aggregate"}

        if requires_column_target and strong:
            selected_columns = [strong[0][0]]
            if _is_column_ambiguous(strong, intent):
                needs_clarification = True
                top3 = strong[:3]
                clarify = {
                    "kind": "column",
                    "question": "후보가 여러 개예요. 번호(1/2/3)로 선택해 주세요. 아니라면 '아니다'라고 입력해 주세요.",
                    "candidates": [{"id": col, "label": col, "score": round(score, 4)} for col, score in top3],
                    "context": {
                        "intent": intent,
                        "raw_query": text,
                        "dataset_id": dataset_targets[0] if dataset_targets else None,
                        "all_candidates": [{"id": col, "label": col, "score": round(score, 4)} for col, score in strong[:10]],
                    },
                }
                selected_columns = [c["id"] for c in clarify["candidates"]]
        elif requires_column_target and all_columns:
            needs_clarification = True
            clarify = {
                "kind": "column",
                "question": "요청한 컬럼을 정확히 찾지 못했어요. 관련 후보를 골라 주세요.",
                "candidates": [{"id": col, "label": col, "score": 0.0} for col in all_columns[:3]],
                "context": {
                    "intent": intent,
                    "raw_query": text,
                    "dataset_id": dataset_targets[0] if dataset_targets else None,
                    "all_candidates": [{"id": col, "label": col, "score": 0.0} for col in all_columns[:10]],
                },
            }
            selected_columns = [c["id"] for c in clarify["candidates"]]

    args: dict[str, Any] = {}
    if intent == "preview":
        args["limit"] = 20
    if intent == "aggregate":
        args["metric"] = "mean"
    if intent == "plot":
        args["plot_mode"] = _detect_plot_mode(text)
        args["top_n"] = 20
        plot_cols, plot_clarify, plot_needs = _plot_column_resolution(text, all_columns)
        args["plot_columns"] = plot_cols
        if plot_needs:
            needs_clarification = True
            clarify = plot_clarify
            selected_columns = plot_cols
        elif not plot_cols and args["plot_mode"] in {"category", "hist"} and _looks_like_plot_column_request(text) and all_columns:
            needs_clarification = True
            clarify = {
                "kind": "column",
                "question": "요청한 컬럼을 찾지 못했어요. 후보(1/2/3)를 골라 주세요.",
                "candidates": [{"id": col, "label": col, "score": 0.0} for col in all_columns[:3]],
                "context": {
                    "intent": "plot",
                    "raw_query": text,
                    "all_candidates": [{"id": col, "label": col, "score": 0.0} for col in all_columns[:10]],
                },
            }
            selected_columns = [c["id"] for c in clarify["candidates"]]

    action = {
        "intent": intent,
        "targets": {
            "datasets": dataset_targets,
            "sheets": [],
            "columns": selected_columns,
        },
        "args": args,
        "needs_clarification": needs_clarification,
        "clarify": clarify,
    }

    return [action]
