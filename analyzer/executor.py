from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

try:
    import pandas as pd
except Exception:  # noqa: BLE001
    pd = None

from .cards import Card, make_chart_card, make_table_card, make_text_card
from .normalize import normalize_text
from .plotter import create_plot_cards

LAT_NAMES = {normalize_text(v) for v in ["lat", "latitude", "위도", "gps_lat"]}
LON_NAMES = {normalize_text(v) for v in ["lon", "lng", "longitude", "경도", "gps_lon"]}
SUPPORTED_FANOUT_INTENTS = {"summary", "validate", "schema", "preview", "columns", "plot"}


ExecutionTrace = dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_pandas() -> str | None:
    if pd is None:
        return "pandas가 설치되지 않아 실행 엔진을 사용할 수 없습니다."
    return None


def _resolve_dataset_ids(action: dict[str, Any], session_state: dict[str, Any]) -> list[str]:
    targets = action.get("targets", {})
    ids = [d for d in targets.get("datasets", []) if d]
    if ids:
        return ids
    active = session_state.get("active_dataset") or {}
    active_id = active.get("dataset_id")
    if active_id:
        return [active_id]
    datasets = session_state.get("datasets", [])
    return [datasets[0]["dataset_id"]] if datasets else []


def _dataset_bundle(dataset_id: str, session_state: dict[str, Any]) -> tuple[str, Any] | None:
    registry = session_state.get("registry")
    metas = {d["dataset_id"]: d for d in session_state.get("datasets", [])}
    if not registry:
        return None
    df = registry.get_frame(dataset_id)
    if df is None:
        return None
    name = str((metas.get(dataset_id) or {}).get("name") or dataset_id)
    return name, df


def _title(dataset_name: str, suffix: str) -> str:
    return f"[{dataset_name}] {suffix}"


def _columns_for_action(action: dict[str, Any], df: Any) -> tuple[list[str], list[str]]:
    target_cols = [str(c) for c in action.get("targets", {}).get("columns", []) if c]
    if not target_cols:
        return df.columns.tolist(), []
    existing = [c for c in target_cols if c in df.columns]
    missing = [c for c in target_cols if c not in df.columns]
    return existing, missing


def _missing_ratio_table(df: Any, top_n: int = 20) -> Any:
    if df.empty:
        return pd.DataFrame(columns=["column", "missing_count", "missing_rate(%)"])
    missing_count = df.isna().sum()
    missing_ratio = (missing_count / max(len(df), 1) * 100).round(2)
    table = pd.DataFrame(
        {
            "column": missing_count.index.astype(str),
            "missing_count": missing_count.values.astype(int),
            "missing_rate(%)": missing_ratio.values,
        }
    )
    return table.sort_values(["missing_rate(%)", "missing_count"], ascending=[False, False]).head(top_n).reset_index(drop=True)


def _schema_table(df: Any) -> Any:
    rows = []
    for col in df.columns:
        ser = df[col]
        nn = ser.dropna()
        rows.append(
            {
                "column": str(col),
                "dtype": str(ser.dtype),
                "non_null": int(ser.notna().sum()),
                "null_count": int(ser.isna().sum()),
                "sample": "" if nn.empty else str(nn.iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _numeric_describe_table(df: Any) -> Any:
    num = df.select_dtypes(include=["number"])
    if num.empty:
        return pd.DataFrame(columns=["column", "count", "mean", "std", "min", "max"])
    desc = num.describe().T.reset_index().rename(columns={"index": "column"})
    use_cols = [c for c in ["column", "count", "mean", "std", "min", "max"] if c in desc.columns]
    return desc[use_cols].round(4)


def _validate_geo(df: Any) -> list[str]:
    lat_col = None
    lon_col = None
    for c in df.columns:
        norm = normalize_text(str(c))
        if lat_col is None and norm in LAT_NAMES:
            lat_col = c
        if lon_col is None and norm in LON_NAMES:
            lon_col = c

    notes: list[str] = []
    if lat_col is not None:
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        notes.append(f"위도 범위 위반(-90~90): {int(((lat < -90) | (lat > 90)).fillna(False).sum())}건")
    if lon_col is not None:
        lon = pd.to_numeric(df[lon_col], errors="coerce")
        notes.append(f"경도 범위 위반(-180~180): {int(((lon < -180) | (lon > 180)).fillna(False).sum())}건")
    return notes


def _execute_for_df(action: dict[str, Any], dataset_name: str, df: Any) -> list[Card]:
    intent = action.get("intent", "summary")
    cards: list[Card] = []
    cols, missing_targets = _columns_for_action(action, df)

    if action.get("targets", {}).get("columns") and not cols:
        wanted = ", ".join(action.get("targets", {}).get("columns", []))
        cards.append(make_text_card(_title(dataset_name, "컬럼 안내"), f"요청한 컬럼({wanted})이 파일에 없습니다."))
        return cards

    selected_df = df[cols] if cols and len(cols) < len(df.columns) else df
    if missing_targets:
        cards.append(make_text_card(_title(dataset_name, "컬럼 안내"), f"일부 컬럼이 없어 제외했습니다: {', '.join(missing_targets)}"))

    if intent == "columns":
        all_cols = [str(c) for c in df.columns.tolist()]
        msg = "\n".join(f"- {c}" for c in all_cols[:50])
        if len(all_cols) > 50:
            msg += "\n... 더 보려면 검색어를 더 구체화해 주세요."
        cards.append(make_text_card(_title(dataset_name, "컬럼"), msg, {"total_columns": len(all_cols)}))
        return cards

    if intent == "preview":
        limit = int(action.get("args", {}).get("limit", 20))
        cards.append(make_table_card(_title(dataset_name, f"상위 {limit}행"), selected_df.head(limit)))
        return cards

    if intent == "schema":
        cards.append(make_table_card(_title(dataset_name, "스키마"), _schema_table(selected_df)))
        return cards

    if intent == "summary":
        rows, cols_n = selected_df.shape
        cards.append(
            make_text_card(
                _title(dataset_name, "요약"),
                "\n".join(
                    [
                        f"행: {rows:,}",
                        f"열: {cols_n:,}",
                        f"중복 행: {int(selected_df.duplicated().sum()):,}",
                        f"숫자형 컬럼 수: {selected_df.select_dtypes(include=['number']).shape[1]:,}",
                    ]
                ),
            )
        )
        cards.append(make_table_card(_title(dataset_name, "결측 Top20"), _missing_ratio_table(selected_df, top_n=20)))
        desc = _numeric_describe_table(selected_df)
        if not desc.empty:
            cards.append(make_table_card(_title(dataset_name, "숫자형 통계"), desc.head(20)))
        return cards

    if intent == "validate":
        notes = [f"중복 행 수: {int(selected_df.duplicated().sum()):,}"]
        notes.extend(_validate_geo(selected_df))
        if len(notes) == 1:
            notes.append("좌표(위도/경도) 컬럼이 없어 범위 검증은 생략했습니다.")
        cards.append(make_text_card(_title(dataset_name, "검증"), "\n".join(notes)))
        cards.append(make_table_card(_title(dataset_name, "결측률 Top20"), _missing_ratio_table(selected_df, top_n=20)))
        return cards

    cards.append(make_text_card("미지원 intent", f"intent={intent}는 아직 미지원입니다."))
    return cards


def _compare_schema(dataset_frames: list[tuple[str, Any]]) -> list[Card]:
    if len(dataset_frames) < 2:
        return [make_text_card("비교 안내", "비교를 위해서는 2개 이상의 파일이 필요합니다.")]

    cols_by_name = {name: {str(c) for c in df.columns.tolist()} for name, df in dataset_frames}
    names = list(cols_by_name.keys())
    common = set.intersection(*(cols_by_name[n] for n in names)) if names else set()

    rows = []
    all_cols = sorted(set.union(*(cols_by_name[n] for n in names)))
    for col in all_cols:
        row = {"column": col}
        for n in names:
            row[n] = "Y" if col in cols_by_name[n] else ""
        rows.append(row)

    only_rows = []
    for n in names:
        only_rows.append({"dataset": n, "only_columns": ", ".join(sorted(cols_by_name[n] - common)) or "-"})

    cards: list[Card] = [
        make_text_card(
            "[비교] 스키마 요약",
            f"공통 컬럼 수: {len(common)}\n공통 컬럼(최대 30): {', '.join(sorted(common)[:30]) or '-'}",
        ),
        make_table_card("[비교] 스키마 Presence Matrix", pd.DataFrame(rows).head(200)),
        make_table_card("[비교] 파일별 고유 컬럼", pd.DataFrame(only_rows)),
    ]
    return cards


def _compare_missingness(dataset_frames: list[tuple[str, Any]], top_n: int = 20) -> list[Card]:
    per_file_rows = []
    miss_map: dict[str, dict[str, float]] = {}
    for name, df in dataset_frames:
        miss = (df.isna().sum() / max(len(df), 1) * 100).round(2)
        miss_map[name] = {str(k): float(v) for k, v in miss.to_dict().items()}
        top = miss.sort_values(ascending=False).head(top_n)
        for col, rate in top.items():
            per_file_rows.append({"dataset": name, "column": str(col), "missing_rate(%)": float(rate)})

    common_cols = sorted(set.intersection(*(set(m.keys()) for m in miss_map.values()))) if miss_map else []
    common_rows = []
    for col in common_cols:
        row = {"column": col}
        vals = [miss_map[name].get(col, 0.0) for name, _ in dataset_frames]
        for name, _ in dataset_frames:
            row[f"{name} missing(%)"] = miss_map[name].get(col, 0.0)
        row["max_missing(%)"] = max(vals) if vals else 0.0
        row["spread(%)"] = (max(vals) - min(vals)) if vals else 0.0
        common_rows.append(row)

    common_df = pd.DataFrame(common_rows)
    if not common_df.empty:
        common_df = common_df.sort_values(["max_missing(%)", "spread(%)"], ascending=[False, False]).head(top_n)

    return [
        make_table_card("[비교] 파일별 결측률 Top20", pd.DataFrame(per_file_rows)),
        make_table_card("[비교] 공통 컬럼 결측률 비교", common_df if not common_df.empty else pd.DataFrame(columns=["column"])),
    ]


def _compare_numeric(dataset_frames: list[tuple[str, Any]], max_cols: int = 20) -> list[Card]:
    numeric_sets = []
    stats_by_dataset: dict[str, Any] = {}
    for name, df in dataset_frames:
        num = df.select_dtypes(include=["number"])
        stats_by_dataset[name] = num
        numeric_sets.append(set(num.columns.tolist()))

    if not numeric_sets:
        return [make_text_card("[비교] 숫자형 비교", "숫자형 컬럼이 없습니다.")]

    common = sorted(set.intersection(*numeric_sets)) if numeric_sets else []
    if not common:
        return [make_text_card("[비교] 숫자형 비교", "공통 숫자형 컬럼이 없어 비교를 생략했습니다.")]

    rows = []
    target_cols = common[:max_cols]
    for col in target_cols:
        for name, _ in dataset_frames:
            ser = pd.to_numeric(stats_by_dataset[name][col], errors="coerce")
            rows.append(
                {
                    "column": str(col),
                    "dataset": name,
                    "count": int(ser.notna().sum()),
                    "mean": round(float(ser.mean()), 4) if ser.notna().any() else None,
                    "min": round(float(ser.min()), 4) if ser.notna().any() else None,
                    "max": round(float(ser.max()), 4) if ser.notna().any() else None,
                }
            )

    text = ""
    if len(common) > max_cols:
        text = f"공통 숫자형 컬럼 {len(common)}개 중 상위 {max_cols}개만 표시했습니다."

    cards: list[Card] = []
    if text:
        cards.append(make_text_card("[비교] 숫자형 통계 안내", text))
    cards.append(make_table_card("[비교] 공통 숫자형 통계 비교", pd.DataFrame(rows)))
    return cards


def _execute_compare(action: dict[str, Any], session_state: dict[str, Any]) -> list[Card]:
    dataset_ids = _resolve_dataset_ids(action, session_state)
    dataset_frames: list[tuple[str, Any]] = []
    for did in dataset_ids:
        bundle = _dataset_bundle(did, session_state)
        if bundle:
            dataset_frames.append(bundle)

    if len(dataset_frames) < 2:
        return [make_text_card("비교 안내", "비교할 파일을 2개 이상 선택해 주세요.")]

    cards: list[Card] = [make_text_card("[비교] 실행 요약", "\n".join([f"- {name}" for name, _ in dataset_frames]))]
    cards.extend(_compare_schema(dataset_frames))
    cards.extend(_compare_missingness(dataset_frames, top_n=20))
    cards.extend(_compare_numeric(dataset_frames, max_cols=20))
    return cards


def _execute_plot(action: dict[str, Any], session_state: dict[str, Any]) -> list[Card]:
    dataset_ids = _resolve_dataset_ids(action, session_state)
    if not dataset_ids:
        return [make_text_card("실행 실패", "시각화할 데이터셋이 없습니다. 파일을 먼저 첨부해 주세요.")]

    cards: list[Card] = []
    session_id = session_state.get("session_id")
    selected_cols = action.get("targets", {}).get("columns", [])
    plot_mode = str(action.get("args", {}).get("plot_mode", "auto"))

    for did in dataset_ids:
        bundle = _dataset_bundle(did, session_state)
        if not bundle:
            cards.append(make_text_card("실행 실패", f"데이터프레임 로드 실패: {did}"))
            continue
        name, df = bundle
        plot_cards = create_plot_cards(
            df=df,
            dataset_name=name,
            session_id=session_id,
            args={"top_n": 20, "columns": selected_cols, "mode": plot_mode},
        )
        if not plot_cards:
            cards.append(make_text_card(f"[{name}] 시각화 안내", "생성 가능한 차트를 찾지 못했습니다."))
            continue
        for c in plot_cards:
            if c.get("type") == "text":
                cards.append(make_text_card(c.get("title", f"[{name}] 시각화 안내"), c.get("text", ""), c.get("meta", {})))
            else:
                cards.append(make_chart_card(c.get("title", f"[{name}] 차트"), c.get("image_data_uri", ""), c.get("meta", {})))

    if len(dataset_ids) >= 2 and len(cards) > 8:
        cards = cards[:8] + [make_text_card("시각화 안내", "차트가 많아 상위 8개만 표시했습니다.")]

    return cards


def _code_lines_for_intent(intent: str, dataset_ids: list[str], columns: list[str], args: dict[str, Any]) -> list[str]:
    did = dataset_ids[0] if dataset_ids else "<dataset_id>"
    lines = [
        "import pandas as pd",
        "",
        "# 앱 내부: dataset_id -> DataFrame",
        f"df = registry.get_df('{did}')",
    ]
    if columns:
        lines.append(f"target_cols = {columns!r}")
        lines.append("existing_cols = [c for c in target_cols if c in df.columns]")
        lines.append("df = df[existing_cols] if existing_cols else df")

    if intent == "summary":
        lines.extend(
            [
                "rows, cols = df.shape",
                "dup_count = int(df.duplicated().sum())",
                "missing_rate = (df.isna().mean() * 100).round(2)",
                "missing_top20 = missing_rate.sort_values(ascending=False).head(20)",
                "numeric_summary = df.select_dtypes(include=['number']).describe().T",
            ]
        )
    elif intent == "validate":
        lines.extend(
            [
                "dup_count = int(df.duplicated().sum())",
                "missing_rate = (df.isna().mean() * 100).round(2)",
                "missing_top20 = missing_rate.sort_values(ascending=False).head(20)",
                "lat_series = pd.to_numeric(df.get('lat', pd.Series(dtype='float')), errors='coerce')",
                "lon_series = pd.to_numeric(df.get('lon', pd.Series(dtype='float')), errors='coerce')",
                "lat_out_of_range = int(((lat_series < -90) | (lat_series > 90)).fillna(False).sum())",
                "lon_out_of_range = int(((lon_series < -180) | (lon_series > 180)).fillna(False).sum())",
            ]
        )
    elif intent == "compare":
        lines = [
            "import pandas as pd",
            "",
            "# 앱 내부: dataset_id -> DataFrame",
            f"dataset_ids = {dataset_ids!r}",
            "frames = {did: registry.get_df(did) for did in dataset_ids}",
            "columns_by_dataset = {did: set(df.columns.astype(str)) for did, df in frames.items()}",
            "common_columns = set.intersection(*columns_by_dataset.values()) if columns_by_dataset else set()",
            "schema_diff = {did: sorted(cols - common_columns) for did, cols in columns_by_dataset.items()}",
            "missing_rate = {did: (df.isna().mean() * 100).round(2) for did, df in frames.items()}",
            "# 값 분포 비교 예시",
            "value_dist = {did: df.iloc[:, 0].value_counts(normalize=True).head(10) for did, df in frames.items() if not df.empty}",
        ]
    elif intent == "plot":
        lines.extend(
            [
                "numeric_cols = df.select_dtypes(include=['number']).columns.tolist()",
                "if numeric_cols:",
                "    ax = df[numeric_cols[0]].dropna().plot(kind='hist', bins=20, title='Auto Plot')",
                "    fig = ax.get_figure()",
                "    fig.tight_layout()",
            ]
        )
    elif intent == "schema":
        lines.extend(
            [
                "schema = pd.DataFrame({",
                "    'column': df.columns.astype(str),",
                "    'dtype': [str(df[c].dtype) for c in df.columns],",
                "    'non_null': [int(df[c].notna().sum()) for c in df.columns],",
                "    'null_count': [int(df[c].isna().sum()) for c in df.columns],",
                "})",
            ]
        )
    elif intent == "preview":
        limit = int(args.get("limit", 20))
        lines.append(f"preview_df = df.head({limit})")
    elif intent == "columns":
        lines.append("all_columns = df.columns.astype(str).tolist()")

    return lines


def _steps_for_intent(intent: str, dataset_names: list[str], columns: list[str]) -> list[str]:
    steps = [f"대상 데이터셋 확인: {', '.join(dataset_names) if dataset_names else '없음'}"]
    if columns:
        steps.append(f"요청 컬럼 필터 적용: {', '.join(columns)}")
    else:
        steps.append("요청 컬럼이 없어서 전체 컬럼 사용")

    intent_step = {
        "summary": "기본 요약(행/열/중복), 결측률, 숫자형 통계 계산",
        "validate": "중복/결측률과 좌표 범위(위도/경도) 검증 수행",
        "compare": "파일 간 스키마(set 비교), 결측률, 숫자형 기초통계 비교",
        "plot": "숫자형 중심 자동 차트 생성",
        "schema": "컬럼 dtype/non-null/null/sample 기반 스키마 테이블 생성",
        "preview": "상위 N행 미리보기 생성",
        "columns": "전체 컬럼 목록 생성",
    }
    steps.append(intent_step.get(intent, f"intent={intent} 실행"))
    steps.append("결과 카드를 렌더링하여 채팅에 출력")
    return steps


def _build_trace(action: dict[str, Any], session_state: dict[str, Any], dataset_ids: list[str], dataset_names: list[str]) -> ExecutionTrace:
    intent = str(action.get("intent") or "summary")
    columns = [str(c) for c in action.get("targets", {}).get("columns", []) if c]
    args = action.get("args", {}) if isinstance(action.get("args"), dict) else {}
    datasets = [{"dataset_id": did, "name": dataset_names[idx] if idx < len(dataset_names) else did} for idx, did in enumerate(dataset_ids)]
    return {
        "trace_id": str(uuid4()),
        "session_id": session_state.get("session_id"),
        "intent": intent,
        "datasets": datasets,
        "columns": columns,
        "args": args,
        "steps": _steps_for_intent(intent, dataset_names, columns),
        "code": "\n".join(_code_lines_for_intent(intent, dataset_ids, columns, args)),
        "created_at": _utc_now_iso(),
    }


def execute_actions(actions: list[dict[str, Any]], session_state: dict[str, Any]) -> tuple[list[Card], list[ExecutionTrace], list[str | None]]:
    pandas_err = _ensure_pandas()
    if pandas_err:
        return [make_text_card("실행 실패", pandas_err)], [], [None]

    registry = session_state.get("registry")
    if not registry:
        return [make_text_card("실행 실패", "데이터 레지스트리를 찾지 못했습니다.")], [], [None]

    cards: list[Card] = []
    traces: list[ExecutionTrace] = []
    card_trace_ids: list[str | None] = []

    for action in actions:
        intent = action.get("intent")
        dataset_ids = _resolve_dataset_ids(action, session_state)
        names_by_id: dict[str, str] = {}
        for did in dataset_ids:
            bundle = _dataset_bundle(did, session_state)
            if bundle:
                names_by_id[did] = bundle[0]

        trace = _build_trace(action, session_state, dataset_ids, [names_by_id.get(did, did) for did in dataset_ids])
        traces.append(trace)
        trace_id = trace["trace_id"]

        if intent == "plot":
            new_cards = _execute_plot(action, session_state)
            cards.extend(new_cards)
            card_trace_ids.extend([trace_id for _ in new_cards])
            continue
        if intent == "compare":
            new_cards = _execute_compare(action, session_state)
            cards.extend(new_cards)
            card_trace_ids.extend([trace_id for _ in new_cards])
            continue

        if not dataset_ids:
            new_cards = [make_text_card("실행 실패", "분석할 데이터셋이 없습니다. 파일을 먼저 첨부해 주세요.")]
            cards.extend(new_cards)
            card_trace_ids.extend([trace_id])
            continue

        if intent in SUPPORTED_FANOUT_INTENTS and len(dataset_ids) >= 1:
            for did in dataset_ids:
                bundle = _dataset_bundle(did, session_state)
                if not bundle:
                    new_cards = [make_text_card("실행 실패", f"데이터프레임 로드 실패: {did}")]
                    cards.extend(new_cards)
                    card_trace_ids.extend([trace_id])
                    continue
                name, df = bundle
                new_cards = _execute_for_df(action, name, df)
                cards.extend(new_cards)
                card_trace_ids.extend([trace_id for _ in new_cards])
            continue

        bundle = _dataset_bundle(dataset_ids[0], session_state)
        if not bundle:
            new_cards = [make_text_card("실행 실패", f"데이터프레임 로드 실패: {dataset_ids[0]}")]
            cards.extend(new_cards)
            card_trace_ids.extend([trace_id])
            continue
        name, df = bundle
        new_cards = _execute_for_df(action, name, df)
        cards.extend(new_cards)
        card_trace_ids.extend([trace_id for _ in new_cards])

    return cards, traces, card_trace_ids
