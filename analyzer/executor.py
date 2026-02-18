from __future__ import annotations

from typing import Any

try:
    import pandas as pd
except Exception:  # noqa: BLE001
    pd = None

from .cards import Card, make_table_card, make_text_card
from .normalize import normalize_text


LAT_NAMES = {normalize_text(v) for v in ["lat", "latitude", "위도", "gps_lat"]}
LON_NAMES = {normalize_text(v) for v in ["lon", "lng", "longitude", "경도", "gps_lon"]}


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


def _columns_for_action(action: dict[str, Any], df: pd.DataFrame) -> list[str]:
    targets = action.get("targets", {})
    target_cols = [str(c) for c in targets.get("columns", []) if c]
    if not target_cols:
        return df.columns.tolist()
    return [c for c in target_cols if c in df.columns]


def _missing_ratio_table(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
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
    table = table.sort_values(["missing_rate(%)", "missing_count"], ascending=[False, False]).head(top_n)
    return table.reset_index(drop=True)


def _schema_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        ser = df[col]
        non_null = int(ser.notna().sum())
        null_cnt = int(ser.isna().sum())
        sample = ""
        nn = ser.dropna()
        if not nn.empty:
            sample = str(nn.iloc[0])
        rows.append(
            {
                "column": str(col),
                "dtype": str(ser.dtype),
                "non_null": non_null,
                "null_count": null_cnt,
                "sample": sample,
            }
        )
    return pd.DataFrame(rows)


def _numeric_describe_table(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=["number"])
    if num.empty:
        return pd.DataFrame(columns=["column", "count", "mean", "std", "min", "max"])
    desc = num.describe().T.reset_index().rename(columns={"index": "column"})
    use_cols = [c for c in ["column", "count", "mean", "std", "min", "max"] if c in desc.columns]
    return desc[use_cols].round(4)


def _validate_geo(df: pd.DataFrame) -> list[str]:
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
        invalid = int(((lat < -90) | (lat > 90)).fillna(False).sum())
        notes.append(f"위도 범위 위반(-90~90): {invalid}건")
    if lon_col is not None:
        lon = pd.to_numeric(df[lon_col], errors="coerce")
        invalid = int(((lon < -180) | (lon > 180)).fillna(False).sum())
        notes.append(f"경도 범위 위반(-180~180): {invalid}건")
    return notes


def _execute_for_df(action: dict[str, Any], dataset_name: str, df: pd.DataFrame) -> list[Card]:
    intent = action.get("intent", "summary")
    cards: list[Card] = []
    cols = _columns_for_action(action, df)
    selected_df = df[cols] if cols and len(cols) < len(df.columns) else df

    if intent == "columns":
        all_cols = [str(c) for c in df.columns.tolist()]
        head = all_cols[:50]
        msg = "\n".join(f"- {c}" for c in head)
        if len(all_cols) > 50:
            msg += "\n... 더 보려면 컬럼명을 검색해 주세요."
        cards.append(make_text_card(f"{dataset_name} 컬럼", msg, {"total_columns": len(all_cols)}))
        return cards

    if intent == "preview":
        limit = int(action.get("args", {}).get("limit", 20))
        cards.append(make_table_card(f"{dataset_name} 상위 {limit}행", selected_df.head(limit)))
        return cards

    if intent == "schema":
        cards.append(make_table_card(f"{dataset_name} 스키마", _schema_table(selected_df)))
        return cards

    if intent == "summary":
        rows, cols_n = selected_df.shape
        dup_count = int(selected_df.duplicated().sum())
        missing_top = _missing_ratio_table(selected_df, top_n=20)
        text = (
            f"행: {rows:,}\n"
            f"열: {cols_n:,}\n"
            f"중복 행: {dup_count:,}\n"
            f"숫자형 컬럼 수: {selected_df.select_dtypes(include=['number']).shape[1]:,}"
        )
        cards.append(make_text_card(f"{dataset_name} 요약", text))
        cards.append(make_table_card(f"{dataset_name} 결측 Top20", missing_top))
        desc = _numeric_describe_table(selected_df)
        if not desc.empty:
            cards.append(make_table_card(f"{dataset_name} 숫자형 통계", desc.head(20)))
        return cards

    if intent == "validate":
        dup_count = int(selected_df.duplicated().sum())
        missing_top = _missing_ratio_table(selected_df, top_n=20)
        notes = [f"중복 행 수: {dup_count:,}"]
        notes.extend(_validate_geo(selected_df))
        if len(notes) == 1:
            notes.append("좌표(위도/경도) 컬럼이 없어 범위 검증은 생략했습니다.")
        cards.append(make_text_card(f"{dataset_name} 검증", "\n".join(notes)))
        cards.append(make_table_card(f"{dataset_name} 결측률 Top20", missing_top))
        return cards

    cards.append(make_text_card("미지원 intent", f"intent={intent}는 아직 실행 엔진에서 미지원입니다."))
    return cards


def execute_actions(actions: list[dict[str, Any]], session_state: dict[str, Any]) -> list[Card]:
    pandas_err = _ensure_pandas()
    if pandas_err:
        return [make_text_card("실행 실패", pandas_err)]

    registry = session_state.get("registry")
    datasets_meta = {d["dataset_id"]: d for d in session_state.get("datasets", [])}
    cards: list[Card] = []

    if not registry:
        return [make_text_card("실행 실패", "데이터 레지스트리를 찾지 못했습니다.")]

    for action in actions:
        dataset_ids = _resolve_dataset_ids(action, session_state)
        if not dataset_ids:
            cards.append(make_text_card("실행 실패", "분석할 데이터셋이 없습니다. 파일을 먼저 첨부해 주세요."))
            continue

        for dataset_id in dataset_ids:
            df = registry.get_frame(dataset_id)
            meta = datasets_meta.get(dataset_id, {})
            name = str(meta.get("name") or dataset_id)
            if df is None:
                cards.append(make_text_card("실행 실패", f"데이터프레임 로드 실패: {name}"))
                continue
            cards.extend(_execute_for_df(action, name, df))

    return cards
