from __future__ import annotations

import base64
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from shared.constants import analyzer_home

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
except Exception:  # noqa: BLE001
    plt = None
    fm = None

TIME_KEYWORDS = ["date", "time", "timestamp", "일시", "날짜", "시간", "측정일시", "수집일시"]
TIME_EXCLUDE = ["위도", "경도", "lat", "lon", "lng", "gps"]
EPOCH_HINTS = ["epoch", "unix", "timestamp", "time_ms", "time_s", "ts"]
_FONT_APPLIED = False
_HAS_KO_FONT = False


def _slug(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", text.strip())
    return text[:80] or "dataset"


def _fig_to_data_uri(fig: Any, save_path: Path) -> str:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    data = buf.getvalue()
    save_path.write_bytes(data)
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _ensure_font() -> bool:
    global _FONT_APPLIED, _HAS_KO_FONT
    if _FONT_APPLIED:
        return _HAS_KO_FONT
    if plt is None or fm is None:
        return False
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False
    _HAS_KO_FONT = any(c in available for c in ["Malgun Gothic", "NanumGothic", "AppleGothic"])
    _FONT_APPLIED = True
    return _HAS_KO_FONT


def _valid_year_ratio(ts: pd.Series) -> float:
    if ts.empty:
        return 0.0
    y = ts.dt.year
    return float(((y >= 1970) & (y <= 2100)).mean())


def _time_column(df: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
    cols = [str(c) for c in df.columns]
    if df.empty:
        return None, None

    for c in cols:
        if any(k in c.lower() for k in TIME_EXCLUDE):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            ts = pd.to_datetime(df[c], errors="coerce")
            if ts.notna().sum() >= 5:
                return c, ts

    sample_n = min(200, len(df))
    for c in cols:
        lc = c.lower()
        if any(k in lc for k in TIME_EXCLUDE):
            continue
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]) or any(k in lc for k in TIME_KEYWORDS):
            sample = pd.to_datetime(df[c].head(sample_n), errors="coerce")
            if float(sample.notna().mean()) >= 0.80 and _valid_year_ratio(sample.dropna()) >= 0.95 and sample.dt.floor("D").nunique() >= 5:
                return c, pd.to_datetime(df[c], errors="coerce")

    for c in cols:
        lc = c.lower()
        if any(k in lc for k in TIME_EXCLUDE) or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if not any(h in lc for h in EPOCH_HINTS):
            continue
        num = pd.to_numeric(df[c], errors="coerce")
        med = float(num.dropna().median()) if num.notna().any() else 0.0
        unit = "s" if 1e9 <= med <= 2e9 else ("ms" if 1e12 <= med <= 2e12 else None)
        if not unit:
            continue
        parsed = pd.to_datetime(num, unit=unit, errors="coerce")
        if float(parsed.notna().mean()) >= 0.95 and _valid_year_ratio(parsed.dropna()) >= 0.95:
            return c, parsed

    return None, None


def create_plot_cards(df: pd.DataFrame, dataset_name: str, session_id: str | None, args: dict[str, Any]) -> list[dict[str, Any]]:
    if plt is None:
        return []

    has_ko_font = _ensure_font()
    top_n = int(args.get("top_n", 20))
    preferred_cols = [str(c) for c in args.get("columns", []) if c]
    mode = str(args.get("mode", "auto")).lower()

    title_missing = f"결측률 Top{top_n}" if has_ko_font else f"Missing Top{top_n}"
    title_hist = "히스토그램" if has_ko_font else "Histogram"
    title_category = f"범주 Top{top_n}" if has_ko_font else f"Category Top{top_n}"
    title_ts = "시계열(일단위 count)" if has_ko_font else "Timeseries (daily count)"

    cards: list[dict[str, Any]] = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sess = _slug(session_id or "session")
    name_slug = _slug(dataset_name)
    out_dir = analyzer_home() / "results" / "charts" / sess

    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0].head(top_n)
    if mode == "missing":
        if missing.empty:
            return [{"type": "text", "title": f"[{dataset_name}] 시각화 안내", "text": "결측 컬럼이 없어 결측률 그래프를 생략했습니다."}]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(missing.index.astype(str), missing.values)
        ax.set_title(f"[{dataset_name}] {title_missing}")
        ax.set_ylabel("missing %")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        path = out_dir / f"{name_slug}_{stamp}_missing.png"
        uri = _fig_to_data_uri(fig, path)
        plt.close(fig)
        return [{"type": "chart", "title": f"[{dataset_name}] {title_missing}", "image_data_uri": uri, "meta": {"kind": "missing_top", "top_n": top_n, "path": str(path)}}]

    if mode in {"auto", "hist"}:
        num = df.select_dtypes(include=["number"])
        if not num.empty:
            candidate_cols = [c for c in preferred_cols if c in num.columns] + [c for c in num.columns if c not in preferred_cols]
            num_col = candidate_cols[0] if candidate_cols else None
            if num_col is not None:
                series = pd.to_numeric(df[num_col], errors="coerce").dropna()
                if not series.empty:
                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    ax.hist(series.values, bins=30)
                    ax.set_title(f"[{dataset_name}] {title_hist}: {num_col}")
                    ax.set_xlabel(str(num_col))
                    path = out_dir / f"{name_slug}_{stamp}_hist.png"
                    uri = _fig_to_data_uri(fig, path)
                    plt.close(fig)
                    cards.append({"type": "chart", "title": f"[{dataset_name}] {title_hist}: {num_col}", "image_data_uri": uri, "meta": {"kind": "histogram", "column": str(num_col), "bins": 30, "path": str(path)}})
        if mode == "hist":
            return cards[:1]

    if mode in {"auto", "category"}:
        cat_candidates = [str(c) for c in df.columns if str(df[c].dtype) in {"object", "category"} and 2 <= int(df[c].nunique(dropna=True)) <= 2000]
        cat_col = next((c for c in preferred_cols if c in cat_candidates), cat_candidates[0] if cat_candidates else None)
        if cat_col is not None:
            vc = df[cat_col].astype(str).value_counts().head(top_n)
            if not vc.empty:
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.bar(vc.index.astype(str), vc.values)
                ax.set_title(f"[{dataset_name}] {title_category}: {cat_col}")
                ax.tick_params(axis="x", rotation=45)
                path = out_dir / f"{name_slug}_{stamp}_category.png"
                uri = _fig_to_data_uri(fig, path)
                plt.close(fig)
                cards.append({"type": "chart", "title": f"[{dataset_name}] {title_category}: {cat_col}", "image_data_uri": uri, "meta": {"kind": "category_top", "column": str(cat_col), "top_n": top_n, "path": str(path)}})
        if mode == "category":
            return cards[:2]

    if mode in {"auto", "timeseries"}:
        tcol, ts = _time_column(df)
        if mode == "timeseries" and (tcol is None or ts is None):
            return [{"type": "text", "title": f"[{dataset_name}] 시각화 안내", "text": "시간 컬럼을 못 찾아 시계열을 생략했습니다."}]
        if tcol is not None and ts is not None:
            ts_count = ts.dropna().dt.floor("D").value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(ts_count.index, ts_count.values)
            ax.set_title(f"[{dataset_name}] {title_ts}: {tcol}")
            ax.set_ylabel("count")
            path = out_dir / f"{name_slug}_{stamp}_timeseries.png"
            uri = _fig_to_data_uri(fig, path)
            plt.close(fig)
            cards.append({"type": "chart", "title": f"[{dataset_name}] {title_ts}", "image_data_uri": uri, "meta": {"kind": "timeseries_count_daily", "time_column": str(tcol), "path": str(path)}})

    return cards[:4]
