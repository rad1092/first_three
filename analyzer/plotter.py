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
    import matplotlib.pyplot as plt
except Exception:  # noqa: BLE001
    plt = None

TIME_KEYWORDS = ["date", "time", "timestamp", "일시", "날짜", "시간", "측정일시", "수집일시"]


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


def _time_column(df: pd.DataFrame) -> str | None:
    cols = [str(c) for c in df.columns]
    keyword_hits = [c for c in cols if any(k in c.lower() for k in TIME_KEYWORDS)]
    if keyword_hits:
        return keyword_hits[0]

    best_col = None
    best_ratio = 0.0
    sample_n = min(50, len(df))
    if sample_n == 0:
        return None
    for c in cols:
        ser = pd.to_datetime(df[c].head(sample_n), errors="coerce")
        ratio = float(ser.notna().mean())
        if ratio > best_ratio:
            best_ratio = ratio
            best_col = c
    return best_col if best_ratio >= 0.6 else None


def create_plot_cards(df: pd.DataFrame, dataset_name: str, session_id: str | None, args: dict[str, Any]) -> list[dict[str, Any]]:
    if plt is None:
        return []

    top_n = int(args.get("top_n", 20))
    preferred_cols = [str(c) for c in args.get("columns", []) if c]

    cards: list[dict[str, Any]] = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sess = _slug(session_id or "session")
    name_slug = _slug(dataset_name)
    out_dir = analyzer_home() / "results" / "charts" / sess

    # 1) missing rate bar
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0].head(top_n)
    if not missing.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(missing.index.astype(str), missing.values)
        ax.set_title(f"[{dataset_name}] 결측률 Top{top_n}")
        ax.set_ylabel("missing %")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        path = out_dir / f"{name_slug}_{stamp}_missing.png"
        uri = _fig_to_data_uri(fig, path)
        plt.close(fig)
        cards.append(
            {
                "type": "chart",
                "title": f"[{dataset_name}] 결측률 Top{top_n}",
                "image_data_uri": uri,
                "meta": {"kind": "missing_top", "top_n": top_n, "path": str(path)},
            }
        )

    # 2) numeric histogram
    num = df.select_dtypes(include=["number"])
    num_col = None
    if not num.empty:
        candidate_cols = [c for c in preferred_cols if c in num.columns] + [c for c in num.columns if c not in preferred_cols]
        for c in candidate_cols:
            valid = pd.to_numeric(num[c], errors="coerce").dropna()
            if len(valid) >= 200:
                num_col = c
                break
        if num_col is None and candidate_cols:
            num_col = candidate_cols[0]

    if num_col is not None:
        series = pd.to_numeric(df[num_col], errors="coerce").dropna()
        if not series.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.hist(series.values, bins=30)
            ax.set_title(f"[{dataset_name}] 히스토그램: {num_col}")
            ax.set_xlabel(str(num_col))
            path = out_dir / f"{name_slug}_{stamp}_hist.png"
            uri = _fig_to_data_uri(fig, path)
            plt.close(fig)
            cards.append(
                {
                    "type": "chart",
                    "title": f"[{dataset_name}] 히스토그램: {num_col}",
                    "image_data_uri": uri,
                    "meta": {"kind": "histogram", "column": str(num_col), "bins": 30, "path": str(path)},
                }
            )

    # 3) categorical top20
    cat_candidates: list[str] = []
    for c in df.columns:
        if str(df[c].dtype) in {"object", "category"}:
            nunique = int(df[c].nunique(dropna=True))
            if 2 <= nunique <= 2000:
                cat_candidates.append(str(c))

    cat_col = None
    for c in preferred_cols:
        if c in cat_candidates:
            cat_col = c
            break
    if cat_col is None and cat_candidates:
        cat_col = cat_candidates[0]

    if cat_col is not None:
        vc = df[cat_col].astype(str).value_counts().head(top_n)
        if not vc.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(vc.index.astype(str), vc.values)
            ax.set_title(f"[{dataset_name}] 범주 Top{top_n}: {cat_col}")
            ax.tick_params(axis="x", rotation=45)
            path = out_dir / f"{name_slug}_{stamp}_category.png"
            uri = _fig_to_data_uri(fig, path)
            plt.close(fig)
            cards.append(
                {
                    "type": "chart",
                    "title": f"[{dataset_name}] 범주 Top{top_n}: {cat_col}",
                    "image_data_uri": uri,
                    "meta": {"kind": "category_top", "column": str(cat_col), "top_n": top_n, "path": str(path)},
                }
            )

    # 4) timeseries
    tcol = _time_column(df)
    if tcol is not None:
        ts = pd.to_datetime(df[tcol], errors="coerce").dropna()
        if not ts.empty:
            ts_count = ts.dt.floor("D").value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(ts_count.index, ts_count.values)
            ax.set_title(f"[{dataset_name}] 시계열(일단위 count): {tcol}")
            ax.set_ylabel("count")
            path = out_dir / f"{name_slug}_{stamp}_timeseries.png"
            uri = _fig_to_data_uri(fig, path)
            plt.close(fig)
            cards.append(
                {
                    "type": "chart",
                    "title": f"[{dataset_name}] 시계열(일단위 count)",
                    "image_data_uri": uri,
                    "meta": {"kind": "timeseries_count_daily", "time_column": str(tcol), "path": str(path)},
                }
            )

    # enforce 2~4 as possible
    if len(cards) > 4:
        cards = cards[:4]
    return cards
