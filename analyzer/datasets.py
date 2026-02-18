from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from .loader import load_csv, load_xlsx


@dataclass
class DatasetMeta:
    dataset_id: str
    path: str
    name: str
    kind: str
    loaded_ok: bool
    row_count: int
    col_count: int
    sheet_name: str | None = None
    encoding: str | None = None
    notes: str | None = None
    columns: list[str] | None = None


class DatasetRegistry:
    def __init__(self) -> None:
        self._datasets: dict[str, DatasetMeta] = {}
        self._frames: dict[str, pd.DataFrame] = {}
        self._active_dataset_id: str | None = None

    def clear(self) -> None:
        self._datasets.clear()
        self._frames.clear()
        self._active_dataset_id = None

    def _build_meta_from_loaded(self, p: Path, kind: str, df: pd.DataFrame, meta: dict[str, Any]) -> DatasetMeta:
        return DatasetMeta(
            dataset_id=str(uuid4()),
            path=str(p),
            name=p.name,
            kind=kind,
            loaded_ok=True,
            row_count=int(df.shape[0]),
            col_count=int(df.shape[1]),
            sheet_name=meta.get("sheet_name"),
            encoding=meta.get("encoding"),
            notes=meta.get("notes"),
            columns=[str(c) for c in df.columns.tolist()],
        )

    def register_files(self, file_paths: list[str]) -> list[DatasetMeta]:
        results: list[DatasetMeta] = []

        for raw_path in file_paths:
            p = Path(raw_path)
            kind = "xlsx" if p.suffix.lower() in {".xlsx", ".xlsm", ".xltx", ".xltm"} else "csv"

            try:
                if kind == "xlsx":
                    df, meta = load_xlsx(str(p))
                else:
                    df, meta = load_csv(str(p))

                dataset = self._build_meta_from_loaded(p, kind, df, meta)
                self._frames[dataset.dataset_id] = df
            except Exception as exc:  # noqa: BLE001
                dataset = DatasetMeta(
                    dataset_id=str(uuid4()),
                    path=str(p),
                    name=p.name,
                    kind=kind,
                    loaded_ok=False,
                    row_count=0,
                    col_count=0,
                    columns=[],
                    notes=(
                        "자동 로드에 실패했습니다. 인코딩/헤더/시트 선택을 자동으로 복구 시도했지만 완료하지 못했습니다. "
                        "파일을 다시 저장한 뒤 재시도해 주세요. "
                        f"원인: {exc}"
                    ),
                )

            self._datasets[dataset.dataset_id] = dataset
            self._active_dataset_id = dataset.dataset_id
            results.append(dataset)

        return results

    def add_dataset_meta(self, dataset: DatasetMeta, make_active: bool = False) -> None:
        self._datasets[dataset.dataset_id] = dataset
        if make_active:
            self._active_dataset_id = dataset.dataset_id

    def set_active(self, dataset_id: str) -> DatasetMeta | None:
        if dataset_id not in self._datasets:
            return None
        self._active_dataset_id = dataset_id
        return self._datasets[dataset_id]

    def get_active(self) -> DatasetMeta | None:
        if not self._active_dataset_id:
            return None
        return self._datasets.get(self._active_dataset_id)

    def list_all(self) -> list[DatasetMeta]:
        return list(self._datasets.values())

    def get_dataset(self, dataset_id: str) -> DatasetMeta | None:
        return self._datasets.get(dataset_id)

    def get_frame(self, dataset_id: str) -> pd.DataFrame | None:
        return self._frames.get(dataset_id)

    def as_state(self) -> dict[str, Any]:
        active = self.get_active()
        return {
            "datasets": [asdict(d) for d in self.list_all()],
            "active_dataset": asdict(active) if active else None,
        }
