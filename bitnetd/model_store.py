from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from shared.constants import bitnet_home, ensure_dirs

DEFAULT_MODEL_ID = "microsoft/bitnet-b1.58-2B-4T"
DEFAULT_REVISION = "main"
_MANIFEST_PATH = bitnet_home() / "config" / "manifest.json"


@dataclass(slots=True)
class ModelManifest:
    model_id: str
    revision: str
    snapshot_path: str


def manifest_path() -> Path:
    ensure_dirs()
    return _MANIFEST_PATH


def load_manifest() -> ModelManifest | None:
    path = manifest_path()
    if not path.exists():
        return None

    raw = json.loads(path.read_text(encoding="utf-8"))
    snapshot_path = str(raw.get("snapshot_path", "")).strip()
    model_id = str(raw.get("model_id", "")).strip()
    revision = str(raw.get("revision", "")).strip()
    if not snapshot_path or not model_id or not revision:
        return None
    return ModelManifest(model_id=model_id, revision=revision, snapshot_path=snapshot_path)


def _write_manifest(
    *,
    model_id: str,
    revision: str,
    snapshot_path: Path,
) -> None:
    files = sorted(p.name for p in snapshot_path.iterdir()) if snapshot_path.exists() else []
    payload: dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
        "snapshot_path": str(snapshot_path),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    manifest_path().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_model_snapshot(
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
) -> Path:
    ensure_dirs()
    manifest = load_manifest()
    if manifest and manifest.revision != revision:
        old_path = Path(manifest.snapshot_path)
        if old_path.exists() and old_path.is_dir():
            shutil.rmtree(old_path, ignore_errors=True)

    local_dir = bitnet_home() / "models"
    snapshot_path_str = snapshot_download(
        repo_id=model_id,
        revision=revision,
        resume_download=True,
        local_dir=local_dir,
    )
    snapshot_path = Path(snapshot_path_str)
    _write_manifest(model_id=model_id, revision=revision, snapshot_path=snapshot_path)
    return snapshot_path
