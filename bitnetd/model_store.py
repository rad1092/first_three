from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from shared.constants import bitnet_home, ensure_dirs

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "microsoft/bitnet-b1.58-2B-4T"
DEFAULT_REVISION = "main"
_MANIFEST_PATH = bitnet_home() / "config" / "manifest.json"


class ModelDownloadError(RuntimeError):
    pass


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


def _write_manifest(*, model_id: str, revision: str, snapshot_path: Path) -> None:
    files = sorted(p.name for p in snapshot_path.iterdir()) if snapshot_path.exists() else []
    payload: dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
        "snapshot_path": str(snapshot_path),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    manifest_path().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_remove_snapshot(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        return
    try:
        shutil.rmtree(path)
        logger.info("Removed previous model snapshot path=%s", path)
    except Exception as exc:
        logger.warning("Failed to remove previous snapshot path=%s error=%s", path, exc)


def resolve_model_snapshot(model_id: str = DEFAULT_MODEL_ID, revision: str = DEFAULT_REVISION) -> Path:
    ensure_dirs()
    manifest = load_manifest()
    if manifest and (manifest.model_id != model_id or manifest.revision != revision):
        logger.info(
            "Model identity changed (old=%s@%s, new=%s@%s); re-downloading",
            manifest.model_id,
            manifest.revision,
            model_id,
            revision,
        )
        _safe_remove_snapshot(Path(manifest.snapshot_path))

    local_dir = bitnet_home() / "models"
    logger.info("Starting model download model_id=%s revision=%s local_dir=%s", model_id, revision, local_dir)
    try:
        snapshot_path_str = snapshot_download(
            repo_id=model_id,
            revision=revision,
            resume_download=True,
            local_dir=local_dir,
        )
    except Exception as exc:
        logger.error("Model download failed model_id=%s revision=%s error=%s", model_id, revision, exc)
        raise ModelDownloadError(f"model_download_failed: {exc}") from exc

    snapshot_path = Path(snapshot_path_str)
    logger.info("Model download completed snapshot_path=%s", snapshot_path)
    _write_manifest(model_id=model_id, revision=revision, snapshot_path=snapshot_path)
    return snapshot_path
