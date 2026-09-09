"""Append-only case logging for the interactive segmentation web UI."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
from PIL import Image

try:
    from .sam_backend import Point
except ImportError:
    from sam_backend import Point


CASE_ID_PATTERN = re.compile(r"^\d{8}T\d{6}\.\d{6}Z_[0-9a-f]{8}$")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_timestamp(moment: Optional[datetime] = None) -> str:
    return (moment or utc_now()).isoformat(timespec="microseconds").replace("+00:00", "Z")


def filename_timestamp(moment: Optional[datetime] = None) -> str:
    return (moment or utc_now()).strftime("%Y%m%dT%H%M%S.%fZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prompt(point: Point) -> Dict[str, Any]:
    x_value, y_value, label = point
    return {
        "x": round(float(x_value), 2),
        "y": round(float(y_value), 2),
        "label": int(label),
        "label_name": "foreground" if label else "background",
    }


class CaseLogger:
    """Persist one immutable input and the mask produced by every point prompt."""

    def __init__(self, output_root: Path | str, model: Dict[str, Any]) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.output_root, 0o700)
        self.model = model
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _case_dir(self, case_id: str) -> Path:
        if not CASE_ID_PATTERN.fullmatch(case_id):
            raise ValueError(f"非法 case_id: {case_id}")
        return self.output_root / case_id

    def _lock(self, case_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(case_id, threading.Lock())

    @staticmethod
    def _save_png(array: np.ndarray, path: Path, mode: str) -> None:
        Image.fromarray(array, mode=mode).save(path, format="PNG")
        os.chmod(path, 0o600)

    @staticmethod
    def _write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
        temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with temporary_path.open("w", encoding="utf-8") as file_handle:
            json.dump(manifest, file_handle, ensure_ascii=False, indent=2)
            file_handle.write("\n")
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)

    @staticmethod
    def _read_manifest(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as file_handle:
            return json.load(file_handle)

    def start_case(self, image: np.ndarray, session_id: str) -> str:
        created_at = utc_now()
        case_id = f"{filename_timestamp(created_at)}_{uuid.uuid4().hex[:8]}"
        case_dir = self._case_dir(case_id)
        case_dir.mkdir(mode=0o700)
        (case_dir / "masks").mkdir(mode=0o700)
        input_path = case_dir / "input.png"
        self._save_png(image, input_path, "RGB")
        manifest = {
            "schema_version": 1,
            "case_id": case_id,
            "created_at": iso_timestamp(created_at),
            "updated_at": iso_timestamp(created_at),
            "status": "active",
            "session_hash": hashlib.sha256(session_id.encode("utf-8")).hexdigest(),
            "input": {
                "path": "input.png",
                "sha256": _sha256(input_path),
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
            },
            "model": self.model,
            "interactions": [],
        }
        self._write_manifest(case_dir / "case.json", manifest)
        return case_id

    def record_interaction(
        self,
        case_id: str,
        action: str,
        active_points: Sequence[Point],
        mask: Optional[np.ndarray] = None,
        score: Optional[float] = None,
        point: Optional[Point] = None,
        recorded_at: Optional[datetime] = None,
    ) -> None:
        case_dir = self._case_dir(case_id)
        manifest_path = case_dir / "case.json"
        with self._lock(case_id):
            manifest = self._read_manifest(manifest_path)
            event_time = recorded_at or utc_now()
            sequence = len(manifest["interactions"]) + 1
            interaction: Dict[str, Any] = {
                "sequence": sequence,
                "action": action,
                "recorded_at": iso_timestamp(event_time),
                "active_prompts": [_prompt(item) for item in active_points],
            }
            if point is not None:
                interaction["prompt"] = _prompt(point)
            if score is not None:
                interaction["score"] = float(score)
            if mask is not None:
                mask_name = f"{sequence:04d}_{filename_timestamp(event_time)}.png"
                mask_path = case_dir / "masks" / mask_name
                self._save_png(mask.astype(np.uint8) * 255, mask_path, "L")
                interaction["mask_path"] = f"masks/{mask_name}"
                interaction["mask_sha256"] = _sha256(mask_path)
            completed_at = utc_now()
            interaction["completed_at"] = iso_timestamp(completed_at)
            manifest["interactions"].append(interaction)
            manifest["updated_at"] = iso_timestamp(completed_at)
            self._write_manifest(manifest_path, manifest)

    def save_final(self, case_id: str, mask: np.ndarray, overlay: np.ndarray) -> Path:
        case_dir = self._case_dir(case_id)
        manifest_path = case_dir / "case.json"
        with self._lock(case_id):
            self._save_png(mask.astype(np.uint8) * 255, case_dir / "final_mask.png", "L")
            self._save_png(overlay, case_dir / "final_overlay.png", "RGB")
            manifest = self._read_manifest(manifest_path)
            saved_at = utc_now()
            manifest["status"] = "saved"
            manifest["saved_at"] = iso_timestamp(saved_at)
            manifest["updated_at"] = iso_timestamp(saved_at)
            manifest["final"] = {
                "mask_path": "final_mask.png",
                "overlay_path": "final_overlay.png",
            }
            self._write_manifest(manifest_path, manifest)
        return case_dir

    def export_all_cases(self) -> Path:
        case_dirs = sorted(
            path
            for path in self.output_root.iterdir()
            if path.is_dir()
            and CASE_ID_PATTERN.fullmatch(path.name)
            and (path / "case.json").is_file()
        )
        if not case_dirs:
            raise ValueError("当前没有可导出的 case 日志")
        export_dir = self.output_root / "exports"
        export_dir.mkdir(exist_ok=True, mode=0o700)
        os.chmod(export_dir, 0o700)
        archive_path = export_dir / f"all_cases_{filename_timestamp()}_{uuid.uuid4().hex[:8]}.zip"
        temporary_path = archive_path.with_name(f".{archive_path.name}.{uuid.uuid4().hex}.tmp")
        with zipfile.ZipFile(temporary_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for case_dir in case_dirs:
                with self._lock(case_dir.name):
                    for path in sorted(case_dir.rglob("*")):
                        if path.is_file():
                            archive.write(path, Path(case_dir.name) / path.relative_to(case_dir))
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, archive_path)
        return archive_path

    def close_case(self, case_id: Optional[str], reason: str) -> None:
        if not case_id:
            return
        manifest_path = self._case_dir(case_id) / "case.json"
        with self._lock(case_id):
            manifest = self._read_manifest(manifest_path)
            if manifest["status"] == "active":
                closed_at = utc_now()
                manifest["status"] = reason
                manifest["closed_at"] = iso_timestamp(closed_at)
                manifest["updated_at"] = iso_timestamp(closed_at)
                self._write_manifest(manifest_path, manifest)
