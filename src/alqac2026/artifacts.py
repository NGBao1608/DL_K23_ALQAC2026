from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path


EXPORT_FILES = (
    "submission.json",
    "validation.json",
    "manifest.json",
)
MAX_SUBMISSION_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class DriveArtifactLayout:
    root: Path
    track: str
    run_id: str

    def __post_init__(self) -> None:
        if self.track not in {"public", "private"}:
            raise ValueError("track must be public or private")
        if not self.run_id.strip() or "/" in self.run_id or "\\" in self.run_id:
            raise ValueError("run_id must be one non-empty path component")

    @property
    def cache_backup(self) -> Path:
        return self.root / "cache" / "case_api.sqlite"

    @property
    def run_dir(self) -> Path:
        return self.root / "runs" / self.track / self.run_id

    @property
    def export_dir(self) -> Path:
        return self.root / "exports" / self.run_id

    @property
    def private_input(self) -> Path:
        return self.root / "inputs" / "private" / "ALQAC_private_test.json"


def restore_sqlite_cache(backup_path: str | Path, local_path: str | Path) -> bool:
    """Restore a verified Drive backup to local storage.

    Returns False when no backup exists. Existing local data is replaced only after
    the copied database passes SQLite integrity validation.
    """
    source = Path(backup_path)
    target = Path(local_path)
    if target.exists():
        _validate_sqlite(target)
        if _sqlite_backup_pending(target):
            # A failed prior Drive backup left newer committed local state. Preserve
            # it so the next live client can publish it before any cache hit/request.
            return False
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        _validate_sqlite(temporary)
        temporary.replace(target)
        return True
    finally:
        if temporary.exists():
            temporary.unlink()


def restore_directory(backup_dir: str | Path, local_dir: str | Path) -> bool:
    source = Path(backup_dir)
    target = Path(local_dir)
    if not source.is_dir():
        return False
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)
    return True


def backup_directory(local_dir: str | Path, backup_dir: str | Path) -> Path:
    source = Path(local_dir)
    target = Path(backup_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Artifact directory does not exist: {source}")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)
    return target


def export_run(run_dir: str | Path, export_dir: str | Path) -> Path:
    source = Path(run_dir)
    target = Path(export_dir)
    if target.exists():
        raise FileExistsError(f"Export directory already exists: {target}")
    validation_path = source / "validation.json"
    manifest_path = source / "manifest.json"
    if not validation_path.is_file() or not manifest_path.is_file():
        raise ValueError("Run is not exportable; validation or manifest is missing")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = manifest.get("run", {}) if isinstance(manifest, dict) else {}
    if validation.get("status") != "PASS" or run.get("status") != "completed":
        raise ValueError("Run is not exportable; validation/run status is not complete")
    if validation.get("cases") != run.get("completed"):
        raise ValueError("Run is not exportable; validated and completed counts differ")
    submission_path = source / "submission.json"
    if not submission_path.is_file():
        raise ValueError("Run is not exportable; submission is missing")
    if submission_path.stat().st_size > MAX_SUBMISSION_BYTES:
        raise ValueError("Run is not exportable; submission exceeds 10 MB")
    submission_payload = json.loads(submission_path.read_text(encoding="utf-8"))
    if not isinstance(submission_payload, list):
        raise ValueError("Run is not exportable; submission root is not a list")
    if len(submission_payload) != validation.get("cases"):
        raise ValueError("Run is not exportable; submission case count changed")
    actual_sha256 = _sha256(submission_path)
    actual_bytes = submission_path.stat().st_size
    manifest_submission = manifest.get("submission", {})
    if (
        validation.get("submission_sha256") != actual_sha256
        or validation.get("submission_bytes") != actual_bytes
        or manifest_submission.get("sha256") != actual_sha256
        or manifest_submission.get("bytes") != actual_bytes
        or manifest_submission.get("cases") != len(submission_payload)
    ):
        raise ValueError(
            "Run is not exportable; submission bytes do not match validation/manifest"
        )

    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for filename in EXPORT_FILES:
        candidate = source / filename
        if candidate.is_file():
            destination = target / filename
            shutil.copy2(candidate, destination)
            copied.append(destination)
    required = {"submission.json", "validation.json", "manifest.json"}
    if not required.issubset({path.name for path in copied}):
        missing = sorted(required - {path.name for path in copied})
        raise ValueError(f"Run is not exportable; missing artifacts: {missing}")
    write_sha256sums(copied, target / "SHA256SUMS")
    return target


def write_sha256sums(paths: list[Path], output_path: str | Path) -> Path:
    output = Path(output_path)
    lines = [f"{_sha256(path)}  {path.name}" for path in sorted(paths)]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _validate_sqlite(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute("PRAGMA integrity_check").fetchone()
    finally:
        connection.close()
    if row is None or row[0] != "ok":
        raise ValueError(f"SQLite cache integrity check failed: {row}")


def _sqlite_backup_pending(path: Path) -> bool:
    connection = sqlite3.connect(path)
    try:
        table = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'cache_state'
            """
        ).fetchone()
        if table is None:
            return False
        row = connection.execute(
            "SELECT value FROM cache_state WHERE key = 'backup_pending'"
        ).fetchone()
        return row is not None and row[0] == "1"
    finally:
        connection.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
