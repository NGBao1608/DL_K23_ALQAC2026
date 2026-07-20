import json
from pathlib import Path

import pytest

from alqac2026.artifacts import DriveArtifactLayout, export_run, restore_sqlite_cache
from alqac2026.case_retrieval import SQLiteEvidenceCache


def test_restore_sqlite_cache_verifies_and_replaces_local_copy(tmp_path):
    backup_path = tmp_path / "drive" / "case_api.sqlite"
    backup = SQLiteEvidenceCache(backup_path)
    backup.put("case_1", "query", [{"chunk_id": "opaque"}])
    backup.close()

    local_path = tmp_path / "local" / "case_api.sqlite"
    assert restore_sqlite_cache(backup_path, local_path)
    restored = SQLiteEvidenceCache(local_path)
    assert restored.contains("case_1", "query")
    restored.close()


def test_export_run_contains_only_allowlisted_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "submission.json").write_text("[]", encoding="utf-8")
    (run_dir / "validation.json").write_text(
        json.dumps({"status": "PASS", "cases": 60}), encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"run": {"status": "completed", "completed": 60}}),
        encoding="utf-8",
    )
    (run_dir / "ALQAC_private_test.json").write_text("private", encoding="utf-8")
    (run_dir / "contexts.checkpoint.json").write_text("context", encoding="utf-8")

    export_dir = export_run(run_dir, tmp_path / "export")
    exported = {path.name for path in export_dir.iterdir()}
    assert exported == {
        "submission.json",
        "validation.json",
        "manifest.json",
        "SHA256SUMS",
    }


def test_export_run_rejects_failed_or_partial_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "submission.json").write_text("[]", encoding="utf-8")
    (run_dir / "validation.json").write_text(
        json.dumps({"status": "PASS", "cases": 2}), encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"run": {"status": "failed", "completed": 1}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not complete"):
        export_run(run_dir, tmp_path / "export")


def test_drive_layout_separates_public_and_private_runs(tmp_path):
    public = DriveArtifactLayout(Path(tmp_path), "public", "run-v1")
    private = DriveArtifactLayout(Path(tmp_path), "private", "run-v1")
    assert public.run_dir != private.run_dir
    assert public.cache_backup == private.cache_backup
