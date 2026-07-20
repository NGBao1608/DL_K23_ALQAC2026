import hashlib
import json
from pathlib import Path

import pytest

from alqac2026.artifacts import DriveArtifactLayout, export_run, restore_sqlite_cache
from alqac2026.case_retrieval import SQLiteEvidenceCache


def _write_bound_run(run_dir, submission, *, status="completed", completed=None):
    run_dir.mkdir()
    submission_path = run_dir / "submission.json"
    submission_path.write_text(json.dumps(submission), encoding="utf-8")
    digest = hashlib.sha256(submission_path.read_bytes()).hexdigest()
    size = submission_path.stat().st_size
    case_count = len(submission)
    (run_dir / "validation.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "cases": case_count,
                "submission_sha256": digest,
                "submission_bytes": size,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run": {
                    "status": status,
                    "completed": case_count if completed is None else completed,
                },
                "submission": {
                    "sha256": digest,
                    "bytes": size,
                    "cases": case_count,
                },
            }
        ),
        encoding="utf-8",
    )


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


def test_restore_preserves_newer_local_cache_with_pending_backup(tmp_path):
    backup_path = tmp_path / "drive" / "case_api.sqlite"
    stale = SQLiteEvidenceCache(backup_path)
    stale.close()

    local_path = tmp_path / "local" / "case_api.sqlite"
    local = SQLiteEvidenceCache(local_path)
    local.record_success(
        "case_1",
        "query",
        "original",
        [{"chunk_id": "opaque"}],
    )
    assert local.has_pending_backup()
    local.close()

    assert not restore_sqlite_cache(backup_path, local_path)
    preserved = SQLiteEvidenceCache(local_path)
    assert preserved.contains("case_1", "query")
    assert preserved.has_pending_backup()
    preserved.close()


def test_export_run_contains_only_allowlisted_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    _write_bound_run(
        run_dir,
        [
            {
                "case_id": "case_1",
                "prediction": "A_WIN",
                "case_evidence": [],
                "law_evidence": [],
            }
        ],
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
    _write_bound_run(run_dir, [{"case_id": "case_1"}], status="failed")
    with pytest.raises(ValueError, match="not complete"):
        export_run(run_dir, tmp_path / "export")


def test_export_run_rejects_submission_modified_after_validation(tmp_path):
    run_dir = tmp_path / "run"
    _write_bound_run(run_dir, [{"case_id": "case_1"}])
    (run_dir / "submission.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="case count changed|do not match"):
        export_run(run_dir, tmp_path / "export")


def test_export_run_rejects_stale_claimed_case_count(tmp_path):
    run_dir = tmp_path / "run"
    _write_bound_run(run_dir, [])
    validation_path = run_dir / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["cases"] = 60
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run"]["completed"] = 60
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="case count changed"):
        export_run(run_dir, tmp_path / "export")


def test_drive_layout_separates_public_and_private_runs(tmp_path):
    public = DriveArtifactLayout(Path(tmp_path), "public", "run-v1")
    private = DriveArtifactLayout(Path(tmp_path), "private", "run-v1")
    assert public.run_dir != private.run_dir
    assert public.cache_backup == private.cache_backup
