import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import alqac2026.colab_workflow as colab_workflow
from alqac2026.colab_workflow import (
    _load_or_create_full_budget,
    _require_smoke_gate,
    _verify_existing_export,
    _validate_private_corpus,
    _validate_stage_inputs,
    _write_or_verify_json,
    run_public_stage,
)


def test_stage_contract_exposes_only_smoke_and_full():
    _validate_stage_inputs("public", "smoke", "run-v1", 4)
    _validate_stage_inputs("private", "full", "run-v1", 0)
    with pytest.raises(ValueError, match="smoke or full"):
        _validate_stage_inputs("public", "resume", "run-v1", 4)
    with pytest.raises(ValueError, match="smoke or full"):
        _validate_stage_inputs("public", "runtime_check", "run-v1", 4)


def test_public_live_calls_require_explicit_approval():
    with pytest.raises(ValueError, match="APPROVE_PUBLIC_API_CALLS"):
        run_public_stage(
            stage="smoke",
            run_id="run-v1",
            repo_root=".",
            drive_root=".",
            local_root=".",
            config_path="configs/candidate.yaml",
            approved_live_api=False,
        )


def test_full_budget_is_created_once_and_reused(tmp_path):
    plan = {"cache_misses": 96}
    assert (
        _load_or_create_full_budget(
            workflow_root=tmp_path,
            api_plan=plan,
            retry_reserve=4,
        )
        == 100
    )
    saved = json.loads((tmp_path / "full_budget.json").read_text())
    assert saved == {
        "approved_max_network_calls": 100,
        "initial_cache_misses": 96,
        "retry_reserve": 4,
    }
    assert (
        _load_or_create_full_budget(
            workflow_root=tmp_path,
            api_plan={"cache_misses": 10},
            retry_reserve=99,
        )
        == 100
    )


def test_full_requires_smoke_gate_on_same_commit(tmp_path):
    (tmp_path / "smoke_gate.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "cases": 2,
                "git_commit": "a" * 40,
            }
        )
    )
    (tmp_path / "runtime_check.json").write_text(json.dumps({"status": "PASS"}))
    _require_smoke_gate(tmp_path, {"commit": "a" * 40})
    with pytest.raises(ValueError, match="pinned commit"):
        _require_smoke_gate(tmp_path, {"commit": "b" * 40})


def test_private_corpus_matches_reviewed_contract(tmp_path, monkeypatch):
    corpus_path = tmp_path / "private_test_60_cases_extracted_corpus.json"
    corpus_path.write_text("{}")
    articles = [
        SimpleNamespace(law_id=f"law-{index % 14}") for index in range(2820)
    ]
    monkeypatch.setattr(
        colab_workflow,
        "sha256_file",
        lambda _: colab_workflow.PRIVATE_CORPUS_SHA256,
    )
    monkeypatch.setattr(colab_workflow, "load_law_corpus", lambda _: articles)

    _validate_private_corpus(corpus_path)


def test_workflow_config_cannot_drift_for_existing_run_id(tmp_path):
    path = tmp_path / "workflow_config.json"
    _write_or_verify_json(path, {"value": 1})
    _write_or_verify_json(path, {"value": 1})
    with pytest.raises(ValueError, match="configuration changed"):
        _write_or_verify_json(path, {"value": 2})


def test_existing_export_must_match_full_run_and_checksums(tmp_path):
    stage_dir = tmp_path / "full"
    export_dir = tmp_path / "export"
    stage_dir.mkdir()
    export_dir.mkdir()
    lines = []
    for filename in ("submission.json", "validation.json", "manifest.json"):
        payload = f"{filename}-payload"
        (stage_dir / filename).write_text(payload)
        (export_dir / filename).write_text(payload)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        lines.append(f"{digest}  {filename}")
    (export_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")

    _verify_existing_export(stage_dir, export_dir)
    (export_dir / "submission.json").write_text("replaced")
    with pytest.raises(ValueError, match="does not match"):
        _verify_existing_export(stage_dir, export_dir)
