import json

import pytest

import alqac2026.runner as runner
from alqac2026.schemas import InferenceCase, LawArticle, LawEvidence


class FakeLawRetriever:
    def search(self, query, top_k=None):
        return [LawEvidence("law", 1, "article", 1.0, 1)]


def _patch_small_run(monkeypatch):
    monkeypatch.setattr(
        runner, "load_inference_cases", lambda path: [InferenceCase("case_1", "query")]
    )
    monkeypatch.setattr(
        runner,
        "load_law_corpus",
        lambda path: [LawArticle("law", 1, 1, "article")],
    )
    monkeypatch.setattr(
        runner, "create_law_retriever", lambda articles, config: FakeLawRetriever()
    )


def test_resume_rejects_changed_execution_identity(tmp_path, monkeypatch):
    _patch_small_run(monkeypatch)
    run_dir = tmp_path / "run"
    runner.run_experiment(
        "configs/baseline.yaml", "input.json", resume_run=run_dir, mock=True, limit=1
    )
    with pytest.raises(ValueError, match="different config, input, mock mode, or limit"):
        runner.run_experiment(
            "configs/baseline.yaml",
            "input.json",
            resume_run=run_dir,
            mock=True,
            limit=None,
        )


def test_failure_always_writes_manifest(tmp_path, monkeypatch):
    _patch_small_run(monkeypatch)
    monkeypatch.setattr(
        runner,
        "build_submission",
        lambda results, **kwargs: (_ for _ in ()).throw(ValueError("forced failure")),
    )
    run_dir = tmp_path / "failed"
    with pytest.raises(ValueError, match="forced failure"):
        runner.run_experiment(
            "configs/baseline.yaml", "input.json", resume_run=run_dir, mock=True
        )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run"]["status"] == "failed"
    assert "forced failure" in manifest["run"]["error"]
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "api_stats.json").exists()


def test_non_mock_run_requires_explicit_api_budget():
    with pytest.raises(ValueError, match="explicit max_network_calls"):
        runner.run_experiment("configs/baseline.yaml", "input.json", mock=False)


def test_live_run_requires_distinct_external_cache_backup(tmp_path):
    cache_path = tmp_path / "cache.sqlite"
    with pytest.raises(ValueError, match="external cache_backup_db"):
        runner.run_experiment(
            "configs/baseline.yaml",
            "input.json",
            execution_mode="live",
            max_network_calls=0,
        )
    with pytest.raises(ValueError, match="paths must differ"):
        runner.run_experiment(
            "configs/baseline.yaml",
            "input.json",
            execution_mode="live",
            max_network_calls=0,
            cache_db=cache_path,
            cache_backup_db=cache_path,
        )


def test_cache_override_is_recorded_in_resolved_config(tmp_path, monkeypatch):
    _patch_small_run(monkeypatch)
    run_dir = tmp_path / "run"
    cache_path = tmp_path / "external" / "case_api.sqlite"
    runner.run_experiment(
        "configs/baseline.yaml",
        "input.json",
        resume_run=run_dir,
        mock=True,
        limit=1,
        cache_db=cache_path,
    )
    resolved = json.loads(
        (run_dir / "config.resolved.json").read_text(encoding="utf-8")
    )
    assert resolved["config"]["paths"]["cache_db"] == str(cache_path)


def test_mock_run_emits_safe_structured_per_case_progress(
    tmp_path, monkeypatch, capsys
):
    _patch_small_run(monkeypatch)
    runner.run_experiment(
        "configs/baseline.yaml",
        "input.json",
        resume_run=tmp_path / "run",
        mock=True,
        limit=1,
    )

    events = [
        json.loads(line.removeprefix(runner.PROGRESS_PREFIX))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(runner.PROGRESS_PREFIX)
    ]
    case_events = [event for event in events if event.get("case_id") == "case_1"]

    assert [(event["stage"], event["status"]) for event in case_events] == [
        ("preparation", "started"),
        ("preparation", "completed"),
        ("prediction", "started"),
        ("prediction", "completed"),
    ]
    completed = case_events[-1]
    assert completed["index"] == 1
    assert completed["total"] == 1
    assert completed["prediction"] == "PARTIAL_A_WIN"
    assert completed["case_evidence_count"] == 0
    assert completed["law_evidence_count"] == 1
    assert "reasoning" not in completed
    assert "raw_output" not in completed
    assert "case_query" not in completed


def test_cache_only_run_uses_real_pipeline_without_network_or_token(
    tmp_path, monkeypatch
):
    _patch_small_run(monkeypatch)
    monkeypatch.setattr(
        runner,
        "create_predictor",
        lambda config: runner.OutcomePredictor(runner.FixedBackend()),
    )
    monkeypatch.setenv("ALQAC_TEAM_TOKEN", "must-not-be-read")
    run_dir = tmp_path / "cache-only"
    result = runner.run_experiment(
        "configs/candidate.yaml",
        "input.json",
        resume_run=run_dir,
        cache_db=tmp_path / "cache.sqlite",
        execution_mode="cache-only",
        max_network_calls=0,
    )
    api_plan = json.loads((run_dir / "api_plan.json").read_text(encoding="utf-8"))
    api_stats = json.loads((run_dir / "api_stats.json").read_text(encoding="utf-8"))
    assert result["validation"]["status"] == "PASS"
    assert api_plan["execution_mode"] == "cache-only"
    assert api_plan["cache_misses"] == 2
    assert api_stats["run_network_attempts"] == 0


def test_private_selection_profile_is_reduced_to_scalar_and_fingerprint(
    tmp_path, monkeypatch
):
    _patch_small_run(monkeypatch)
    monkeypatch.setattr(
        runner,
        "create_predictor",
        lambda config: runner.OutcomePredictor(runner.FixedBackend()),
    )
    source_profile = tmp_path / "public-selection.json"
    source_profile.write_text(
        json.dumps(
            {
                "submission_law_top_k": 3,
                "scores": {"3": 0.8, "4": 0.7},
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "private"
    runner.run_experiment(
        "configs/candidate.yaml",
        "input.json",
        resume_run=run_dir,
        cache_db=tmp_path / "cache.sqlite",
        execution_mode="cache-only",
        max_network_calls=0,
        selection_profile=source_profile,
    )
    copied_profile = json.loads(
        (run_dir / "selection_profile.json").read_text(encoding="utf-8")
    )
    assert copied_profile["submission_law_top_k"] == 3
    assert "scores" not in copied_profile
    assert len(copied_profile["source_profile_sha256"]) == 64
