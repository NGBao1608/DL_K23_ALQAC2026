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
        lambda results: (_ for _ in ()).throw(ValueError("forced failure")),
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
