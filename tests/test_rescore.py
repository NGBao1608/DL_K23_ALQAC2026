import json

import pytest

import alqac2026.rescore as rescore
from alqac2026.pipeline import PreparedCase, PreparedCaseStore
from alqac2026.schemas import InferenceCase


def _write_input(path, cases):
    path.write_text(
        json.dumps(
            [
                {"case_id": case.case_id, "case_query": case.case_query}
                for case in cases
            ]
        ),
        encoding="utf-8",
    )


def test_rescore_requires_complete_exact_prepared_contexts(tmp_path):
    cases = [
        InferenceCase("case_1", "query one"),
        InferenceCase("case_2", "query two"),
    ]
    input_path = tmp_path / "input.json"
    _write_input(input_path, cases)
    contexts_path = tmp_path / "contexts.json"
    PreparedCaseStore(contexts_path).put(PreparedCase(cases[0], [], [], 0))

    with pytest.raises(ValueError, match="case_2"):
        rescore.rescore_prepared_cases(
            config_path="configs/candidate_rescore_v1.yaml",
            input_path=input_path,
            prepared_contexts_path=contexts_path,
            run_dir=tmp_path / "run",
        )


def test_rescore_uses_cache_only_zero_network_contract(tmp_path, monkeypatch):
    case = InferenceCase("case_1", "query")
    input_path = tmp_path / "input.json"
    _write_input(input_path, [case])
    contexts_path = tmp_path / "contexts.json"
    PreparedCaseStore(contexts_path).put(PreparedCase(case, [], [], 0))
    captured = {}

    def fake_run_experiment(**kwargs):
        captured.update(kwargs)
        assert (kwargs["resume_run"] / "contexts.checkpoint.json").is_file()
        for filename, payload in (
            ("api_stats.json", {"run_network_attempts": 0}),
            ("manifest.json", {"run": {"status": "completed"}}),
            ("validation.json", {"status": "PASS"}),
        ):
            (kwargs["resume_run"] / filename).write_text(
                json.dumps(payload), encoding="utf-8"
            )
        return {"validation": {"status": "PASS"}}

    monkeypatch.setattr(rescore, "run_experiment", fake_run_experiment)
    result = rescore.rescore_prepared_cases(
        config_path="configs/candidate_rescore_v1.yaml",
        input_path=input_path,
        prepared_contexts_path=contexts_path,
        run_dir=tmp_path / "run",
        corpus_path=tmp_path / "private-corpus.json",
    )

    assert captured["execution_mode"] == "cache-only"
    assert captured["max_network_calls"] == 0
    assert captured["cache_db"].name == "cache-only.sqlite"
    assert captured["corpus_path"] == tmp_path / "private-corpus.json"
    assert result["network_contract"]["api_token_required"] is False
    manifest = json.loads(
        (tmp_path / "run/manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["prepared_rescore"]["prepared_contexts_sha256"]) == 64
