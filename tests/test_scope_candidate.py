import json

import pytest

import alqac2026.scope_candidate as scope_candidate
from alqac2026 import runner
from alqac2026.pipeline import PreparedCase, PreparedCaseStore
from alqac2026.schemas import CaseEvidence, InferenceCase, OutcomeLabel
from alqac2026.scope_candidate import (
    ScopeCandidatePredictor,
    order_case_evidence,
    parse_scope_prediction,
)


class SequenceBackend:
    tokenizer = None

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []
        self.max_input_tokens = 4096
        self.max_new_tokens = 320

    def count_input_tokens(self, system_prompt, user_prompt):
        return len((system_prompt + user_prompt).split())

    def generate(self, system_prompt, user_prompt):
        self.prompts.append(user_prompt)
        return next(self.outputs)

    def activate_oom_recovery(self):
        self.max_input_tokens = 3072
        self.max_new_tokens = 256
        return True


def _predictor(backend):
    return ScopeCandidatePredictor(
        backend,
        max_input_tokens=4096,
        context_law_top_k=1,
        oom_max_input_tokens=3072,
        oom_context_law_top_k=0,
        verification_reserve_tokens=512,
        case_token_share=0.85,
        law_token_cap=512,
        case_evidence_chars=2600,
        law_evidence_chars=1200,
        boundary_low=0.45,
        boundary_high=0.55,
    )


def test_scope_parser_derives_label_and_detects_inconsistent_ratio():
    parsed, strict = parse_scope_prediction(
        '{"source_role":"COURT_DECISION","claim":"trả nợ",'
        '"accepted":"một phần","scope":"AT_MOST_HALF","ratio":0.7}'
    )

    assert strict
    assert parsed.label is OutcomeLabel.PARTIAL_B_WIN
    assert parsed.inconsistent


def test_scope_parser_rejects_missing_claim():
    with pytest.raises(ValueError, match="claim"):
        parse_scope_prediction(
            '{"source_role":"COURT_DECISION","claim":"",'
            '"accepted":"","scope":"NONE","ratio":0}'
        )


def test_scope_parser_rejects_extra_verbose_fields():
    with pytest.raises(ValueError, match="extra fields"):
        parse_scope_prediction(
            '{"source_role":"COURT_DECISION","claim":"trả nợ",'
            '"accepted":"toàn bộ","scope":"ALL","ratio":1,'
            '"reasoning":"không thuộc schema"}'
        )


def test_role_aware_ordering_prioritizes_court_decision_over_party_statement():
    case = InferenceCase("case_1", "Yêu cầu trả khoản nợ 100 triệu đồng")
    party = CaseEvidence(
        "party",
        "Nguyên đơn trình bày và đề nghị Tòa án chấp nhận 100 triệu đồng.",
        0.99,
        "operative_verdict",
    )
    court = CaseEvidence(
        "court",
        "Hội đồng xét xử quyết định chấp nhận một phần yêu cầu trả nợ.",
        0.5,
        "remedy_scope",
    )

    ordered = order_case_evidence(case, [party, court])

    assert [item.chunk_id for item in ordered] == ["court", "party"]


def test_unclear_scope_runs_targeted_verifier():
    backend = SequenceBackend(
        [
            (
                '{"source_role":"UNKNOWN","claim":"trả nợ","accepted":"",'
                '"scope":"UNCLEAR","ratio":null}'
            ),
            (
                '{"source_role":"COURT_DECISION","claim":"trả nợ",'
                '"accepted":"toàn bộ","scope":"ALL","ratio":1}'
            ),
        ]
    )
    predictor = _predictor(backend)

    label, _, raw = predictor.predict(
        InferenceCase("case_1", "Yêu cầu trả nợ"),
        [
            CaseEvidence(
                "opaque",
                "Hội đồng xét xử quyết định chấp nhận toàn bộ yêu cầu.",
                1.0,
                "operative_verdict",
            )
        ],
        [],
    )

    assert label is OutcomeLabel.A_WIN
    assert len(backend.prompts) == 2
    assert "---SCOPE_VERIFICATION---" in raw
    assert predictor.last_diagnostics.output_verification == "passed"
    assert predictor.case_diagnostics[-1]["verification_triggers"]


def test_scope_repair_reuses_evidence_and_does_not_repeat_invalid_output():
    backend = SequenceBackend(
        [
            "invalid raw output",
            (
                '{"source_role":"COURT_DECISION","claim":"trả nợ",'
                '"accepted":"toàn bộ","scope":"ALL","ratio":1}'
            ),
        ]
    )
    predictor = _predictor(backend)

    label, _, raw = predictor.predict(
        InferenceCase("case_1", "Yêu cầu trả nợ"),
        [
            CaseEvidence(
                "opaque",
                "Hội đồng xét xử quyết định chấp nhận toàn bộ yêu cầu.",
                1.0,
                "operative_verdict",
            )
        ],
        [],
    )

    assert label is OutcomeLabel.A_WIN
    assert "---SCOPE_REPAIR---" in raw
    assert "Hội đồng xét xử quyết định" in backend.prompts[1]
    assert "invalid raw output" not in backend.prompts[1]
    assert predictor.last_diagnostics.output_repair_used


def test_oom_recovery_can_remove_law_context():
    predictor = _predictor(SequenceBackend([]))

    assert predictor.activate_oom_recovery()
    assert predictor.max_input_tokens == 3072
    assert predictor.context_law_top_k == 0


def test_candidate_runner_restores_default_predictor_factory(
    tmp_path,
    monkeypatch,
):
    original_factory = runner.create_predictor
    dummy_predictor = type("DummyPredictor", (), {"case_diagnostics": []})()
    captured = {}

    monkeypatch.setattr(
        scope_candidate,
        "create_scope_candidate_predictor",
        lambda config: dummy_predictor,
    )

    def fake_rescore_prepared_cases(**kwargs):
        captured.update(kwargs)
        created = runner.create_predictor(
            {
                "model_name": "Qwen/Qwen3-8B",
                "revision": (
                    "b968826d9c46dd6066d109eabc6255188de91218"
                ),
                "prompt_variant": "scope-enum-v1",
            }
        )
        assert created is dummy_predictor
        run_dir = kwargs["run_dir"]
        run_dir.mkdir(parents=True)
        for filename, payload in (
            ("manifest.json", {"run": {"status": "completed"}}),
            ("validation.json", {"status": "PASS"}),
        ):
            (run_dir / filename).write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
        return {"validation": {"status": "PASS"}}

    monkeypatch.setattr(
        scope_candidate,
        "rescore_prepared_cases",
        fake_rescore_prepared_cases,
    )
    contexts = tmp_path / "source/full/contexts.json"
    contexts.parent.mkdir(parents=True)
    contexts.write_text("{}", encoding="utf-8")
    result = scope_candidate.run_scope_candidate(
        config_path="configs/qwen3_cached_rescore_scope_v1.yaml",
        input_path=tmp_path / "input.json",
        prepared_contexts_path=contexts,
        run_dir=tmp_path / "candidate/full",
    )

    assert runner.create_predictor is original_factory
    assert captured["run_dir"] == tmp_path / "candidate/full"
    assert result["scope_candidate"]["case_api_network_calls"] == 0
    diagnostics = json.loads(
        (tmp_path / "candidate/full/scope_diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostics["candidate"] == "qwen3_cached_rescore_scope_v1"


def test_candidate_integration_uses_complete_context_without_http(
    tmp_path,
    monkeypatch,
):
    case = InferenceCase("case_scope_1", "Yêu cầu trả nợ")
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps(
            [{"case_id": case.case_id, "case_query": case.case_query}]
        ),
        encoding="utf-8",
    )
    contexts_path = tmp_path / "source/full/contexts.json"
    PreparedCaseStore(contexts_path).put(PreparedCase(case, [], [], 0))
    backend = SequenceBackend(
        [
            (
                '{"source_role":"COURT_DECISION","claim":"trả nợ",'
                '"accepted":"toàn bộ","scope":"ALL","ratio":1}'
            )
        ]
    )
    predictor = _predictor(backend)
    monkeypatch.setattr(
        scope_candidate,
        "create_scope_candidate_predictor",
        lambda config: predictor,
    )

    def forbidden_http_client(*args, **kwargs):
        raise AssertionError("Case Content API client must not be constructed")

    monkeypatch.setattr(runner, "CaseContentClient", forbidden_http_client)
    run_dir = tmp_path / "candidate/full"
    result = scope_candidate.run_scope_candidate(
        config_path="configs/qwen3_cached_rescore_scope_v1.yaml",
        input_path=input_path,
        prepared_contexts_path=contexts_path,
        run_dir=run_dir,
        corpus_path="data/raw/corpus_law_pub.json",
    )

    api_stats = json.loads(
        (run_dir / "api_stats.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (run_dir / "validation.json").read_text(encoding="utf-8")
    )
    submission = json.loads(
        (run_dir / "submission.json").read_text(encoding="utf-8")
    )
    assert api_stats["run_network_attempts"] == 0
    assert validation["status"] == "PASS"
    assert validation["scope_candidate"]["source_artifacts_immutable"]
    assert submission[0]["prediction"] == "A_WIN"
    assert result["network_contract"]["api_token_required"] is False


def test_candidate_refuses_to_write_inside_source_artifact_tree(tmp_path):
    source_dir = tmp_path / "private-candidate-v2" / "full"
    source_dir.mkdir(parents=True)
    contexts = source_dir / "contexts.checkpoint.json"
    contexts.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the source"):
        scope_candidate.run_scope_candidate(
            config_path="configs/qwen3_cached_rescore_scope_v1.yaml",
            input_path=tmp_path / "input.json",
            prepared_contexts_path=contexts,
            run_dir=source_dir / "scope-attempt",
        )
