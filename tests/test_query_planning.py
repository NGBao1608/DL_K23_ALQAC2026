import json

import pytest

from alqac2026.query_planning import (
    DeterministicQueryComposer,
    DeterministicQueryPlanner,
    LLMAssistedQueryPlanner,
    PlannerDeadlineExceeded,
    QueryPlanStore,
    create_query_planner,
    evaluate_evidence_sufficiency,
)
from alqac2026.schemas import CaseEvidence, InferenceCase


CASE_QUERY = (
    "Nguyên đơn yêu cầu chia di sản thừa kế theo pháp luật, gồm quyền sử dụng "
    "thửa đất 107 diện tích 92,5m2 và tiền bồi thường."
)


class FakeBackend:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.calls = 0
        self.released = False

    def generate(self, system_prompt, user_prompt):
        self.calls += 1
        if self.error:
            raise self.error
        return self.output

    def release(self):
        self.released = True


def _valid_output():
    return json.dumps(
        {
            "case_type": "inheritance",
            "main_claim": "chia di sản thừa kế theo pháp luật",
            "requested_remedies": ["chia di sản thừa kế"],
            "legal_objects": ["thửa đất 107", "tiền bồi thường"],
            "amounts_or_areas": ["92,5m2"],
        },
        ensure_ascii=False,
    )


def test_llm_planner_valid_json_uses_llm_plan():
    backend = FakeBackend(_valid_output())
    result = LLMAssistedQueryPlanner(backend).plan(CASE_QUERY)
    assert result.strategy == "llm"
    assert result.failure_type is None
    assert result.plan.main_claim == "chia di sản thừa kế theo pháp luật"


@pytest.mark.parametrize(
    "error",
    [
        PlannerDeadlineExceeded("deadline"),
        OSError("model load failed"),
    ],
)
def test_planner_runtime_failure_falls_back_deterministically(error):
    result = LLMAssistedQueryPlanner(FakeBackend(error=error)).plan(CASE_QUERY)
    assert result.strategy == "deterministic_fallback"
    assert result.failure_type == type(error).__name__
    assert result.failure_code in {"generation_timeout", "model_load_failure"}
    assert result.plan.main_claim == "chia di sản thừa kế theo pháp luật"


@pytest.mark.parametrize(
    "output",
    [
        "not json",
        '{"case_type":"inheritance"}',
        json.dumps(
            {
                "case_type": "inheritance",
                "main_claim": "Tòa án chấp nhận toàn bộ yêu cầu",
                "requested_remedies": ["chia di sản thừa kế"],
                "legal_objects": ["thửa đất 107"],
                "amounts_or_areas": ["92,5m2"],
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "case_type": "inheritance",
                "main_claim": "chia di sản thừa kế theo pháp luật",
                "requested_remedies": [],
                "legal_objects": [],
                "amounts_or_areas": [],
            },
            ensure_ascii=False,
        ),
    ],
)
def test_invalid_missing_or_non_exact_llm_output_falls_back(output):
    result = LLMAssistedQueryPlanner(FakeBackend(output)).plan(CASE_QUERY)
    assert result.strategy == "deterministic_fallback"
    assert result.failure_type == "PlannerOutputError"
    assert result.failure_code in {
        "missing_json_object",
        "schema_mismatch",
        "ungrounded_span",
        "missing_requested_remedy",
    }


def test_deterministic_planner_same_input_same_output():
    planner = DeterministicQueryPlanner()
    assert planner.plan(CASE_QUERY) == planner.plan(CASE_QUERY)


def test_deterministic_main_claim_preserves_grouped_money():
    result = DeterministicQueryPlanner().plan(
        "Nguyên đơn yêu cầu bị đơn hoàn trả 142.800.000 đồng."
    )
    assert result.plan.main_claim == "bị đơn hoàn trả 142.800.000 đồng"


def test_qwen8_planner_configuration_enables_quantized_loading_without_model_load():
    planner = create_query_planner(
        {
            "strategy": "llm_assisted",
            "model_name": "Qwen/Qwen3-8B",
            "revision": "revision",
            "thinking": False,
            "do_sample": False,
            "load_in_4bit": True,
        }
    )
    assert planner.backend.model_name == "Qwen/Qwen3-8B"
    assert planner.backend.load_in_4bit is True


def test_query_composer_is_concise_deduplicated_and_bounded():
    plan = DeterministicQueryPlanner().plan(CASE_QUERY).plan
    queries = DeterministicQueryComposer().compose(plan)
    assert 2 <= len(queries) <= 3
    normalized = [query.text.casefold() for query in queries]
    assert len(normalized) == len(set(normalized))
    assert all(8 <= len(query.text.split()) <= 25 for query in queries)
    assert all(CASE_QUERY.casefold() != query.text.casefold() for query in queries)


def test_query_plan_artifact_reuses_same_fingerprint_without_regeneration(tmp_path):
    backend = FakeBackend(_valid_output())
    planner = LLMAssistedQueryPlanner(backend)
    store = QueryPlanStore(tmp_path / "query_plans.json")
    case = InferenceCase("case_1", CASE_QUERY)
    kwargs = {
        "planner": planner,
        "composer": DeterministicQueryComposer(),
        "configured_strategy": "llm_assisted",
        "model_revision": "revision",
    }
    first = store.get_or_create(case, **kwargs)
    second = store.get_or_create(case, **kwargs)
    assert first == second
    assert backend.calls == 1
    artifact = json.loads((tmp_path / "query_plans.json").read_text())
    components = artifact["case_1"]["fingerprint_components"]
    assert set(components) == {
        "case_id",
        "case_query_sha256",
        "planner_strategy",
        "planner_model_revision",
        "planner_prompt_version",
        "composer_version",
    }
    assert len(components["case_query_sha256"]) == 64
    assert artifact["case_1"]["planner_failure_code"] is None


def test_evidence_gate_rejects_duplicate_or_party_role_evidence():
    plan = DeterministicQueryPlanner().plan(CASE_QUERY).plan
    party_text = (
        "Đại diện Viện kiểm sát đề nghị Tòa án chấp nhận yêu cầu chia di sản "
        "thừa kế đối với thửa đất 107."
    )
    result = evaluate_evidence_sufficiency(
        [
            CaseEvidence("duplicate", party_text, 1.0, "operative_verdict"),
            CaseEvidence("duplicate", party_text, 0.9, "remedy_scope"),
        ],
        plan,
    )
    assert not result.sufficient
    assert result.has_duplicate_chunk_id
    assert "duplicate_chunk_id" in result.reasons
