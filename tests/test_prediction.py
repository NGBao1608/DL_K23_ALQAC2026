import pytest

from alqac2026.prediction import (
    DECISION_FIRST_SYSTEM_PROMPT,
    OutcomePredictor,
    build_user_prompt,
    parse_prediction,
)
from alqac2026.schemas import CaseEvidence, InferenceCase, LawEvidence, OutcomeLabel


class RepairBackend:
    def __init__(self):
        self.calls = 0

    def generate(self, system_prompt, user_prompt):
        self.calls += 1
        if self.calls == 1:
            return "not json"
        return '{"reasoning":"Hợp lệ sau repair.","label":"A_WIN"}'


def test_parse_prediction_accepts_valid_json():
    label, reasoning = parse_prediction(
        '{"reasoning":"Tòa chấp nhận toàn bộ.","label":"A_WIN"}'
    )
    assert label is OutcomeLabel.A_WIN
    assert reasoning


def test_parse_prediction_rejects_invalid_label():
    with pytest.raises(ValueError):
        parse_prediction('{"reasoning":"x","label":"UNKNOWN"}')


def test_predictor_repairs_exactly_once():
    backend = RepairBackend()
    predictor = OutcomePredictor(backend)
    label, _, raw = predictor.predict(InferenceCase("case_1", "query"), [], [])
    assert label is OutcomeLabel.A_WIN
    assert backend.calls == 2
    assert "---REPAIR---" in raw


def test_decision_first_prompt_prioritizes_main_claim_and_decision_evidence():
    assert "yêu cầu chính" in DECISION_FIRST_SYSTEM_PROMPT
    assert "Tuyên xử" in DECISION_FIRST_SYSTEM_PROMPT
    assert "án phí" in DECISION_FIRST_SYSTEM_PROMPT
    assert "lớn hơn 50%" in DECISION_FIRST_SYSTEM_PROMPT


def test_candidate_context_uses_configured_character_budgets():
    prompt = build_user_prompt(
        InferenceCase("case_1", "tranh chấp hợp đồng"),
        [CaseEvidence("opaque", "A" * 2500, 1.0, "court_decision")],
        [LawEvidence("law", 1, "B" * 1500, 1.0)],
        case_evidence_chars=2200,
        law_evidence_chars=1200,
    )
    assert "A" * 2200 in prompt
    assert "A" * 2201 not in prompt
    assert "B" * 1200 in prompt
    assert "B" * 1201 not in prompt
