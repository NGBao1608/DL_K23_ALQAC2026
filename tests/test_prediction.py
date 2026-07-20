import pytest

import alqac2026.prediction as prediction_module
from alqac2026.prediction import (
    DECISION_FIRST_SYSTEM_PROMPT,
    OutcomePredictor,
    build_user_prompt,
    parse_prediction,
    parse_structured_prediction,
)
from alqac2026.schemas import CaseEvidence, InferenceCase, LawEvidence, OutcomeLabel


class RepairBackend:
    def __init__(self):
        self.calls = 0

    def generate(self, system_prompt, user_prompt):
        self.calls += 1
        if self.calls == 1:
            return "not json"
        return (
            '{"main_claim":"x","accepted_scope":"toàn bộ",'
            '"acceptance_ratio":1.0,"reasoning":"Hợp lệ sau repair.",'
            '"label":"A_WIN"}'
        )


def test_parse_prediction_accepts_valid_json():
    label, reasoning = parse_prediction(
        '{"main_claim":"x","accepted_scope":"toàn bộ",'
        '"acceptance_ratio":1.0,"reasoning":"Tòa chấp nhận toàn bộ.",'
        '"label":"A_WIN"}'
    )
    assert label is OutcomeLabel.A_WIN
    assert reasoning


def test_parse_prediction_rejects_invalid_label():
    with pytest.raises(ValueError):
        parse_prediction(
            '{"main_claim":"x","accepted_scope":"x",'
            '"acceptance_ratio":null,"reasoning":"x","label":"UNKNOWN"}'
        )


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


def test_ratio_normalizes_an_inconsistent_partial_label():
    parsed = parse_structured_prediction(
        '{"main_claim":"x","accepted_scope":"một nửa",'
        '"acceptance_ratio":0.5,"reasoning":"Một nửa.",'
        '"label":"PARTIAL_A_WIN"}'
    )
    assert parsed.label is OutcomeLabel.PARTIAL_B_WIN
    assert parsed.inconsistent


class QueueBackend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    def generate(self, system_prompt, user_prompt):
        self.calls += 1
        return next(self.outputs)


def test_partial_prediction_runs_deterministic_verifier():
    backend = QueueBackend(
        [
            '{"main_claim":"x","accepted_scope":"một phần",'
            '"acceptance_ratio":0.4,"reasoning":"bước đầu",'
            '"label":"PARTIAL_B_WIN"}',
            '{"main_claim":"x","accepted_scope":"phần lớn",'
            '"acceptance_ratio":0.6,"reasoning":"đã kiểm tra",'
            '"label":"PARTIAL_A_WIN"}',
        ]
    )
    predictor = OutcomePredictor(backend)
    label, reasoning, raw = predictor.predict(
        InferenceCase("case_1", "query"), [], []
    )
    assert label is OutcomeLabel.PARTIAL_A_WIN
    assert reasoning == "đã kiểm tra"
    assert backend.calls == 2
    assert "---VERIFICATION---" in raw


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


class TokenCountingBackend:
    def __init__(self):
        self.tokenizer = WordTokenizer()
        self.prompts = []

    def count_input_tokens(self, system_prompt, user_prompt):
        return len((system_prompt + " " + user_prompt).split())

    def generate(self, system_prompt, user_prompt):
        self.prompts.append(user_prompt)
        return (
            '{"main_claim":"x","accepted_scope":"toàn bộ",'
            '"acceptance_ratio":1.0,"reasoning":"đủ","label":"A_WIN"}'
        )


def test_token_aware_context_preserves_final_instruction_and_budget():
    backend = TokenCountingBackend()
    predictor = OutcomePredictor(
        backend,
        max_input_tokens=400,
        verification_reserve_tokens=20,
    )
    predictor.predict(
        InferenceCase("case_1", "tranh chấp hợp đồng"),
        [CaseEvidence("opaque", "evidence " * 300, 1.0, "court_decision")],
        [LawEvidence("law", 1, "article " * 300, 1.0)],
    )
    prompt = backend.prompts[0]
    assert "Chỉ trả về JSON theo schema đã yêu cầu." in prompt
    assert backend.count_input_tokens(predictor.system_prompt, prompt) <= 400


def test_token_aware_verifier_also_stays_within_input_budget():
    class PartialTokenBackend(TokenCountingBackend):
        def generate(self, system_prompt, user_prompt):
            self.prompts.append(user_prompt)
            return (
                '{"main_claim":"x","accepted_scope":"một phần",'
                '"acceptance_ratio":0.4,"reasoning":"đủ",'
                '"label":"PARTIAL_B_WIN"}'
            )

    backend = PartialTokenBackend()
    predictor = OutcomePredictor(
        backend,
        max_input_tokens=400,
        verification_reserve_tokens=80,
    )
    predictor.predict(
        InferenceCase("case_1", "tranh chấp hợp đồng"),
        [CaseEvidence("opaque", "evidence " * 300, 1.0, "court_decision")],
        [LawEvidence("law", 1, "article " * 300, 1.0)],
    )
    assert len(backend.prompts) == 2
    assert all(
        backend.count_input_tokens(predictor.system_prompt, prompt) <= 400
        for prompt in backend.prompts
    )


def test_create_predictor_forwards_optional_adapter_without_loading_models(
    monkeypatch
):
    captured = {}

    class StubBackend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def generate(self, system_prompt, user_prompt):
            return '{"reasoning":"x","label":"A_WIN"}'

    monkeypatch.setattr(
        prediction_module, "TransformersQwenBackend", StubBackend
    )
    predictor = prediction_module.create_predictor(
        {
            "model_name": "Qwen/Qwen3-8B",
            "adapter_path": "/drive/adapter",
            "prompt_variant": "decision_first_v2",
        }
    )
    assert captured["adapter_path"] == "/drive/adapter"
    assert predictor.system_prompt == DECISION_FIRST_SYSTEM_PROMPT
