import pytest

import alqac2026.prediction as prediction_module
from alqac2026.pipeline import ALQACPipeline, CheckpointStore, PreparedCase
from alqac2026.prediction import (
    DECISION_FIRST_SYSTEM_PROMPT,
    OutcomePredictor,
    build_user_prompt,
    parse_prediction,
    parse_structured_prediction,
)
from alqac2026.schemas import (
    CaseEvidence,
    InferenceCase,
    LawEvidence,
    OutcomeLabel,
    PredictionResult,
)


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
            "max_input_tokens": 4096,
            "max_new_tokens": 192,
            "oom_max_input_tokens": 3072,
            "oom_max_new_tokens": 160,
            "attn_implementation": "sdpa",
            "cache_implementation": "offloaded",
            "context_law_top_k": 3,
            "oom_context_law_top_k": 2,
        }
    )
    assert captured["adapter_path"] == "/drive/adapter"
    assert captured["max_input_tokens"] == 4096
    assert captured["max_new_tokens"] == 192
    assert captured["oom_max_input_tokens"] == 3072
    assert captured["oom_max_new_tokens"] == 160
    assert captured["attn_implementation"] == "sdpa"
    assert captured["cache_implementation"] == "offloaded"
    assert predictor.system_prompt == DECISION_FIRST_SYSTEM_PROMPT
    assert predictor.context_law_top_k == 3
    assert predictor.oom_context_law_top_k == 2


def test_predictor_activates_compact_oom_profile():
    class ProfileBackend:
        def __init__(self):
            self.activated = 0

        def generate(self, system_prompt, user_prompt):
            raise AssertionError("not used")

        def activate_oom_recovery(self):
            self.activated += 1
            return True

    backend = ProfileBackend()
    predictor = OutcomePredictor(
        backend,
        max_input_tokens=4096,
        context_law_top_k=3,
        oom_max_input_tokens=3072,
        oom_context_law_top_k=2,
    )

    assert predictor.activate_oom_recovery()
    assert predictor.max_input_tokens == 3072
    assert predictor.context_law_top_k == 2
    assert backend.activated == 1


def test_transformers_backend_uses_offloaded_cache_for_cuda_generation(
    monkeypatch,
):
    import torch

    captured = {}

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        eos_token_id = 9

        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, *args, **kwargs):
            return Batch(input_ids=torch.tensor([[1, 2]]))

        def decode(self, tokens, skip_special_tokens=True):
            return "decoded"

    class Model:
        device = "cpu"

        def generate(self, **kwargs):
            captured.update(kwargs)
            return torch.tensor([[1, 2, 3]])

    backend = object.__new__(prediction_module.TransformersQwenBackend)
    backend.tokenizer = Tokenizer()
    backend.model = Model()
    backend.max_input_tokens = 4096
    backend.max_new_tokens = 192
    backend.cache_implementation = "offloaded"
    backend.thinking = False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert backend.generate("system", "user") == "decoded"
    assert captured["cache_implementation"] == "offloaded"
    assert captured["max_new_tokens"] == 192
    assert captured["do_sample"] is False


def test_case_prediction_failure_uses_submission_safe_operative_fallback():
    class FailingPredictor:
        def predict(self, case, case_evidence, law_evidence):
            raise ValueError("invalid model output")

    prepared = PreparedCase(
        case=InferenceCase("case_1", "Nguyên đơn yêu cầu trả nợ."),
        case_evidence=[
            CaseEvidence(
                "opaque",
                "Hội đồng xét xử quyết định không chấp nhận yêu cầu khởi kiện.",
                1.0,
                "operative_verdict",
            )
        ],
        law_evidence=[],
        api_calls=2,
    )
    pipeline = ALQACPipeline(
        None,
        None,
        FailingPredictor(),
        allow_prediction_fallback=True,
        prediction_fallback_label=OutcomeLabel.PARTIAL_B_WIN,
    )
    result = pipeline.predict_prepared(prepared)
    assert result.status == "completed"
    assert result.prediction is OutcomeLabel.B_WIN
    assert result.error == "PredictionFallback:ValueError"
    assert result.prediction_attempts == 1
    assert result.prediction_failure_types == ["ValueError"]


def test_case_prediction_retries_three_times_with_same_prepared_context():
    class RecoveringPredictor:
        def __init__(self):
            self.calls = 0
            self.inputs = []

        def predict(self, case, case_evidence, law_evidence):
            self.calls += 1
            self.inputs.append((case, case_evidence, law_evidence))
            if self.calls <= 3:
                raise RuntimeError("transient generation failure")
            return OutcomeLabel.A_WIN, "Recovered prediction.", '{"label":"A_WIN"}'

    prepared = PreparedCase(
        case=InferenceCase("case_1", "Nguyên đơn yêu cầu trả nợ."),
        case_evidence=[
            CaseEvidence(
                "opaque",
                "Tuyên xử: chấp nhận toàn bộ yêu cầu khởi kiện.",
                1.0,
                "operative_verdict",
            )
        ],
        law_evidence=[],
        api_calls=2,
    )
    predictor = RecoveringPredictor()
    retry_events = []
    pipeline = ALQACPipeline(
        None,
        None,
        predictor,
        allow_prediction_fallback=True,
        max_prediction_retries=3,
        prediction_retry_callback=retry_events.append,
    )

    result = pipeline.predict_prepared(prepared)

    assert result.status == "completed"
    assert result.prediction is OutcomeLabel.A_WIN
    assert result.error is None
    assert result.prediction_attempts == 4
    assert result.prediction_failure_types == ["RuntimeError"] * 3
    assert predictor.calls == 4
    assert retry_events == [
        {
            "case_id": "case_1",
            "failed_attempt": 1,
            "next_attempt": 2,
            "max_attempts": 4,
            "error_type": "RuntimeError",
        },
        {
            "case_id": "case_1",
            "failed_attempt": 2,
            "next_attempt": 3,
            "max_attempts": 4,
            "error_type": "RuntimeError",
        },
        {
            "case_id": "case_1",
            "failed_attempt": 3,
            "next_attempt": 4,
            "max_attempts": 4,
            "error_type": "RuntimeError",
        },
    ]
    assert all(
        case is prepared.case
        and case_evidence is prepared.case_evidence
        and law_evidence is prepared.law_evidence
        for case, case_evidence, law_evidence in predictor.inputs
    )


def test_case_prediction_uses_fallback_only_after_all_retries_are_exhausted():
    class FailingPredictor:
        def __init__(self):
            self.calls = 0

        def predict(self, case, case_evidence, law_evidence):
            self.calls += 1
            raise ValueError("still invalid")

    predictor = FailingPredictor()
    pipeline = ALQACPipeline(
        None,
        None,
        predictor,
        allow_prediction_fallback=True,
        max_prediction_retries=3,
    )
    result = pipeline.predict_prepared(
        PreparedCase(
            case=InferenceCase("case_1", "Nguyên đơn yêu cầu trả nợ."),
            case_evidence=[],
            law_evidence=[],
            api_calls=0,
        )
    )

    assert predictor.calls == 4
    assert result.status == "completed"
    assert result.prediction is OutcomeLabel.B_WIN
    assert result.error == "PredictionFallback:ValueError"
    assert result.prediction_attempts == 4
    assert result.prediction_failure_types == ["ValueError"] * 4


def test_cuda_oom_gets_one_compact_retry_with_same_prepared_context():
    class OutOfMemoryError(RuntimeError):
        pass

    class RecoveringPredictor:
        def __init__(self):
            self.calls = 0
            self.activated = 0
            self.inputs = []

        def predict(self, case, case_evidence, law_evidence):
            self.calls += 1
            self.inputs.append((case, case_evidence, law_evidence))
            if self.calls == 1:
                raise OutOfMemoryError("CUDA out of memory")
            return OutcomeLabel.A_WIN, "Recovered compactly.", "{}"

        def activate_oom_recovery(self):
            self.activated += 1
            return True

    prepared = PreparedCase(
        case=InferenceCase("case_1", "Nguyên đơn yêu cầu trả nợ."),
        case_evidence=[],
        law_evidence=[],
        api_calls=3,
    )
    predictor = RecoveringPredictor()
    retry_events = []
    pipeline = ALQACPipeline(
        None,
        None,
        predictor,
        allow_prediction_fallback=True,
        max_prediction_retries=3,
        max_oom_retries=1,
        prediction_retry_callback=retry_events.append,
    )

    result = pipeline.predict_prepared(prepared)

    assert result.status == "completed"
    assert result.prediction is OutcomeLabel.A_WIN
    assert result.prediction_attempts == 2
    assert result.prediction_failure_types == ["OutOfMemoryError"]
    assert predictor.calls == 2
    assert predictor.activated == 1
    assert retry_events[0]["error_type"] == "OutOfMemoryError"
    assert all(
        case is prepared.case
        and case_evidence is prepared.case_evidence
        and law_evidence is prepared.law_evidence
        for case, case_evidence, law_evidence in predictor.inputs
    )


def test_second_cuda_oom_falls_back_without_two_more_identical_retries():
    class OutOfMemoryError(RuntimeError):
        pass

    class FailingPredictor:
        def __init__(self):
            self.calls = 0
            self.activated = 0

        def predict(self, case, case_evidence, law_evidence):
            self.calls += 1
            raise OutOfMemoryError("CUDA out of memory")

        def activate_oom_recovery(self):
            self.activated += 1
            return True

    predictor = FailingPredictor()
    pipeline = ALQACPipeline(
        None,
        None,
        predictor,
        allow_prediction_fallback=True,
        max_prediction_retries=3,
        max_oom_retries=1,
    )

    result = pipeline.predict_prepared(
        PreparedCase(
            case=InferenceCase("case_1", "Nguyên đơn yêu cầu trả nợ."),
            case_evidence=[],
            law_evidence=[],
            api_calls=0,
        )
    )

    assert predictor.calls == 2
    assert predictor.activated == 1
    assert result.status == "completed"
    assert result.prediction is OutcomeLabel.B_WIN
    assert result.error == "PredictionFallback:OutOfMemoryError"
    assert result.prediction_attempts == 2
    assert result.prediction_failure_types == ["OutOfMemoryError"] * 2


@pytest.mark.parametrize("value", [-1, 4, True, 1.5])
def test_case_prediction_retry_cap_must_be_zero_to_three(value):
    with pytest.raises(ValueError, match="integer from 0 to 3"):
        ALQACPipeline(
            None,
            None,
            object(),
            max_prediction_retries=value,
        )


@pytest.mark.parametrize("value", [-1, 2, True, 1.5])
def test_oom_retry_cap_must_be_zero_or_one(value):
    with pytest.raises(ValueError, match="integer from 0 to 1"):
        ALQACPipeline(
            None,
            None,
            object(),
            max_oom_retries=value,
        )


def test_prediction_retry_metadata_survives_checkpoint_resume(tmp_path):
    store = CheckpointStore(tmp_path / "predictions.json")
    result = PredictionResult(
        case_id="case_1",
        prediction=OutcomeLabel.A_WIN,
        prediction_attempts=3,
        prediction_failure_types=["RuntimeError", "ValueError"],
    )
    store.put(result)

    restored = CheckpointStore(tmp_path / "predictions.json").get("case_1")

    assert restored is not None
    assert restored.prediction_attempts == 3
    assert restored.prediction_failure_types == ["RuntimeError", "ValueError"]
