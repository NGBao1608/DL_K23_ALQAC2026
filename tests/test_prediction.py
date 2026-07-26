import pytest

from alqac2026.prediction import OutcomePredictor, parse_prediction
from alqac2026.schemas import InferenceCase, OutcomeLabel


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

