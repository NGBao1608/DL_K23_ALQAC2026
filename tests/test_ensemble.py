import numpy as np

from alqac2026.ensemble import (
    OutcomeConfig,
    PrecedentBank,
    _ADJ_SYSTEM,
    _adjudicator_user,
    baseline_user,
    evaluate,
    parse_label,
    predict_outcome,
)
from alqac2026.prediction import SYSTEM_PROMPT


def test_base_prompt_matches_single_pass_predictor():
    # With no signals enabled the ensemble base prompt must match the single-pass
    # predictor exactly, so greedy decoding yields the same no-signal prediction.
    assert _ADJ_SYSTEM == SYSTEM_PROMPT
    case = {"case_id": "case_1", "query": "tranh chấp đất", "ev": "đoạn bằng chứng"}
    expected = (
        "## Vụ án\ncase_id: case_1\ncase_query: tranh chấp đất\n\n"
        "## Bằng chứng truy hồi (đoạn quyết định ưu tiên lên đầu)\nđoạn bằng chứng\n\n"
        "Xác định yêu cầu chính rồi chọn đúng một nhãn. Chỉ trả JSON."
    )
    assert baseline_user(case) == expected
    assert _adjudicator_user(case, "", "", "", "") == expected  # no toggles == base


def test_enabled_signal_changes_the_prompt():
    case = {"case_id": "c1", "query": "q", "ev": "e"}
    assert _adjudicator_user(case, "[án lệ 1]", "", "", "") != baseline_user(case)
    assert "Viện Kiểm Sát" in _adjudicator_user(case, "", "đề nghị một phần", "", "")


class FakeEmbedder:
    # "q<i>" -> one-hot at i in a fixed-dim space, so an arbitrary query embeds to
    # the same space as the bank (works for queries absent from the bank too).
    DIM = 10

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        vecs = np.zeros((len(texts), self.DIM), dtype="float32")
        for j, text in enumerate(texts):
            digits = "".join(ch for ch in text if ch.isdigit()) or "0"
            vecs[j, int(digits) % self.DIM] = 1.0
        return vecs


CASES = [
    {"case_id": f"c{i}", "query": f"q{i}", "gold": "A_WIN", "ev": "evidence", "reasoning": "r"}
    for i in range(4)
]


def test_parse_label_whole_token():
    assert parse_label("suy luận...\nNHÃN: PARTIAL_A_WIN") == "PARTIAL_A_WIN"
    assert parse_label("kết luận cuối là B_WIN") == "B_WIN"  # not PARTIAL_B_WIN
    assert parse_label("không có nhãn nào") is None


def test_precedent_excludes_self():
    bank = PrecedentBank(FakeEmbedder(), CASES)
    got = bank.retrieve("q0", k=2, exclude_case_id="c0")
    assert all(p["case_id"] != "c0" for p in got)
    assert len(got) == 2
    assert "Vụ tương tự" in bank.block("q0", 2, exclude_case_id="c0")
    # A query absent from the bank (e.g. a private case) still retrieves.
    assert bank.retrieve("q7", k=1)


def test_self_consistency_majority_vote():
    outputs = iter(["NHÃN: A_WIN", "NHÃN: A_WIN", "NHÃN: B_WIN"])

    def gen(system, user, *, thinking=False, temperature=None, max_new=512):
        return next(outputs)

    label = predict_outcome(
        CASES[0], gen, OutcomeConfig(self_consistency=3), precedent_bank=None
    )
    assert label == "A_WIN"


def test_toggles_control_prompt_and_calls():
    seen = {"vks": 0, "debate": 0}

    def gen(system, user, *, thinking=False, temperature=None, max_new=512):
        if "VIỆN KIỂM SÁT" in system:
            seen["vks"] += 1
            return '{"trich":"x","lap_truong":"MOT_PHAN"}'
        if "luật sư" in system:
            seen["debate"] += 1
            return "lập luận"
        # adjudicator: the VKS partial hint must have reached the prompt
        assert "một phần" in user
        return "NHÃN: PARTIAL_A_WIN"

    bank = PrecedentBank(FakeEmbedder(), CASES)
    label = predict_outcome(
        CASES[0],
        gen,
        OutcomeConfig(use_precedents=True, use_vks_stance=True, use_debate=True),
        precedent_bank=bank,
    )
    assert label == "PARTIAL_A_WIN"
    assert seen["vks"] == 1
    assert seen["debate"] == 2  # one advocate per side


def test_falls_back_to_greedy_baseline_when_samples_unparseable():
    # The adjudicator (with signals) returns garbage; the greedy baseline fallback
    # must still yield a valid label — never a crash, never a blind majority guess.
    n_baseline_calls = {"n": 0}

    def gen(system, user, *, thinking=False, temperature=None, max_new=512):
        # baseline fallback is the one call with no extra context block
        if "## Các vụ" not in user and "Viện Kiểm Sát" not in user:
            n_baseline_calls["n"] += 1
            return '{"reasoning":"x","label":"B_WIN"}'
        return "rác không parse được"

    def gen_n(system, user, *, n, temperature=0.7, thinking=False, max_new=512):
        return ["rác"] * n  # all samples unparseable

    bank = PrecedentBank(FakeEmbedder(), CASES)
    label = predict_outcome(
        CASES[0],
        gen,
        OutcomeConfig(self_consistency=5, use_precedents=True),
        precedent_bank=bank,
        gen_n=gen_n,
    )
    assert label == "B_WIN"  # from the greedy baseline fallback
    assert n_baseline_calls["n"] == 1


def test_ensemble_predictor_matches_pipeline_interface_and_falls_back():
    from alqac2026.ensemble import EnsembleOutcomePredictor, evidence_text
    from alqac2026.schemas import CaseEvidence, InferenceCase, OutcomeLabel

    evidence = [
        CaseEvidence("case_1_seg_a", "Tòa tuyên xử chấp nhận một phần", 5.0, "court_decision"),
        CaseEvidence("case_1_seg_b", "trình bày của nguyên đơn", 3.0, "original"),
    ]
    # decision-first ordering: the ruling chunk leads.
    assert evidence_text(evidence).startswith("[court_decision]")

    class Backend:
        def generate(self, system, user, temperature=None, thinking=None, max_new_tokens=None):
            return '{"reasoning":"x","label":"PARTIAL_A_WIN"}'

        def generate_samples(self, system, user, n, temperature=0.7, thinking=None, max_new_tokens=None):
            return ['{"reasoning":"x","label":"PARTIAL_A_WIN"}'] * n

    predictor = EnsembleOutcomePredictor(
        Backend(), OutcomeConfig(self_consistency=5), precedent_bank=None
    )
    case = InferenceCase("case_1", "tranh chấp đất")
    label, reasoning, _ = predictor.predict(case, evidence, [])
    assert label is OutcomeLabel.PARTIAL_A_WIN

    class BrokenBackend:
        def generate(self, *a, **k):
            raise RuntimeError("boom")

        def generate_samples(self, *a, **k):
            raise RuntimeError("boom")

    class Fallback:
        def predict(self, case, case_evidence, law_evidence):
            return OutcomeLabel.B_WIN, "fallback", ""

    safe = EnsembleOutcomePredictor(
        BrokenBackend(), OutcomeConfig(self_consistency=5), fallback=Fallback()
    )
    label, reasoning, _ = safe.predict(case, evidence, [])
    assert label is OutcomeLabel.B_WIN and reasoning == "fallback"


def test_evaluate_reports_accuracy():
    def gen(system, user, *, thinking=False, temperature=None, max_new=512):
        return "NHÃN: A_WIN"

    report = evaluate(CASES, gen, OutcomeConfig(), precedent_bank=None)
    assert report["accuracy"] == 1.0  # every gold is A_WIN
