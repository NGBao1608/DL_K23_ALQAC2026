import pytest

from alqac2026.data import load_inference_cases, load_law_corpus
from alqac2026.schemas import CaseEvidence, LawEvidence, OutcomeLabel, PredictionResult
from alqac2026.submission import build_submission, validate_submission


def _valid_submission():
    cases = load_inference_cases("data/raw/ALQAC2026_public_test.json")[:1]
    corpus = load_law_corpus("data/raw/corpus_law_pub.json")
    law = corpus[0]
    result = PredictionResult(
        case_id=cases[0].case_id,
        prediction=OutcomeLabel.A_WIN,
        law_evidence=[LawEvidence(law.law_id, law.aid, law.text, 1.0)],
    )
    return build_submission([result]), cases, corpus


def test_valid_submission_passes():
    submission, cases, corpus = _valid_submission()
    assert validate_submission(submission, cases, corpus)["status"] == "PASS"
    assert set(submission[0]) == {
        "case_id",
        "prediction",
        "case_evidence",
        "law_evidence",
    }


def test_extra_field_is_rejected():
    submission, cases, corpus = _valid_submission()
    submission[0]["api_calls"] = 8
    with pytest.raises(ValueError, match="invalid fields"):
        validate_submission(submission, cases, corpus)


def test_invalid_law_is_rejected():
    submission, cases, corpus = _valid_submission()
    submission[0]["law_evidence"] = [{"law_id": "bad", "aid": -1}]
    with pytest.raises(ValueError, match="Unknown law evidence"):
        validate_submission(submission, cases, corpus)


def test_official_seg_chunk_id_passes():
    cases = load_inference_cases("data/raw/ALQAC2026_public_test.json")[:1]
    corpus = load_law_corpus("data/raw/corpus_law_pub.json")
    case_id = cases[0].case_id
    result = PredictionResult(
        case_id=case_id,
        prediction=OutcomeLabel.A_WIN,
        case_evidence=[
            CaseEvidence(f"{case_id}_seg_7c1f5f485bd7c7e7", "x", 1.0, "court_decision")
        ],
        law_evidence=[LawEvidence(corpus[0].law_id, corpus[0].aid, corpus[0].text, 1.0)],
    )
    submission = build_submission([result])
    assert submission[0]["case_evidence"] == [f"{case_id}_seg_7c1f5f485bd7c7e7"]
    assert validate_submission(submission, cases, corpus)["status"] == "PASS"


def test_chunk_id_from_other_case_is_rejected():
    submission, cases, corpus = _valid_submission()
    submission[0]["case_evidence"] = ["case_9999_seg_deadbeef"]
    with pytest.raises(ValueError, match="Invalid chunk prefix"):
        validate_submission(submission, cases, corpus)

