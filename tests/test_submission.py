import pytest

from alqac2026.data import load_inference_cases, load_law_corpus
from alqac2026.schemas import CaseEvidence, LawEvidence, OutcomeLabel, PredictionResult
from alqac2026.submission import (
    MAX_SUBMISSION_BYTES,
    build_submission,
    load_submission,
    validate_submission,
)


def _valid_submission():
    cases = load_inference_cases("data/raw/ALQAC2026_public_test.json")[:1]
    corpus = load_law_corpus("data/raw/corpus_law_pub.json")
    law = corpus[0]
    result = PredictionResult(
        case_id=cases[0].case_id,
        prediction=OutcomeLabel.A_WIN,
        case_evidence=[
            CaseEvidence(
                "case_4101_seg_9b2898839a509e4f", "evidence", 1.0
            )
        ],
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


@pytest.mark.parametrize(
    "identifier",
    [
        "case_4101_seg_9b2898839a509e4f",
        "opaque-id-with-no-case-prefix",
        "case_4101_chunk_2",
    ],
)
def test_case_evidence_identifier_is_opaque_and_preserved(identifier):
    submission, cases, corpus = _valid_submission()
    submission[0]["case_evidence"] = [identifier]
    assert validate_submission(submission, cases, corpus)["status"] == "PASS"
    assert submission[0]["case_evidence"] == [identifier]


@pytest.mark.parametrize("identifier", ["", "   ", None, 123])
def test_invalid_case_evidence_identifier_is_rejected(identifier):
    submission, cases, corpus = _valid_submission()
    submission[0]["case_evidence"] = [identifier]
    with pytest.raises(ValueError, match="non-empty strings"):
        validate_submission(submission, cases, corpus)


def test_duplicate_case_evidence_is_rejected():
    submission, cases, corpus = _valid_submission()
    identifier = submission[0]["case_evidence"][0]
    submission[0]["case_evidence"] = [identifier, identifier]
    with pytest.raises(ValueError, match="Duplicate case evidence"):
        validate_submission(submission, cases, corpus)


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


@pytest.mark.parametrize(
    "law",
    [
        {"law_id": "", "aid": 1},
        {"law_id": 123, "aid": 1},
        {"law_id": "47/2010/QH12", "aid": "1"},
        {"law_id": "47/2010/QH12", "aid": 1.0},
        {"law_id": "47/2010/QH12", "aid": True},
    ],
)
def test_invalid_law_types_are_rejected(law):
    submission, cases, corpus = _valid_submission()
    submission[0]["law_evidence"] = [law]
    with pytest.raises(ValueError, match="Invalid law evidence types"):
        validate_submission(submission, cases, corpus)


def test_invalid_case_id_type_is_rejected():
    submission, cases, corpus = _valid_submission()
    submission[0]["case_id"] = 123
    with pytest.raises(ValueError, match="invalid case_id"):
        validate_submission(submission, cases, corpus)


def test_submission_size_limit_is_enforced(tmp_path):
    path = tmp_path / "submission.json"
    path.write_bytes(b" " * (MAX_SUBMISSION_BYTES + 1))
    with pytest.raises(ValueError, match="10 MB"):
        load_submission(path)
