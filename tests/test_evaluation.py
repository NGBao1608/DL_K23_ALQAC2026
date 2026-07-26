from alqac2026.evaluation import evaluate_law_rankings, evaluate_public
from alqac2026.schemas import (
    LawEvidence,
    OutcomeLabel,
    PredictionResult,
    PublicGold,
)


def test_law_recall_at_5_and_10():
    gold = {
        "case_1": PublicGold(
            "case_1",
            OutcomeLabel.A_WIN,
            law_evidence=(("law", 1), ("law", 10)),
        )
    }
    ranking = {"case_1": [("law", index) for index in range(1, 11)]}
    metrics = evaluate_law_rankings(ranking, gold)
    assert metrics["law_recall_at_5"] == 0.5
    assert metrics["law_recall_at_10"] == 1.0


def test_public_law_f1_excludes_cases_without_resolved_gold():
    gold = {
        "case_1": PublicGold(
            "case_1", OutcomeLabel.A_WIN, law_evidence=(("law", 1),)
        ),
        "case_2": PublicGold("case_2", OutcomeLabel.B_WIN),
    }
    results = [
        PredictionResult(
            "case_1",
            OutcomeLabel.A_WIN,
            law_evidence=[LawEvidence("law", 1, "", 1.0)],
        ),
        PredictionResult(
            "case_2",
            OutcomeLabel.B_WIN,
            law_evidence=[LawEvidence("law", 999, "", 1.0)],
        ),
    ]
    metrics = evaluate_public(results, gold)
    assert metrics["law_micro_f1"] == 1.0
    assert metrics["law_evaluated_cases"] == 1

