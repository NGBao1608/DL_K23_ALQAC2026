from __future__ import annotations

from collections.abc import Iterable

from .schemas import PredictionResult, PublicGold


def _micro_f1(predicted: list[set], gold: list[set]) -> float:
    true_positive = sum(len(pred & expected) for pred, expected in zip(predicted, gold))
    false_positive = sum(len(pred - expected) for pred, expected in zip(predicted, gold))
    false_negative = sum(len(expected - pred) for pred, expected in zip(predicted, gold))
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def law_recall_at_k(predicted: list[tuple[str, int]], gold: set, k: int) -> float:
    if not gold:
        raise ValueError("Law recall is undefined for an empty gold set")
    return len(set(predicted[:k]) & gold) / len(gold)


def evaluate_law_rankings(
    rankings: dict[str, list[tuple[str, int]]],
    gold_by_case: dict[str, PublicGold],
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, float | int]:
    eligible = [
        case_id
        for case_id in rankings
        if case_id in gold_by_case and gold_by_case[case_id].law_evidence
    ]
    result: dict[str, float | int] = {"evaluated_cases": len(eligible)}
    for k in ks:
        recalls = [
            law_recall_at_k(
                rankings[case_id], set(gold_by_case[case_id].law_evidence), k
            )
            for case_id in eligible
        ]
        result[f"law_recall_at_{k}"] = sum(recalls) / len(recalls) if recalls else 0.0
    return result


def evaluate_public(
    results: Iterable[PredictionResult],
    gold_by_case: dict[str, PublicGold],
    *,
    law_top_k: int | None = None,
) -> dict[str, float | int | None]:
    values = list(results)
    missing = [item.case_id for item in values if item.case_id not in gold_by_case]
    if missing:
        raise ValueError(f"Predictions contain unknown public case IDs: {missing[:3]}")
    accuracy = sum(
        item.prediction == gold_by_case[item.case_id].verdict_label for item in values
    ) / len(values) if values else 0.0
    law_eligible = [
        item for item in values if gold_by_case[item.case_id].law_evidence
    ]
    law_predictions = [
        {
            (law.law_id, law.aid)
            for law in (
                item.law_evidence[:law_top_k]
                if law_top_k is not None
                else item.law_evidence
            )
        }
        for item in law_eligible
    ]
    law_gold = [
        set(gold_by_case[item.case_id].law_evidence) for item in law_eligible
    ]
    law_f1 = _micro_f1(law_predictions, law_gold)
    law_recall_5_values = [
        law_recall_at_k(
            [(law.law_id, law.aid) for law in item.law_evidence],
            set(gold_by_case[item.case_id].law_evidence),
            5,
        )
        for item in law_eligible
    ]

    has_case_gold = bool(values) and all(
        gold_by_case[item.case_id].case_evidence for item in values
    )
    case_recall = None
    if has_case_gold:
        recalls = []
        for item in values:
            predicted = {evidence.chunk_id for evidence in item.case_evidence}
            expected = set(gold_by_case[item.case_id].case_evidence)
            recalls.append(len(predicted & expected) / len(expected))
        case_recall = sum(recalls) / len(recalls)

    final_score = (
        0.70 * accuracy + 0.20 * case_recall + 0.10 * law_f1
        if case_recall is not None
        else None
    )
    return {
        "cases": len(values),
        "outcome_accuracy": accuracy,
        "law_micro_f1": law_f1,
        "law_evaluated_cases": len(law_eligible),
        "law_recall_at_5": (
            sum(law_recall_5_values) / len(law_recall_5_values)
            if law_recall_5_values
            else 0.0
        ),
        "case_evidence_recall": case_recall,
        "final_score": final_score,
        "format_failures": sum(item.status != "completed" for item in values),
        "api_calls": sum(item.api_calls for item in values),
    }


def select_law_top_k(
    results: Iterable[PredictionResult],
    gold_by_case: dict[str, PublicGold],
    ks: tuple[int, ...] = tuple(range(3, 11)),
) -> dict:
    values = list(results)
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must contain positive values")
    scores = {
        str(k): float(evaluate_public(values, gold_by_case, law_top_k=k)["law_micro_f1"])
        for k in sorted(set(ks))
    }
    best_k = min(
        (int(k) for k, score in scores.items() if score == max(scores.values())),
        default=min(ks),
    )
    return {
        "schema_version": "law-top-k-v1",
        "submission_law_top_k": best_k,
        "selection_metric": "public_law_micro_f1",
        "tie_breaker": "smaller_k",
        "scores": scores,
    }


def build_error_analysis(
    results: Iterable[PredictionResult], gold_by_case: dict[str, PublicGold]
) -> list[dict]:
    errors = []
    for item in results:
        gold = gold_by_case[item.case_id]
        if item.prediction != gold.verdict_label or item.status != "completed":
            errors.append(
                {
                    "case_id": item.case_id,
                    "gold_label": gold.verdict_label.value,
                    "predicted_label": item.prediction.value if item.prediction else None,
                    "status": item.status,
                    "error": item.error,
                    "reasoning": item.reasoning,
                    "case_evidence_count": len(item.case_evidence),
                    "law_evidence": [
                        {"law_id": law.law_id, "aid": law.aid}
                        for law in item.law_evidence
                    ],
                }
            )
    return errors
