from __future__ import annotations

import json
from pathlib import Path

from .schemas import InferenceCase, LawArticle, OutcomeLabel, PredictionResult


SUBMISSION_FIELDS = {"case_id", "prediction", "case_evidence", "law_evidence"}
MAX_SUBMISSION_BYTES = 10 * 1024 * 1024


def build_submission(
    results: list[PredictionResult], law_top_k: int | None = None
) -> list[dict]:
    if law_top_k is not None and law_top_k <= 0:
        raise ValueError("law_top_k must be positive")
    submission = []
    for result in results:
        if result.status != "completed" or result.prediction is None:
            raise ValueError(f"Cannot submit failed prediction: {result.case_id}")
        submission.append(
            {
                "case_id": result.case_id,
                "prediction": result.prediction.value,
                "case_evidence": list(
                    dict.fromkeys(item.chunk_id for item in result.case_evidence)
                ),
                "law_evidence": [
                    {"law_id": item.law_id, "aid": item.aid}
                    for item in list(_unique_law_evidence(result))[:law_top_k]
                ],
            }
        )
    return submission


def _unique_law_evidence(result: PredictionResult):
    seen = set()
    for item in result.law_evidence:
        key = (item.law_id, item.aid)
        if key not in seen:
            seen.add(key)
            yield item


def validate_submission(
    submission: list[dict],
    cases: list[InferenceCase],
    corpus: list[LawArticle],
) -> dict[str, int | str]:
    errors: list[str] = []
    expected_ids = {case.case_id for case in cases}
    submitted_ids = [
        item.get("case_id")
        for item in submission
        if isinstance(item, dict) and isinstance(item.get("case_id"), str)
    ]
    if len(submitted_ids) != len(set(submitted_ids)):
        errors.append("Duplicate case_id")
    missing = expected_ids - set(submitted_ids)
    extra = set(submitted_ids) - expected_ids
    if missing:
        errors.append(f"Missing case_id: {sorted(missing)[:3]}")
    if extra:
        errors.append(f"Unknown case_id: {sorted(extra)[:3]}")
    valid_labels = {label.value for label in OutcomeLabel}
    valid_laws = {(article.law_id, article.aid) for article in corpus}

    for index, item in enumerate(submission):
        if not isinstance(item, dict):
            errors.append(f"Item {index} must be a JSON object")
            continue
        if set(item) != SUBMISSION_FIELDS:
            errors.append(f"Item {index} has invalid fields: {sorted(set(item))}")
            continue
        if not isinstance(item["case_id"], str) or not item["case_id"].strip():
            errors.append(f"Item {index} has invalid case_id")
            continue
        case_id = item["case_id"]
        if item["prediction"] not in valid_labels:
            errors.append(f"Invalid label for {case_id}")
        if not isinstance(item["case_evidence"], list) or not all(
            isinstance(value, str) and bool(value.strip())
            for value in item["case_evidence"]
        ):
            errors.append(
                f"case_evidence must contain only non-empty strings for {case_id}"
            )
        else:
            if len(item["case_evidence"]) != len(set(item["case_evidence"])):
                errors.append(f"Duplicate case evidence for {case_id}")
        if not isinstance(item["law_evidence"], list):
            errors.append(f"law_evidence must be a list for {case_id}")
        else:
            pairs = []
            for law in item["law_evidence"]:
                if not isinstance(law, dict) or set(law) != {"law_id", "aid"}:
                    errors.append(f"Invalid law evidence shape for {case_id}")
                    continue
                if (
                    not isinstance(law["law_id"], str)
                    or not law["law_id"].strip()
                    or not isinstance(law["aid"], int)
                    or isinstance(law["aid"], bool)
                ):
                    errors.append(f"Invalid law evidence types for {case_id}")
                    continue
                pair = (law["law_id"], law["aid"])
                pairs.append(pair)
                if pair not in valid_laws:
                    errors.append(f"Unknown law evidence {pair} for {case_id}")
            if len(pairs) != len(set(pairs)):
                errors.append(f"Duplicate law evidence for {case_id}")
    try:
        json.dumps(submission, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        errors.append(f"Submission is not strict JSON: {error}")
    if errors:
        raise ValueError("Submission validation failed:\n- " + "\n- ".join(errors))
    return {"status": "PASS", "cases": len(submission), "errors": 0}


def load_submission(path: str | Path) -> list[dict]:
    source = Path(path)
    if source.stat().st_size > MAX_SUBMISSION_BYTES:
        raise ValueError("Submission exceeds the official 10 MB size limit")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("Submission root must be a JSON array")
    return value
