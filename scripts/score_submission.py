#!/usr/bin/env python3
"""Score any submission.json against the public gold, fully offline.

Only the public set has gold labels/provisions, so this reports outcome
accuracy and law micro F1 / recall. Case-evidence recall stays null because
the public file ships no official gold chunk ids.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from alqac2026.data import load_law_corpus, load_public_gold
from alqac2026.evaluation import _micro_f1, law_recall_at_k
from alqac2026.submission import load_submission, validate_submission
from alqac2026.data import load_inference_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline scorer vs public gold")
    parser.add_argument("--input", required=True, help="submission.json to score")
    parser.add_argument(
        "--test-data", default="data/raw/ALQAC2026_public_test.json"
    )
    parser.add_argument("--corpus", default="data/raw/corpus_law_pub.json")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Score even if the submission fails strict validation",
    )
    args = parser.parse_args()

    articles = load_law_corpus(args.corpus)
    gold = load_public_gold(args.test_data, articles)
    submission = load_submission(args.input)

    validation: dict = {"status": "SKIPPED"}
    if not args.skip_validation:
        cases = load_inference_cases(args.test_data)
        try:
            validation = validate_submission(submission, cases, articles)
        except ValueError as error:
            validation = {"status": "FAIL", "error": str(error)}

    scored = [item for item in submission if item["case_id"] in gold]
    accuracy = (
        sum(
            item["prediction"] == gold[item["case_id"]].verdict_label.value
            for item in scored
        )
        / len(scored)
        if scored
        else 0.0
    )

    eligible = [item for item in scored if gold[item["case_id"]].law_evidence]
    law_pred = [
        {(law["law_id"], law["aid"]) for law in item.get("law_evidence", [])}
        for item in eligible
    ]
    law_gold = [set(gold[item["case_id"]].law_evidence) for item in eligible]
    law_f1 = _micro_f1(law_pred, law_gold)
    recall5 = (
        sum(
            law_recall_at_k(
                [(law["law_id"], law["aid"]) for law in item.get("law_evidence", [])],
                set(gold[item["case_id"]].law_evidence),
                5,
            )
            for item in eligible
        )
        / len(eligible)
        if eligible
        else 0.0
    )

    seg_ids = sum(
        1
        for item in scored
        for chunk in item.get("case_evidence", [])
        if "_seg_" in chunk
    )
    chunk_ids = sum(
        1
        for item in scored
        for chunk in item.get("case_evidence", [])
        if "_chunk_" in chunk
    )

    report = {
        "input": str(Path(args.input)),
        "validation": validation,
        "scored_cases": len(scored),
        "outcome_accuracy": round(accuracy, 4),
        "law_micro_f1": round(law_f1, 4),
        "law_recall_at_5": round(recall5, 4),
        "case_evidence": {
            "seg_ids": seg_ids,
            "legacy_chunk_ids": chunk_ids,
            "recall": None,
            "note": "public file has no gold chunk ids; recall is leaderboard-only",
        },
        "partial_score_outcome_plus_law": round(0.70 * accuracy + 0.10 * law_f1, 4),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
