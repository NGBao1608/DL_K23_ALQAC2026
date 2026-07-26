#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from alqac2026.data import load_inference_cases, load_law_corpus
from alqac2026.submission import load_submission, validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an ALQAC submission")
    parser.add_argument("--input", required=True, help="submission.json")
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--law-corpus", default="data/raw/corpus_law_pub.json")
    args = parser.parse_args()
    report = validate_submission(
        load_submission(args.input),
        load_inference_cases(args.test_data),
        load_law_corpus(args.law_corpus),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

