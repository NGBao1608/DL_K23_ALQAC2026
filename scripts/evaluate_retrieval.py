#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from alqac2026.config import load_config, write_json
from alqac2026.data import load_inference_cases, load_law_corpus, load_public_gold
from alqac2026.evaluation import evaluate_law_rankings
from alqac2026.law_retrieval import create_law_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate public law retrieval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    articles = load_law_corpus(config["paths"]["corpus"])
    cases = load_inference_cases(config["paths"]["public_test"])
    if args.limit is not None:
        cases = cases[: args.limit]
    gold = load_public_gold(config["paths"]["public_test"], articles)
    settings = dict(config["law_retrieval"])
    settings["index_dir"] = config["paths"]["law_index"]
    retriever = create_law_retriever(articles, settings)
    rankings = {
        case.case_id: [
            (item.law_id, item.aid)
            for item in retriever.search(case.case_query, top_k=10)
        ]
        for case in cases
    }
    metrics = evaluate_law_rankings(rankings, gold, ks=(5, 10))
    report = {
        "config": str(Path(args.config)),
        "strategy": settings["strategy"],
        "metrics": metrics,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

