#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from alqac2026.case_retrieval import SQLiteEvidenceCache, build_api_plan
from alqac2026.config import load_config, write_json
from alqac2026.data import load_inference_cases
from alqac2026.query_planning import (
    DeterministicQueryComposer,
    StoredQueryPlan,
    create_query_planner,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan Case Content API calls without making network requests"
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--cache-db", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--approved-max-network-calls", type=int, default=None)
    args = parser.parse_args()
    if (
        args.approved_max_network_calls is not None
        and args.approved_max_network_calls < 0
    ):
        parser.error("--approved-max-network-calls must be non-negative")

    config = load_config(args.config)
    cases = load_inference_cases(args.input)
    if args.limit is not None:
        cases = cases[: args.limit]
    cache_path = args.cache_db or config["paths"]["cache_db"]
    planner = create_query_planner(config["case_retrieval"]["query_planner"])
    composer = DeterministicQueryComposer()
    query_plans = {}
    try:
        for case in cases:
            result = planner.plan(case.case_query)
            query_plans[case.case_id] = StoredQueryPlan(
                fingerprint="preview",
                planner_strategy=result.strategy,
                planner_failure_type=result.failure_type,
                plan=result.plan,
                queries=tuple(composer.compose(result.plan)),
            )
    finally:
        planner.release()
    cache = SQLiteEvidenceCache(cache_path)
    try:
        report = build_api_plan(
            cases,
            cache,
            query_plans=query_plans,
            max_logical_queries_per_case=int(
                config["case_retrieval"]["max_logical_queries_per_case"]
            ),
            approved_max_network_calls=args.approved_max_network_calls,
        )
    finally:
        cache.close()
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
