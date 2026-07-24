#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from alqac2026.config import (
    build_environment,
    load_config,
    sha256_file,
    source_fingerprint,
    write_json,
)
from alqac2026.data import load_inference_cases, load_law_corpus
from alqac2026.law_retrieval import HybridLawRetriever, create_law_retriever
from alqac2026.pipeline import ALQACPipeline, PreparedCase
from alqac2026.prediction import create_predictor
from alqac2026.query_planning import create_query_planner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the complete local model stack without ALQAC API calls"
    )
    parser.add_argument("--config", default="configs/candidate.yaml")
    parser.add_argument("--input", default="data/raw/ALQAC2026_public_test.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--law-index-dir", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    environment = build_environment()
    report = {
        "status": "running",
        "api_network_attempts": 0,
        "environment": environment,
        "source_sha256": source_fingerprint(
            Path(__file__).resolve().parents[1] / "src" / "alqac2026"
        ),
        "config_sha256": sha256_file(args.config),
        "stages": {},
    }
    write_json(output, report)
    original_excepthook = sys.excepthook

    def record_failure(error_type, error, traceback) -> None:
        report["status"] = "failed"
        report["error"] = f"{error_type.__name__}: {error}"
        write_json(output, report)
        original_excepthook(error_type, error, traceback)

    sys.excepthook = record_failure
    if not args.allow_cpu and not environment["cuda"]["available"]:
        raise RuntimeError("A CUDA GPU is required for the Colab candidate check")

    config = load_config(args.config)
    if args.law_index_dir:
        config["paths"]["law_index"] = args.law_index_dir
    if args.adapter_path:
        config["prediction"]["adapter_path"] = args.adapter_path
    articles = load_law_corpus(config["paths"]["corpus"])
    case = load_inference_cases(args.input)[0]

    started = time.perf_counter()
    planner = create_query_planner(config["case_retrieval"]["query_planner"])
    try:
        planner_result = planner.plan(case.case_query)
    finally:
        planner.release()
    if not planner_result.plan.main_claim:
        raise RuntimeError("Runtime planner and fallback did not produce main_claim")
    if (
        config["case_retrieval"]["query_planner"]["strategy"] == "llm_assisted"
        and planner_result.strategy != "llm"
    ):
        raise RuntimeError(
            "Runtime LLM query planner did not pass validation: "
            f"{planner_result.failure_code or planner_result.failure_type}"
        )
    report["stages"]["query_planner"] = {
        "status": "PASS",
        "seconds": round(time.perf_counter() - started, 3),
        "strategy": planner_result.strategy,
        "failure_type": planner_result.failure_type,
        "failure_code": planner_result.failure_code,
    }

    law_config = dict(config["law_retrieval"])
    law_config["strategy"] = "hybrid_rerank"
    law_config["index_dir"] = config["paths"]["law_index"]
    started = time.perf_counter()
    law_retriever = create_law_retriever(articles, law_config)
    if not isinstance(law_retriever, HybridLawRetriever):
        raise ValueError("Runtime check requires the hybrid embedding/reranker stack")
    candidate_indices = law_retriever.retrieve_candidate_indices(case.case_query)
    law_retriever.release_embedding_model()
    report["stages"]["embedding"] = {
        "status": "PASS",
        "seconds": round(time.perf_counter() - started, 3),
        "candidate_count": len(candidate_indices),
    }

    started = time.perf_counter()
    law_evidence = law_retriever.rerank_candidates(
        case.case_query,
        candidate_indices,
        top_k=int(law_config.get("retrieval_k", law_config["top_k"])),
    )
    law_retriever.release_reranker_model()
    report["stages"]["reranker"] = {
        "status": "PASS",
        "seconds": round(time.perf_counter() - started, 3),
        "evidence_count": len(law_evidence),
    }

    started = time.perf_counter()
    predictor = create_predictor(config["prediction"])
    try:
        prediction_pipeline = ALQACPipeline(
            None,
            None,
            predictor,
            allow_prediction_fallback=False,
            max_prediction_retries=int(
                config["prediction"].get("max_case_retries", 0)
            ),
        )
        prediction_result = prediction_pipeline.predict_prepared(
            PreparedCase(case, [], law_evidence, 0)
        )
    finally:
        backend = getattr(predictor, "backend", None)
        if backend is not None and hasattr(backend, "release"):
            backend.release()
    if prediction_result.status != "completed" or not prediction_result.reasoning:
        raise ValueError("Runtime prediction did not return validated reasoning")
    report["stages"]["generation"] = {
        "status": "PASS",
        "seconds": round(time.perf_counter() - started, 3),
        "label": prediction_result.prediction.value,
        "prediction_attempts": prediction_result.prediction_attempts,
        "prediction_failure_types": prediction_result.prediction_failure_types,
    }
    report["status"] = "PASS"
    report["models"] = {
        "query_planner": config["case_retrieval"]["query_planner"]["model_name"],
        "query_planner_revision": config["case_retrieval"]["query_planner"].get(
            "revision"
        ),
        "embedding": law_config.get("embedding_model"),
        "embedding_revision": law_config.get("embedding_revision"),
        "reranker": law_config.get("reranker_model"),
        "reranker_revision": law_config.get("reranker_revision"),
        "outcome": config["prediction"]["model_name"],
        "outcome_revision": config["prediction"].get("revision"),
        "adapter_path": config["prediction"].get("adapter_path"),
    }
    write_json(output, report)
    print(f"Runtime check PASS: {output}")


if __name__ == "__main__":
    main()
