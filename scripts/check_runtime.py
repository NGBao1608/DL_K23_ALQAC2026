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
from alqac2026.prediction import create_predictor


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
        label, reasoning, _ = predictor.predict(case, [], law_evidence)
    finally:
        backend = getattr(predictor, "backend", None)
        if backend is not None and hasattr(backend, "release"):
            backend.release()
    if not reasoning:
        raise ValueError("Runtime prediction did not return validated reasoning")
    report["stages"]["generation"] = {
        "status": "PASS",
        "seconds": round(time.perf_counter() - started, 3),
        "label": label.value,
    }
    report["status"] = "PASS"
    report["models"] = {
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
