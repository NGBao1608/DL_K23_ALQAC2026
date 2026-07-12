from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path

from .case_retrieval import (
    CaseContentClient,
    CaseEvidenceRetriever,
    SQLiteEvidenceCache,
)
from .config import (
    build_environment,
    build_manifest,
    load_config,
    set_seed,
    source_fingerprint,
    write_json,
)
from .data import load_inference_cases, load_law_corpus, load_public_gold
from .evaluation import build_error_analysis, evaluate_public
from .law_retrieval import create_law_retriever
from .pipeline import ALQACPipeline, CheckpointStore, PreparedCaseStore
from .prediction import OutcomePredictor, create_predictor
from .schemas import InferenceCase
from .submission import build_submission, validate_submission


class EmptyCaseRetriever:
    def retrieve(self, case: InferenceCase):
        return [], 0


class FixedBackend:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return '{"reasoning":"Mock prediction for pipeline validation.","label":"PARTIAL_A_WIN"}'


def run_experiment(
    config_path: str | Path,
    input_path: str | Path,
    output_dir: str | Path | None = None,
    resume_run: str | Path | None = None,
    public_gold_path: str | Path | None = None,
    mock: bool = False,
    limit: int | None = None,
) -> dict:
    config = load_config(config_path)
    set_seed(int(config["run"].get("seed", 42)))
    corpus_path = Path(config["paths"]["corpus"])
    articles = load_law_corpus(corpus_path)
    cases = load_inference_cases(input_path)
    if limit is not None:
        cases = cases[:limit]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(resume_run)
        if resume_run
        else Path(output_dir or config["paths"]["output_dir"])
        / f"{timestamp}_{config['run']['name']}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = run_dir / "config.resolved.json"
    resolved_run = {
        "config": config,
        "execution": {
            "input_path": str(Path(input_path).resolve()),
            "public_gold_path": (
                str(Path(public_gold_path).resolve()) if public_gold_path else None
            ),
            "mock": mock,
            "limit": limit,
            "source_fingerprint": source_fingerprint(Path(__file__).parent),
        },
    }
    if resume_run and resolved_path.exists():
        previous_run = json.loads(resolved_path.read_text(encoding="utf-8"))
        if previous_run != resolved_run:
            raise ValueError("Cannot resume with different config, input, mock mode, or limit")
    write_json(resolved_path, resolved_run)

    manifest = build_manifest(config, corpus_path)
    manifest["run"] = {
        "status": "running",
        "cases": len(cases),
        "mock": mock,
        "completed": 0,
        "api_calls": 0,
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "environment.json", build_environment())
    cache = None
    client = None
    checkpoint = CheckpointStore(run_dir / "predictions.checkpoint.json")
    prepared_store = PreparedCaseStore(run_dir / "contexts.checkpoint.json")
    results_by_id = {}
    try:
        pending_cases = []
        for case in cases:
            existing = checkpoint.get(case.case_id)
            if existing is not None and existing.status == "completed":
                results_by_id[case.case_id] = existing
            else:
                pending_cases.append(case)

        prepared_by_id = {}
        to_prepare = []
        for case in pending_cases:
            prepared = prepared_store.get(case)
            if prepared is None:
                to_prepare.append(case)
            else:
                prepared_by_id[case.case_id] = prepared

        if to_prepare:
            law_config = dict(config["law_retrieval"])
            law_config["index_dir"] = config["paths"]["law_index"]
            if mock:
                law_config["strategy"] = "bm25_only"
            law_retriever = create_law_retriever(articles, law_config)
            if mock:
                case_retriever = EmptyCaseRetriever()
            else:
                token = os.getenv("ALQAC_TEAM_TOKEN", "")
                cache = SQLiteEvidenceCache(config["paths"]["cache_db"])
                client = CaseContentClient(
                    token=token,
                    base_url=config["case_retrieval"]["api_base"],
                    cache=cache,
                    request_interval_seconds=float(
                        config["case_retrieval"]["request_interval_seconds"]
                    ),
                    timeout_seconds=int(config["case_retrieval"]["timeout_seconds"]),
                    retries=int(config["case_retrieval"]["retries"]),
                )
                case_retriever = CaseEvidenceRetriever(
                    client,
                    max_queries=int(config["case_retrieval"]["max_queries"]),
                )
            preparation_pipeline = ALQACPipeline(
                case_retriever, law_retriever, predictor=None
            )
            for case in to_prepare:
                prepared = preparation_pipeline.prepare_case(
                    case, law_top_k=int(config["law_retrieval"]["top_k"])
                )
                prepared_store.put(prepared)
                prepared_by_id[case.case_id] = prepared
            if hasattr(law_retriever, "release_gpu_models"):
                law_retriever.release_gpu_models()

        if pending_cases:
            predictor = (
                OutcomePredictor(FixedBackend())
                if mock
                else create_predictor(config["prediction"])
            )
            prediction_pipeline = ALQACPipeline(None, None, predictor)
            for case in pending_cases:
                result = prediction_pipeline.predict_prepared(prepared_by_id[case.case_id])
                checkpoint.put(result)
                results_by_id[case.case_id] = result

        results = [results_by_id[case.case_id] for case in cases]
        write_json(
            run_dir / "predictions.json",
            [checkpoint.records[case.case_id] for case in cases],
        )
        submission = build_submission(results)
        validation = validate_submission(submission, cases, articles)
        write_json(run_dir / "submission.json", submission)
        write_json(run_dir / "validation.json", validation)

        metrics = None
        if public_gold_path:
            gold = load_public_gold(public_gold_path, articles)
            metrics = evaluate_public(results, gold)
            write_json(run_dir / "metrics.json", metrics)
            write_json(run_dir / "errors.json", build_error_analysis(results, gold))

        manifest["run"].update(
            {
                "status": "completed",
                "completed": sum(result.status == "completed" for result in results),
                "api_calls": sum(result.api_calls for result in results),
            }
        )
        write_json(
            run_dir / "api_stats.json",
            {
                "total_calls": sum(result.api_calls for result in results),
                "per_case": {
                    result.case_id: result.api_calls for result in results
                },
                "current_process_network_calls": (
                    client.network_calls if client is not None else 0
                ),
                "current_process_cache_hits": (
                    client.cache_hits if client is not None else 0
                ),
            },
        )
        write_json(run_dir / "manifest.json", manifest)
        return {
            "run_dir": str(run_dir),
            "validation": validation,
            "metrics": metrics,
        }
    except Exception as error:
        manifest["run"].update(
            {
                "status": "failed",
                "completed": sum(
                    record.get("status") == "completed"
                    for record in checkpoint.records.values()
                ),
                "api_calls": sum(
                    int(record.get("api_calls", 0))
                    for record in checkpoint.records.values()
                ),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        write_json(
            run_dir / "api_stats.json",
            {
                "total_calls": manifest["run"]["api_calls"],
                "per_case": {
                    case_id: int(record.get("api_calls", 0))
                    for case_id, record in checkpoint.records.items()
                },
                "current_process_network_calls": (
                    client.network_calls if client is not None else 0
                ),
                "current_process_cache_hits": (
                    client.cache_hits if client is not None else 0
                ),
            },
        )
        write_json(run_dir / "manifest.json", manifest)
        raise
    finally:
        if cache is not None:
            cache.close()
