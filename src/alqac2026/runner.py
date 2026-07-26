from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

from .case_retrieval import (
    CachedCaseContentClient,
    CaseContentClient,
    CaseEvidenceRetriever,
    SQLiteEvidenceCache,
    build_api_plan,
)
from .config import (
    build_environment,
    build_manifest,
    config_fingerprint,
    load_config,
    set_seed,
    sha256_file,
    source_fingerprint,
    write_json,
)
from .data import load_inference_cases, load_law_corpus, load_public_gold
from .evaluation import build_error_analysis, evaluate_public, select_law_top_k
from .law_retrieval import create_law_retriever, law_index_fingerprint
from .pipeline import ALQACPipeline, CheckpointStore, PreparedCaseStore
from .prediction import OutcomePredictor, create_predictor
from .query_planning import (
    DeterministicQueryComposer,
    QueryPlanStore,
    create_query_planner,
)
from .schemas import InferenceCase, OutcomeLabel
from .submission import build_submission, validate_submission


PROGRESS_PREFIX = "ALQAC_PROGRESS "
EXECUTION_MODES = {"mock", "cache-only", "live"}


def _emit_progress(
    *,
    stage: str,
    status: str,
    case: InferenceCase | None = None,
    index: int | None = None,
    total: int | None = None,
    **details,
) -> None:
    """Emit one safe, machine-readable progress event to stdout."""
    payload = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": stage,
        "status": status,
    }
    if case is not None:
        payload["case_id"] = case.case_id
    if index is not None:
        payload["index"] = index
    if total is not None:
        payload["total"] = total
    payload.update(details)
    print(
        PROGRESS_PREFIX
        + json.dumps(payload, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


class EmptyCaseRetriever:
    def retrieve(self, case: InferenceCase):
        return [], 0


class FixedBackend:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return (
            '{"main_claim":"mock","accepted_scope":"mock partial",'
            '"acceptance_ratio":0.6,'
            '"reasoning":"Mock prediction for pipeline validation.",'
            '"label":"PARTIAL_A_WIN"}'
        )


def run_experiment(
    config_path: str | Path,
    input_path: str | Path,
    output_dir: str | Path | None = None,
    resume_run: str | Path | None = None,
    public_gold_path: str | Path | None = None,
    mock: bool = False,
    limit: int | None = None,
    cache_db: str | Path | None = None,
    max_network_calls: int | None = None,
    execution_mode: str | None = None,
    cache_backup_db: str | Path | None = None,
    law_index_dir: str | Path | None = None,
    selection_profile: str | Path | None = None,
    adapter_path: str | Path | None = None,
    query_plan_store_path: str | Path | None = None,
    corpus_path: str | Path | None = None,
) -> dict:
    config = load_config(config_path)
    mode = execution_mode or ("mock" if mock else "live")
    if mode not in EXECUTION_MODES:
        raise ValueError(f"Unsupported execution_mode: {mode}")
    if mock and mode != "mock":
        raise ValueError("--mock cannot be combined with a non-mock execution mode")
    mock = mode == "mock"
    if cache_db is not None:
        config["paths"]["cache_db"] = str(Path(cache_db))
    if corpus_path is not None:
        config["paths"]["corpus"] = str(Path(corpus_path))
    if law_index_dir is not None:
        config["paths"]["law_index"] = str(Path(law_index_dir))
    if adapter_path is not None:
        config["prediction"]["adapter_path"] = str(Path(adapter_path))
    if mode == "live" and max_network_calls is None:
        raise ValueError(
            "Live runs require an explicit max_network_calls budget"
        )
    if mode == "live" and cache_backup_db is None:
        raise ValueError("Live runs require an external cache_backup_db")
    if (
        mode == "live"
        and Path(cache_backup_db).resolve()
        == Path(config["paths"]["cache_db"]).resolve()
    ):
        raise ValueError("Live cache and external backup paths must differ")
    if max_network_calls is not None and max_network_calls < 0:
        raise ValueError("max_network_calls must be non-negative")
    if mode == "cache-only" and max_network_calls not in {None, 0}:
        raise ValueError("cache-only runs require max_network_calls=0")
    if mode == "cache-only":
        max_network_calls = 0
    selected_law_top_k = _load_selection_top_k(selection_profile)
    set_seed(int(config["run"].get("seed", 42)))
    corpus_path = Path(config["paths"]["corpus"])
    articles = load_law_corpus(corpus_path)
    cases = load_inference_cases(input_path)
    if limit is not None:
        cases = cases[:limit]
    retrieval_config = config["case_retrieval"]
    max_attempts_per_case = int(
        retrieval_config["max_network_attempts_per_case"]
    )
    derived_network_cap = len(cases) * max_attempts_per_case
    if (
        mode == "live"
        and max_network_calls is not None
        and max_network_calls > derived_network_cap
    ):
        raise ValueError(
            "Live network cap exceeds planned_cases × "
            f"max_network_attempts_per_case: {max_network_calls} > "
            f"{derived_network_cap}"
        )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(resume_run)
        if resume_run
        else Path(output_dir or config["paths"]["output_dir"])
        / f"{timestamp}_{config['run']['name']}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = run_dir / "config.resolved.json"
    current_source_fingerprint = source_fingerprint(Path(__file__).parent)
    current_config_fingerprint = config_fingerprint(config)
    query_plan_path = (
        Path(query_plan_store_path)
        if query_plan_store_path is not None
        else run_dir / "query_plans.checkpoint.json"
    )
    resolved_run = {
        "config": config,
        "execution": {
            "input_path": str(Path(input_path).resolve()),
            "public_gold_path": (
                str(Path(public_gold_path).resolve()) if public_gold_path else None
            ),
            "mock": mock,
            "execution_mode": mode,
            "limit": limit,
            "cache_backup_db": (
                str(Path(cache_backup_db).resolve()) if cache_backup_db else None
            ),
            "selection_profile": (
                str(Path(selection_profile).resolve()) if selection_profile else None
            ),
            "submission_law_top_k": selected_law_top_k,
            "source_fingerprint": current_source_fingerprint,
            "config_fingerprint": current_config_fingerprint,
            "query_plan_store_path": str(query_plan_path.resolve()),
        },
    }
    if resume_run and resolved_path.exists():
        previous_run = json.loads(resolved_path.read_text(encoding="utf-8"))
        if previous_run != resolved_run:
            raise ValueError("Cannot resume with different config, input, mock mode, or limit")
    write_json(resolved_path, resolved_run)
    run_key = hashlib.sha256(
        json.dumps(resolved_run, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    manifest = build_manifest(config, corpus_path)
    manifest["source_sha256"] = current_source_fingerprint
    manifest["config_sha256"] = current_config_fingerprint
    manifest["law_index"] = {
        "fingerprint": law_index_fingerprint(config["law_retrieval"], articles),
        "path": str(Path(config["paths"]["law_index"]).resolve()),
    }
    manifest["run"] = {
        "status": "running",
        "cases": len(cases),
        "mock": mock,
        "execution_mode": mode,
        "completed": 0,
        "api_calls": 0,
    }
    input_source = Path(input_path)
    if input_source.is_file():
        manifest["input_sha256"] = sha256_file(input_source)
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "environment.json", build_environment())
    case_positions = {case.case_id: index for index, case in enumerate(cases, start=1)}
    cases_by_id = {case.case_id: case for case in cases}

    def emit_case_api_progress(event: dict[str, object]) -> None:
        payload = dict(event)
        case_id = str(payload.pop("case_id"))
        _emit_progress(
            stage="case_api",
            status=str(payload.pop("status")),
            case=cases_by_id[case_id],
            index=case_positions[case_id],
            total=len(cases),
            **payload,
        )

    def emit_prediction_retry(event: dict[str, object]) -> None:
        payload = dict(event)
        case_id = str(payload.pop("case_id"))
        _emit_progress(
            stage="prediction",
            status="retry_scheduled",
            case=cases_by_id[case_id],
            index=case_positions[case_id],
            total=len(cases),
            **payload,
        )

    _emit_progress(
        stage="run",
        status="started",
        total=len(cases),
        run_name=config["run"]["name"],
        mock=mock,
    )
    cache = None
    client = None
    planner_fallback_case_ids: set[str] = set()
    checkpoint = CheckpointStore(run_dir / "predictions.checkpoint.json")
    prepared_store = PreparedCaseStore(run_dir / "contexts.checkpoint.json")
    case_status_path = run_dir / "case_status.json"
    case_status = (
        json.loads(case_status_path.read_text(encoding="utf-8"))
        if case_status_path.exists()
        else {}
    )
    results_by_id = {}
    try:
        pending_cases = []
        for case in cases:
            existing = checkpoint.get(case.case_id)
            if existing is not None and existing.status == "completed":
                results_by_id[case.case_id] = existing
                _emit_progress(
                    stage="prediction",
                    status="resumed",
                    case=case,
                    index=case_positions[case.case_id],
                    total=len(cases),
                    prediction=(
                        existing.prediction.value if existing.prediction else None
                    ),
                    case_evidence_count=len(existing.case_evidence),
                    law_evidence_count=len(existing.law_evidence),
                    api_calls=existing.api_calls,
                    latency_seconds=round(existing.latency_seconds, 3),
                )
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
                _emit_progress(
                    stage="preparation",
                    status="resumed",
                    case=case,
                    index=case_positions[case.case_id],
                    total=len(cases),
                    case_evidence_count=len(prepared.case_evidence),
                    law_evidence_count=len(prepared.law_evidence),
                    api_calls=prepared.api_calls,
                )

        if not mock:
            cache = SQLiteEvidenceCache(config["paths"]["cache_db"])
            cache.integrity_check()

        if to_prepare:
            query_plans = {}
            if not mock:
                planner_config = retrieval_config["query_planner"]
                planner = create_query_planner(planner_config)
                composer = DeterministicQueryComposer()
                plan_store = QueryPlanStore(query_plan_path)
                try:
                    for case in to_prepare:
                        stored_plan = plan_store.get_or_create(
                            case,
                            planner=planner,
                            composer=composer,
                            configured_strategy=str(planner_config["strategy"]),
                            model_revision=str(planner_config["revision"]),
                            prompt_version=str(planner_config["prompt_version"]),
                            composer_version=str(
                                retrieval_config["composer_version"]
                            ),
                        )
                        query_plans[case.case_id] = stored_plan
                        if (
                            planner_config["strategy"] == "llm_assisted"
                            and stored_plan.planner_strategy != "llm"
                        ):
                            planner_fallback_case_ids.add(case.case_id)
                        case_status.setdefault(case.case_id, {}).update(
                            {
                                "planner_strategy": stored_plan.planner_strategy,
                                "planner_failure_code": (
                                    stored_plan.planner_failure_code
                                ),
                            }
                        )
                        write_json(case_status_path, case_status)
                        _emit_progress(
                            stage="query_planning",
                            status="completed",
                            case=case,
                            index=case_positions[case.case_id],
                            total=len(cases),
                            planner_strategy=stored_plan.planner_strategy,
                            planner_failure_type=stored_plan.planner_failure_type,
                            planner_failure_code=stored_plan.planner_failure_code,
                            query_count=len(stored_plan.queries),
                        )
                finally:
                    planner.release()
            law_config = dict(config["law_retrieval"])
            law_config["index_dir"] = config["paths"]["law_index"]
            if mock:
                law_config["strategy"] = "bm25_only"
            law_retriever = create_law_retriever(articles, law_config)
            if mock:
                case_retriever = EmptyCaseRetriever()
            else:
                api_plan = build_api_plan(
                    to_prepare,
                    cache,
                    query_plans=query_plans,
                    max_logical_queries_per_case=int(
                        retrieval_config["max_logical_queries_per_case"]
                    ),
                    approved_max_network_calls=max_network_calls,
                )
                run_attempts_before = cache.run_attempt_stats(run_key)[
                    "network_attempts"
                ]
                api_plan["run_key"] = run_key
                api_plan["execution_mode"] = mode
                api_plan["run_network_attempts_before"] = run_attempts_before
                api_plan["remaining_approved_attempts"] = max(
                    0, int(max_network_calls or 0) - run_attempts_before
                )
                api_plan["derived_network_cap"] = derived_network_cap
                write_json(run_dir / "api_plan.json", api_plan)
                if mode == "cache-only":
                    client = CachedCaseContentClient(
                        cache=cache,
                        progress_callback=emit_case_api_progress,
                    )
                else:
                    if run_attempts_before > int(max_network_calls):
                        raise ValueError(
                            "Approved API budget is smaller than attempts already "
                            f"recorded for this run: {max_network_calls} < "
                            f"{run_attempts_before}"
                        )
                    from dotenv import load_dotenv

                    load_dotenv()
                    token = os.getenv("ALQAC_TEAM_TOKEN", "")
                    client = CaseContentClient(
                        token=token,
                        base_url=retrieval_config["api_base"],
                        cache=cache,
                        request_interval_seconds=float(
                            retrieval_config["request_interval_seconds"]
                        ),
                        timeout_seconds=int(
                            retrieval_config["timeout_seconds"]
                        ),
                        retries=1 + int(retrieval_config["max_retries"]),
                        max_network_calls=max_network_calls,
                        max_network_attempts_per_case=max_attempts_per_case,
                        run_key=run_key,
                        progress_callback=emit_case_api_progress,
                        backup_path=cache_backup_db,
                    )
                case_retriever = CaseEvidenceRetriever(
                    client,
                    query_plans=query_plans,
                    primary_queries=int(retrieval_config["primary_queries"]),
                    max_logical_queries_per_case=int(
                        retrieval_config["max_logical_queries_per_case"]
                    ),
                )
            preparation_pipeline = ALQACPipeline(
                case_retriever, law_retriever, predictor=None
            )
            for case in to_prepare:
                position = case_positions[case.case_id]
                _emit_progress(
                    stage="preparation",
                    status="started",
                    case=case,
                    index=position,
                    total=len(cases),
                )
                preparation_started = time.perf_counter()
                try:
                    prepared = preparation_pipeline.prepare_case(
                        case,
                        law_top_k=int(
                            config["law_retrieval"].get(
                                "retrieval_k", config["law_retrieval"]["top_k"]
                            )
                        ),
                    )
                except Exception as error:
                    _emit_progress(
                        stage="preparation",
                        status="failed",
                        case=case,
                        index=position,
                        total=len(cases),
                        error_type=type(error).__name__,
                        latency_seconds=round(
                            time.perf_counter() - preparation_started, 3
                        ),
                    )
                    raise
                prepared_store.put(prepared)
                prepared_by_id[case.case_id] = prepared
                case_status.setdefault(case.case_id, {})[
                    "retrieval_failure_codes"
                ] = (
                    list(client.per_case_failure_codes.get(case.case_id, []))
                    if client
                    else []
                )
                case_status.setdefault(case.case_id, {})[
                    "retrieval_recovered_failure_codes"
                ] = (
                    list(
                        client.per_case_recovered_failure_codes.get(
                            case.case_id, []
                        )
                    )
                    if client
                    else []
                )
                write_json(case_status_path, case_status)
                _emit_progress(
                    stage="preparation",
                    status="completed",
                    case=case,
                    index=position,
                    total=len(cases),
                    case_evidence_count=len(prepared.case_evidence),
                    law_evidence_count=len(prepared.law_evidence),
                    api_calls=prepared.api_calls,
                    latency_seconds=round(
                        time.perf_counter() - preparation_started, 3
                    ),
                )
            if hasattr(law_retriever, "release_gpu_models"):
                law_retriever.release_gpu_models()

        if pending_cases:
            predictor = (
                OutcomePredictor(FixedBackend())
                if mock
                else create_predictor(config["prediction"])
            )
            prediction_pipeline = ALQACPipeline(
                None,
                None,
                predictor,
                allow_prediction_fallback=bool(
                    config["prediction"].get("allow_case_fallback", True)
                ),
                prediction_fallback_label=OutcomeLabel(
                    config["prediction"].get("fallback_label", "B_WIN")
                ),
                max_prediction_retries=int(
                    config["prediction"].get("max_case_retries", 0)
                ),
                max_oom_retries=int(
                    config["prediction"].get("max_oom_retries", 1)
                ),
                prediction_retry_callback=emit_prediction_retry,
            )
            for case in pending_cases:
                position = case_positions[case.case_id]
                _emit_progress(
                    stage="prediction",
                    status="started",
                    case=case,
                    index=position,
                    total=len(cases),
                )
                result = prediction_pipeline.predict_prepared(prepared_by_id[case.case_id])
                checkpoint.put(result)
                results_by_id[case.case_id] = result
                case_status.setdefault(case.case_id, {})[
                    "prediction_fallback"
                ] = bool(
                    result.error
                    and result.error.startswith("PredictionFallback:")
                )
                case_status.setdefault(case.case_id, {}).update(
                    {
                        "prediction_attempts": result.prediction_attempts,
                        "prediction_failure_types": result.prediction_failure_types,
                        "output_repair_used": result.output_repair_used,
                        "output_verification": result.output_verification,
                        "prediction_recovered": (
                            result.prediction_attempts > 1
                            and result.status == "completed"
                            and not result.error
                        ),
                    }
                )
                write_json(case_status_path, case_status)
                progress_details = {
                    "prediction": (
                        result.prediction.value if result.prediction else None
                    ),
                    "case_evidence_count": len(result.case_evidence),
                    "law_evidence_count": len(result.law_evidence),
                    "api_calls": result.api_calls,
                    "latency_seconds": round(result.latency_seconds, 3),
                    "prediction_attempts": result.prediction_attempts,
                    "output_repair_used": result.output_repair_used,
                    "output_verification": result.output_verification,
                    "prediction_recovered": (
                        result.prediction_attempts > 1
                        and result.status == "completed"
                        and not result.error
                    ),
                }
                if result.error:
                    progress_details["error_type"] = result.error.split(":", 1)[0]
                _emit_progress(
                    stage="prediction",
                    status=result.status,
                    case=case,
                    index=position,
                    total=len(cases),
                    **progress_details,
                )
            backend = getattr(predictor, "backend", None)
            if backend is not None and hasattr(backend, "release"):
                backend.release()

        results = [results_by_id[case.case_id] for case in cases]
        write_json(
            run_dir / "predictions.json",
            [checkpoint.records[case.case_id] for case in cases],
        )
        metrics = None
        if public_gold_path:
            gold = load_public_gold(public_gold_path, articles)
            if selected_law_top_k is None:
                generated_profile = select_law_top_k(results, gold)
                selected_law_top_k = int(
                    generated_profile["submission_law_top_k"]
                )
                write_json(run_dir / "selection_profile.json", generated_profile)
            metrics = evaluate_public(
                results, gold, law_top_k=selected_law_top_k
            )
            metrics["submission_law_top_k"] = selected_law_top_k
            write_json(run_dir / "metrics.json", metrics)
            write_json(run_dir / "errors.json", build_error_analysis(results, gold))
        elif selection_profile:
            write_json(
                run_dir / "selection_profile.json",
                {
                    "schema_version": "law-top-k-private-v1",
                    "submission_law_top_k": selected_law_top_k,
                    "source_profile_sha256": sha256_file(selection_profile),
                },
            )

        if selected_law_top_k is None:
            selected_law_top_k = int(
                config["law_retrieval"].get(
                    "submission_top_k", config["law_retrieval"]["top_k"]
                )
            )
        submission = build_submission(results, law_top_k=selected_law_top_k)
        validation = validate_submission(submission, cases, articles)
        submission_path = run_dir / "submission.json"
        write_json(submission_path, submission)
        submission_sha256 = sha256_file(submission_path)
        submission_bytes = submission_path.stat().st_size
        validation.update(
            {
                "submission_sha256": submission_sha256,
                "submission_bytes": submission_bytes,
            }
        )
        write_json(run_dir / "validation.json", validation)
        manifest["submission"] = {
            "sha256": submission_sha256,
            "bytes": submission_bytes,
            "cases": len(submission),
        }

        run_network_attempts = (
            cache.run_attempt_stats(run_key)["network_attempts"]
            if cache is not None
            else 0
        )
        if not mock and query_plan_path.exists():
            stored_query_plans = json.loads(
                query_plan_path.read_text(encoding="utf-8")
            )
            planner_fallback_case_ids.update(
                case.case_id
                for case in cases
                if stored_query_plans.get(case.case_id, {}).get(
                    "planner_strategy"
                )
                not in {None, "llm"}
            )
        fallback_case_ids = {
            result.case_id
            for result in results
            if result.error and result.error.startswith("PredictionFallback:")
        }
        prediction_retry_case_ids = {
            result.case_id for result in results if result.prediction_attempts > 1
        }
        prediction_recovered_case_ids = {
            result.case_id
            for result in results
            if result.prediction_attempts > 1
            and result.status == "completed"
            and not result.error
        }
        total_prediction_attempts = sum(
            result.prediction_attempts for result in results
        )
        output_repair_case_ids = {
            result.case_id for result in results if result.output_repair_used
        }
        verification_failed_case_ids = {
            result.case_id
            for result in results
            if result.output_verification == "failed"
        }
        verification_passed_case_ids = {
            result.case_id
            for result in results
            if result.output_verification == "passed"
        }
        retrieval_degraded_case_ids = {
            case.case_id
            for case in cases
            if case_status.get(case.case_id, {}).get(
                "retrieval_failure_codes"
            )
        }
        retrieval_recovered_case_ids = {
            case.case_id
            for case in cases
            if case_status.get(case.case_id, {}).get(
                "retrieval_recovered_failure_codes"
            )
        }
        # Planner fallback is the designed recovery path for a case-scoped
        # planner timeout/load/generation/validation failure. Keep it visible,
        # but reserve degradation for unresolved retrieval or prediction.
        degraded_case_ids = (
            fallback_case_ids
            | retrieval_degraded_case_ids
            | verification_failed_case_ids
        )
        validation.update(
            {
                "degraded_cases": len(degraded_case_ids),
                "fallback_predictions": len(fallback_case_ids),
                "planner_fallbacks": len(planner_fallback_case_ids),
                "recovered_retrieval_cases": len(retrieval_recovered_case_ids),
                "prediction_retry_cases": len(prediction_retry_case_ids),
                "recovered_prediction_cases": len(prediction_recovered_case_ids),
                "prediction_attempts": total_prediction_attempts,
                "output_repair_cases": len(output_repair_case_ids),
                "output_verification_passed_cases": len(
                    verification_passed_case_ids
                ),
                "output_verification_failed_cases": len(
                    verification_failed_case_ids
                ),
            }
        )
        write_json(run_dir / "validation.json", validation)
        manifest["run"].update(
            {
                "status": "completed",
                "completed": sum(result.status == "completed" for result in results),
                "api_calls": run_network_attempts,
                "submission_law_top_k": selected_law_top_k,
                "fallback_predictions": len(fallback_case_ids),
                "planner_fallbacks": len(planner_fallback_case_ids),
                "degraded_cases": len(degraded_case_ids),
                "recovered_retrieval_cases": len(retrieval_recovered_case_ids),
                "prediction_retry_cases": len(prediction_retry_case_ids),
                "recovered_prediction_cases": len(prediction_recovered_case_ids),
                "prediction_attempts": total_prediction_attempts,
                "output_repair_cases": len(output_repair_case_ids),
                "output_verification_passed_cases": len(
                    verification_passed_case_ids
                ),
                "output_verification_failed_cases": len(
                    verification_failed_case_ids
                ),
            }
        )
        write_json(
            run_dir / "api_stats.json",
            _api_stats(
                cases,
                client,
                cache,
                run_key,
                max_network_calls,
                case_status,
            ),
        )
        write_json(run_dir / "manifest.json", manifest)
        _emit_progress(
            stage="run",
            status="completed",
            total=len(cases),
            completed=manifest["run"]["completed"],
            validation_status=validation["status"],
            network_attempts=run_network_attempts,
        )
        return {
            "run_dir": str(run_dir),
            "validation": validation,
            "metrics": metrics,
        }
    except Exception as error:
        run_network_attempts = (
            cache.run_attempt_stats(run_key)["network_attempts"]
            if cache is not None
            else 0
        )
        manifest["run"].update(
            {
                "status": "failed",
                "completed": sum(
                    record.get("status") == "completed"
                    for record in checkpoint.records.values()
                ),
                "api_calls": run_network_attempts,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        write_json(
            run_dir / "api_stats.json",
            _api_stats(
                cases,
                client,
                cache,
                run_key,
                max_network_calls,
                case_status,
            ),
        )
        write_json(run_dir / "manifest.json", manifest)
        _emit_progress(
            stage="run",
            status="failed",
            total=len(cases),
            completed=manifest["run"]["completed"],
            error_type=type(error).__name__,
            network_attempts=run_network_attempts,
        )
        raise
    finally:
        if cache is not None:
            cache.close()


def _load_selection_top_k(path: str | Path | None) -> int | None:
    if path is None:
        return None
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("selection_profile must be a JSON object")
    value = payload.get("submission_law_top_k")
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10:
        raise ValueError("selection_profile submission_law_top_k must be 1..10")
    return value


def _api_stats(
    cases,
    client,
    cache,
    run_key,
    max_network_calls,
    case_status=None,
) -> dict:
    case_status = case_status or {}
    ledger_stats = (
        cache.run_attempt_stats(run_key)
        if cache is not None
        else {"network_attempts": 0, "successful_calls": 0, "per_case": {}}
    )
    per_case = {}
    for case in cases:
        ledger_case = ledger_stats["per_case"].get(case.case_id, {})
        per_case[case.case_id] = {
            "network_attempts": int(ledger_case.get("network_attempts", 0)),
            "successful_calls": int(ledger_case.get("successful_calls", 0)),
            "cache_hits": (
                int(client.per_case_cache_hits.get(case.case_id, 0)) if client else 0
            ),
            "retrieval_failure_codes": (
                list(
                    case_status.get(case.case_id, {}).get(
                        "retrieval_failure_codes",
                        client.per_case_failure_codes.get(case.case_id, [])
                        if client
                        else [],
                    )
                )
            ),
            "retrieval_recovered_failure_codes": (
                list(
                    case_status.get(case.case_id, {}).get(
                        "retrieval_recovered_failure_codes",
                        client.per_case_recovered_failure_codes.get(
                            case.case_id, []
                        )
                        if client
                        else [],
                    )
                )
            ),
        }
    return {
        "run_key": run_key,
        "approved_max_network_calls": max_network_calls,
        "run_network_attempts": int(ledger_stats["network_attempts"]),
        "run_successful_calls": int(ledger_stats["successful_calls"]),
        "run_cache_hits": client.cache_hits if client else 0,
        "per_case": per_case,
        "known_local_cumulative_attempts": (
            cache.known_cumulative_attempts() if cache is not None else 0
        ),
        "official_cumulative_calls": None,
    }
