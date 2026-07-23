from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .artifacts import (
    backup_directory,
    export_run,
    restore_directory,
    restore_sqlite_cache,
)
from .case_retrieval import SQLiteEvidenceCache, build_api_plan
from .config import git_revision, load_config, sha256_file, write_json
from .data import (
    PRIVATE_INPUT_SHA256,
    load_inference_cases,
    load_law_corpus,
    validate_private_input,
)
from .law_retrieval import law_index_fingerprint
from .runner import run_experiment


STAGES = {"smoke", "full"}
TRACK_CASES = {"public": 50, "private": 60}
PRIVATE_CORPUS_FILENAME = "private_test_60_cases_extracted_corpus.json"
PRIVATE_CORPUS_SHA256 = (
    "9d79379e017ce346cf143a71fa82f5170a755c33a0341048c9baacb28c6119b5"
)


def run_public_stage(
    *,
    stage: str,
    run_id: str,
    repo_root: str | Path,
    drive_root: str | Path,
    local_root: str | Path,
    config_path: str | Path,
    approved_live_api: bool,
    retry_reserve: int = 4,
    adapter_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one source-pinned Public smoke/full stage with local evaluation."""
    if not approved_live_api:
        raise ValueError(
            "Public end-to-end evaluation uses the live Case Content API. "
            "Set APPROVE_PUBLIC_API_CALLS=True only after reviewing the call budget."
        )
    root = Path(repo_root)
    return _run_stage(
        track="public",
        stage=stage,
        run_id=run_id,
        repo_root=root,
        drive_root=Path(drive_root),
        local_root=Path(local_root),
        config_path=Path(config_path),
        input_path=root / "data/raw/ALQAC2026_public_test.json",
        corpus_path=root / "data/raw/corpus_law_pub.json",
        selection_profile=None,
        retry_reserve=retry_reserve,
        adapter_path=adapter_path,
    )


def run_private_stage(
    *,
    stage: str,
    run_id: str,
    public_run_id: str,
    repo_root: str | Path,
    drive_root: str | Path,
    local_root: str | Path,
    config_path: str | Path,
    retry_reserve: int = 4,
    adapter_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one source-pinned Private smoke/full stage and validate submission."""
    drive = Path(drive_root)
    private_input = drive / "inputs/private/ALQAC_private_test.json"
    private_corpus = drive / "inputs/private" / PRIVATE_CORPUS_FILENAME
    validate_private_input(private_input)
    _validate_private_corpus(private_corpus)
    selection_profile = (
        drive
        / "runs/public"
        / public_run_id
        / "full"
        / "selection_profile.json"
    )
    if not selection_profile.is_file():
        raise FileNotFoundError(
            f"Missing Public selection profile: {selection_profile}"
        )
    return _run_stage(
        track="private",
        stage=stage,
        run_id=run_id,
        repo_root=Path(repo_root),
        drive_root=drive,
        local_root=Path(local_root),
        config_path=Path(config_path),
        input_path=private_input,
        corpus_path=private_corpus,
        selection_profile=selection_profile,
        retry_reserve=retry_reserve,
        adapter_path=adapter_path,
    )


def _run_stage(
    *,
    track: str,
    stage: str,
    run_id: str,
    repo_root: Path,
    drive_root: Path,
    local_root: Path,
    config_path: Path,
    input_path: Path,
    corpus_path: Path,
    selection_profile: Path | None,
    retry_reserve: int,
    adapter_path: str | Path | None,
) -> dict[str, Any]:
    _validate_stage_inputs(track, stage, run_id, retry_reserve)
    expected_cases = TRACK_CASES[track]
    cases = load_inference_cases(input_path)
    if len(cases) != expected_cases:
        raise ValueError(
            f"{track} input must contain {expected_cases} cases, found {len(cases)}"
        )
    articles = load_law_corpus(corpus_path)
    workflow_root = drive_root / "runs" / track / run_id
    stage_dir = workflow_root / stage
    source_pin = _validate_source_pin(workflow_root, repo_root)

    local_root.mkdir(parents=True, exist_ok=True)
    local_cache = local_root / "cache/case_api.sqlite"
    cache_backup = drive_root / "cache/case_api.sqlite"
    restore_sqlite_cache(cache_backup, local_cache)

    config = load_config(config_path)
    config["paths"]["corpus"] = str(corpus_path.resolve())
    index_fingerprint = law_index_fingerprint(config["law_retrieval"], articles)
    drive_index = drive_root / "indexes/law" / index_fingerprint
    local_index = local_root / "indexes/law" / index_fingerprint
    restore_directory(drive_index, local_index)
    config["paths"]["cache_db"] = str(local_cache.resolve())
    config["paths"]["law_index"] = str(local_index.resolve())
    if adapter_path:
        config["prediction"]["adapter_path"] = str(Path(adapter_path).resolve())
    resolved_config = workflow_root / "workflow_config.json"
    _write_or_verify_json(resolved_config, config)

    planned_cases = cases[:2] if stage == "smoke" else cases
    cache = SQLiteEvidenceCache(local_cache)
    try:
        cache.integrity_check()
        api_plan = build_api_plan(
            planned_cases,
            cache,
            max_queries=2,
            approved_max_network_calls=None,
        )
    finally:
        cache.close()

    if stage == "smoke":
        network_cap = 4
        _run_model_gate(
            repo_root=repo_root,
            workflow_root=workflow_root,
            config_path=resolved_config,
            input_path=input_path,
            local_index=local_index,
            source_pin=source_pin,
            adapter_path=adapter_path,
        )
        backup_directory(local_index, drive_index)
    else:
        _require_smoke_gate(workflow_root, source_pin)
        network_cap = _load_or_create_full_budget(
            workflow_root=workflow_root,
            api_plan=api_plan,
            retry_reserve=retry_reserve,
        )

    api_plan.update(
        {
            "track": track,
            "stage": stage,
            "approved_max_network_calls": network_cap,
            "corpus_sha256": sha256_file(corpus_path),
            "index_fingerprint": index_fingerprint,
        }
    )
    write_json(workflow_root / f"{stage}_api_plan.json", api_plan)

    result = run_experiment(
        config_path=resolved_config,
        input_path=input_path,
        resume_run=stage_dir,
        public_gold_path=input_path if track == "public" else None,
        limit=2 if stage == "smoke" else None,
        cache_db=local_cache,
        max_network_calls=network_cap,
        execution_mode="live",
        cache_backup_db=cache_backup,
        law_index_dir=local_index,
        selection_profile=selection_profile,
        adapter_path=adapter_path,
    )
    _validate_stage_result(
        stage_dir=stage_dir,
        expected_cases=2 if stage == "smoke" else expected_cases,
        source_pin=source_pin,
    )
    if stage == "smoke":
        write_json(
            workflow_root / "smoke_gate.json",
            {
                "status": "PASS",
                "git_commit": source_pin["commit"],
                "cases": 2,
                "track": track,
            },
        )
    else:
        export_dir = drive_root / "exports" / run_id
        if export_dir.exists():
            _verify_existing_export(stage_dir, export_dir)
        else:
            export_run(stage_dir, export_dir)
        result["export_dir"] = str(export_dir)
    result.update(
        {
            "track": track,
            "stage": stage,
            "api_plan": api_plan,
            "index_fingerprint": index_fingerprint,
        }
    )
    return result


def _run_model_gate(
    *,
    repo_root: Path,
    workflow_root: Path,
    config_path: Path,
    input_path: Path,
    local_index: Path,
    source_pin: dict[str, Any],
    adapter_path: str | Path | None,
) -> None:
    report_path = workflow_root / "runtime_check.json"
    command = [
        sys.executable,
        str(repo_root / "scripts/check_runtime.py"),
        "--config",
        str(config_path),
        "--input",
        str(input_path),
        "--output",
        str(report_path),
        "--law-index-dir",
        str(local_index),
    ]
    if adapter_path:
        command.extend(["--adapter-path", str(adapter_path)])
    subprocess.check_call(command, cwd=repo_root)
    report = _read_json(report_path)
    if report.get("status") != "PASS" or report.get("api_network_attempts") != 0:
        raise RuntimeError("Model-only runtime check did not pass")
    source_pin["runtime_check_status"] = "PASS"
    source_pin["runtime_source_sha256"] = report.get("source_sha256")
    write_json(workflow_root / "source_pin.json", source_pin)


def _require_smoke_gate(
    workflow_root: Path, source_pin: dict[str, Any]
) -> None:
    gate = _read_json(workflow_root / "smoke_gate.json")
    if (
        gate.get("status") != "PASS"
        or gate.get("cases") != 2
        or gate.get("git_commit") != source_pin.get("commit")
    ):
        raise ValueError("Full requires a passing two-case smoke on the pinned commit")
    report = _read_json(workflow_root / "runtime_check.json")
    if report.get("status") != "PASS":
        raise ValueError("Full requires the model-only runtime check from smoke")


def _load_or_create_full_budget(
    *,
    workflow_root: Path,
    api_plan: dict[str, Any],
    retry_reserve: int,
) -> int:
    budget_path = workflow_root / "full_budget.json"
    if budget_path.exists():
        value = _read_json(budget_path).get("approved_max_network_calls")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Invalid saved full budget: {budget_path}")
        return value
    value = int(api_plan["cache_misses"]) + retry_reserve
    write_json(
        budget_path,
        {
            "approved_max_network_calls": value,
            "initial_cache_misses": int(api_plan["cache_misses"]),
            "retry_reserve": retry_reserve,
        },
    )
    return value


def _validate_stage_result(
    *,
    stage_dir: Path,
    expected_cases: int,
    source_pin: dict[str, Any],
) -> None:
    manifest = _read_json(stage_dir / "manifest.json")
    validation = _read_json(stage_dir / "validation.json")
    if (
        manifest.get("run", {}).get("status") != "completed"
        or manifest.get("run", {}).get("completed") != expected_cases
        or validation.get("status") != "PASS"
        or validation.get("cases") != expected_cases
    ):
        raise ValueError(f"Stage did not complete and validate: {stage_dir}")
    if manifest.get("git_commit") != source_pin.get("commit"):
        raise ValueError("Stage Git commit does not match source pin")


def _validate_source_pin(
    workflow_root: Path, repo_root: Path
) -> dict[str, Any]:
    pin = _read_json(workflow_root / "source_pin.json")
    if Path.cwd().resolve() != repo_root.resolve():
        os.chdir(repo_root)
    revision, dirty = git_revision()
    if revision != pin.get("commit"):
        raise ValueError(
            f"Checked-out commit {revision} does not match source pin {pin.get('commit')}"
        )
    if dirty:
        raise ValueError("Colab source checkout must be clean")
    return pin


def _validate_private_corpus(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Private law corpus: {path}")
    if sha256_file(path) != PRIVATE_CORPUS_SHA256:
        raise ValueError("Private law corpus SHA-256 does not match the reviewed file")
    articles = load_law_corpus(path)
    law_ids = {article.law_id for article in articles}
    if len(law_ids) != 14 or len(articles) != 2820:
        raise ValueError(
            "Private law corpus must contain the reviewed 14 laws and 2,820 articles"
        )


def _validate_stage_inputs(
    track: str, stage: str, run_id: str, retry_reserve: int
) -> None:
    if track not in TRACK_CASES:
        raise ValueError("track must be public or private")
    if stage not in STAGES:
        raise ValueError("stage must be smoke or full")
    if not run_id.strip() or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be one non-empty path component")
    if retry_reserve < 0:
        raise ValueError("retry_reserve must be non-negative")


def _write_or_verify_json(path: Path, value: Any) -> None:
    if path.exists():
        current = _read_json(path)
        if current != value:
            raise ValueError(f"Workflow configuration changed for existing RUN_ID: {path}")
        return
    write_json(path, value)


def _verify_existing_export(stage_dir: Path, export_dir: Path) -> None:
    checksums_path = export_dir / "SHA256SUMS"
    if not checksums_path.is_file():
        raise ValueError(f"Existing export has no SHA256SUMS: {export_dir}")
    recorded = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        digest, separator, filename = line.partition("  ")
        if not separator or not filename:
            raise ValueError(f"Invalid SHA256SUMS entry: {line}")
        recorded[filename] = digest
    for filename in ("submission.json", "validation.json", "manifest.json"):
        source = stage_dir / filename
        exported = export_dir / filename
        if not source.is_file() or not exported.is_file():
            raise ValueError(f"Existing export is incomplete: {export_dir}")
        source_digest = _sha256(source)
        exported_digest = _sha256(exported)
        if source_digest != exported_digest or recorded.get(filename) != exported_digest:
            raise ValueError(
                f"Existing export does not match the validated full run: {filename}"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
