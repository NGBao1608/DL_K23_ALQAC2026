from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import sha256_file, write_json
from .data import load_inference_cases
from .pipeline import PreparedCaseStore
from .runner import run_experiment


def rescore_prepared_cases(
    *,
    config_path: str | Path,
    input_path: str | Path,
    prepared_contexts_path: str | Path,
    run_dir: str | Path,
    public_gold_path: str | Path | None = None,
    selection_profile: str | Path | None = None,
    adapter_path: str | Path | None = None,
    corpus_path: str | Path | None = None,
    limit: int | None = None,
) -> dict:
    """Re-run outcome prediction from immutable prepared contexts with zero HTTP."""
    source = Path(prepared_contexts_path)
    target_run = Path(run_dir)
    if not source.is_file():
        raise FileNotFoundError(f"Prepared contexts do not exist: {source}")
    if target_run.exists() and any(target_run.iterdir()):
        raise FileExistsError(f"Rescore run directory must be new or empty: {target_run}")

    cases = load_inference_cases(input_path)
    if limit is not None:
        cases = cases[:limit]
    store = PreparedCaseStore(source)
    missing = []
    for case in cases:
        if store.get(case) is None:
            missing.append(case.case_id)
    if missing:
        raise ValueError(
            "Prepared contexts are incomplete for the selected input: "
            + ", ".join(missing)
        )

    target_run.mkdir(parents=True, exist_ok=True)
    target_contexts = target_run / "contexts.checkpoint.json"
    source_sha256 = sha256_file(source)
    source_bytes = source.stat().st_size
    shutil.copy2(source, target_contexts)
    if (
        sha256_file(target_contexts) != source_sha256
        or target_contexts.stat().st_size != source_bytes
    ):
        raise ValueError("Copied prepared contexts failed SHA-256/byte verification")
    local_empty_cache = target_run / "cache-only.sqlite"
    result = run_experiment(
        config_path=config_path,
        input_path=input_path,
        resume_run=target_run,
        public_gold_path=public_gold_path,
        limit=limit,
        cache_db=local_empty_cache,
        max_network_calls=0,
        execution_mode="cache-only",
        selection_profile=selection_profile,
        adapter_path=adapter_path,
        corpus_path=corpus_path,
    )
    network_contract = {
        "execution_mode": "cache-only",
        "max_network_calls": 0,
        "prepared_contexts": str(source.resolve()),
        "prepared_contexts_sha256": source_sha256,
        "prepared_contexts_bytes": source_bytes,
        "api_token_required": False,
    }
    api_stats_path = target_run / "api_stats.json"
    api_stats = json.loads(api_stats_path.read_text(encoding="utf-8"))
    if int(api_stats.get("run_network_attempts", -1)) != 0:
        raise RuntimeError("Prepared-context rescore violated the zero-network contract")
    for filename in ("manifest.json", "validation.json"):
        artifact_path = target_run / filename
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["prepared_rescore"] = network_contract
        write_json(artifact_path, artifact)
    result["network_contract"] = network_contract
    return result
