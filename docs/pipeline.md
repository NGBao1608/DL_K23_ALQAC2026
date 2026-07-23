# ALQAC 2026 Pipeline

This document describes the repository implementation. It is not an official competition specification; official rules live in `docs/competition.md`.

## Status vocabulary

- `implemented`: code exists but may not have run in the target environment.
- `CPU/mock verified`: covered by local tests or mock execution only.
- `GPU/API verified`: executed successfully with pinned models and the official API.
- `leaderboard verified`: confirmed by an official leaderboard result.
- `Not implemented yet`: required behavior is absent or incompatible.

## End-to-end flow

```text
case_id + case_query
        │
        ├── Case Content API retrieval ──→ case evidence
        ├── BM25 / hybrid law retrieval ─→ law evidence
        └── Qwen3-8B prediction ─────────→ outcome label
                                              │
                                              ▼
                                  submission builder + validator
                                              │
                                              ▼
                                      submission.json
```

Public evaluation creates private-like `InferenceCase` objects before calling the same production pipeline. Gold annotations are loaded only after predictions finish.

## 1. Data loading

`load_inference_cases()` reads only `case_id` and `case_query` into `InferenceCase`. `load_public_gold()` separately reads `verdict_label`, resolved law annotations, and optional case-evidence gold.

Status: `CPU/mock verified`.

Data boundary:

```text
Public JSON ──→ InferenceCase ──→ production pipeline ──→ PredictionResult
      └──────→ PublicGold ─────────────────────────────→ evaluator

Private JSON ─→ InferenceCase ──→ production pipeline ──→ submission
```

The following Public-only fields must never become inference features: `verdict_label`, `case_fact`, `judgment_text`, `court_reasoning`, `court_verdict`, and `related_law_provisions`.

## 2. Case evidence retrieval

Live retrieval:

- uses two deterministic production queries: `court_decision` and normalized `case_query`;
- calls official `POST /retrieve` with `X-API-Key`;
- enforces a five-second interval;
- permits at most one retry for `429` and `5xx` within a global run cap;
- stops on `403` and `422`;
- caches successful responses in SQLite; and
- de-duplicates results by exact `chunk_id`.

Important retrieval phrases include “chấp nhận yêu cầu khởi kiện”, “không chấp nhận yêu cầu khởi kiện”, “Hội đồng xét xử nhận định”, and “Tuyên xử”.

Execution is explicit: `mock` uses no models or API, `cache-only` runs the real model stack but returns empty evidence for cache misses, and `live` enables the official API. Cache-only never reads the team token or instantiates the HTTP client and fixes the network cap at zero. Every live run requires an explicit `max_network_calls` value.

These are low-level runner execution modes, not user-facing Colab stages. The
canonical Colab notebooks expose only `smoke` and `full`; both use live
retrieval, while smoke automatically executes the zero-API model gate first.

Preflight counts logical queries, cache hits, and cache misses without contacting the official API. A successful response, its attempt ledger row, and a pending-backup marker are committed in one SQLite transaction. When `cache_backup_db` is configured, every success, HTTP error, or exception produces a verified atomic backup before another request is attempted. On resume, a new live client repairs pending backup state before returning a cache hit or sending a request; backup failure stops the run.

During retrieval, `ALQAC_PROGRESS` records safe lifecycle events (`request_started`, `request_completed`, `http_error`, `request_exception`, `retry_scheduled`, `cache_hit`, and `cache_miss`) without exposing query text, response text, headers, or secrets.

Status: cache, retry, two-query policy, preflight, budget guard, and ledger are `CPU/mock verified`; a clean run against the refreshed official API is not yet recorded as `GPU/API verified`.

Because API calls accumulate permanently across runs, broad query experimentation is prohibited without a reviewed call budget.

## 3. Law evidence retrieval

Baseline:

```text
case query + case evidence → BM25 → top 5
```

Candidate:

```text
BM25 top 50 ─────┐
Dense top 50 ────┼→ RRF(k=60) top 30 ─┐
                 │                     ├→ Vietnamese_Reranker → top 10
Exact citations ───────────────────────┘
```

The candidate uses `AITeamVN/Vietnamese_Embedding`, normalized dot product, Reciprocal Rank Fusion, and `AITeamVN/Vietnamese_Reranker`. A controlled law query uses the case query, dispute phrase, claim/decision sentences, and exact citations instead of arbitrary evidence prefixes. Up to 12 exact `Điều ...` citations expand the reranker pool without bypassing reranking. The index metadata binds corpus content, preprocessing schema, model revision, article keys, and embedding dimension.

Prepared contexts retain the top 10 laws. Qwen sees the first five. Public evaluation selects submission top-k from 3 through 10 by Micro Law F1, ties toward the smaller k, and writes a profile that Private consumes as a scalar only.

Status:

- BM25 path: `CPU/mock verified`.
- Hybrid path: `implemented`; full Colab verification is pending.

## 4. Outcome prediction

The current predictor uses pinned `Qwen/Qwen3-8B` with NF4 4-bit quantization, FP16 compute, double quantization, deterministic generation, and thinking disabled.

The candidate uses the `decision_first_v2` prompt. It identifies the plaintiff's main claim, prioritizes `Tuyên xử`/`Quyết định` evidence, separates procedural or independent claims, and estimates the accepted proportion before selecting a label. Its `>50%` boundary for `PARTIAL_A_WIN` versus `PARTIAL_B_WIN` now matches the official competition definition.

Input context is tokenizer-aware with a 6,144-token input cap and a 384-token output cap. System instructions, `case_query`, and the final JSON instruction are protected; remaining context is allocated approximately 65% to prioritized case evidence and 35% to the first five laws, with unused space reallocated.

The model must emit:

```json
{
  "main_claim": "Primary relief requested",
  "accepted_scope": "Relief accepted by the court",
  "acceptance_ratio": 0.6,
  "reasoning": "Short internal reasoning",
  "label": "A_WIN"
}
```

The parser validates JSON, ratio range, and the official `>50%` boundary. Numeric ratios deterministically normalize inconsistent labels. Partial or inconsistent results receive one deterministic verifier pass; a malformed first output receives one repair. Unrecoverable output creates a failed result that cannot be submitted. `adapter_path` optionally loads an approved PEFT adapter; no fine-tuning data path is enabled.

Status: `implemented`; not yet supported by a clean recorded `GPU/API verified` run.

The pinned Qwen3-8B, Vietnamese Embedding, and Vietnamese Reranker revisions are public, ungated, Apache-2.0, and each below the official 10-billion-parameter limit according to their Hugging Face metadata checked on 2026-07-20. The production pipeline loads these weights locally, does not call a proprietary model API, and must not use externally annotated legal QA or legal entailment datasets.

## 5. Submission builder

The builder emits only:

- `case_id`;
- `prediction`;
- `case_evidence`; and
- `law_evidence`.

Reasoning, evidence text, API call counts, latency, raw model output, and secrets remain internal.

Status: `CPU/mock verified`.

## 6. Submission validator

The validator checks coverage, duplicate cases, exact labels and fields, strict `law_id`/`aid` types, corpus-valid law pairs, non-empty opaque case identifiers, duplicate evidence, strict JSON serialization, and the 10 MB file limit. It preserves identifiers exactly and never infers `_chunk_` or `_seg_` patterns. The runner records the validated submission SHA-256 and byte length in both validation and manifest; export recomputes them and rejects stale or replaced submissions and count mismatches.

Status for refreshed submissions: `CPU/mock verified`; real refreshed API identifiers and leaderboard acceptance are pending.

## 7. Evaluation

Public evaluation reports:

- outcome accuracy;
- Micro Law Evidence F1;
- law Recall@5;
- format failures; and
- run-local API statistics; and
- the selected submission law top-k and per-k selection profile.

The current Public file has no gold case `chunk_id`, so offline Case Evidence Recall, Penalized Case Recall, and FinalScore are unavailable. Official values must come from the leaderboard.

The local API count is operational metadata only. The official efficiency count is the organizer's cumulative server-side count across all runs.

Status: outcome/law evaluation is `CPU/mock verified`; official scoring is not leaderboard verified.

## 8. Checkpointing and artifacts

Each run records resolved configuration, environment, API preflight, prepared contexts, predictions, API statistics, validation, metrics, errors, law selection, and a manifest containing Git/model/corpus/input identifiers. `scripts/check_runtime.py` creates the zero-API embedding/reranker/Qwen gate before live retrieval. In Colab, smoke persists one exact source commit per `RUN_ID`; full requires a completed two-case smoke gate from that commit and automatically resumes only its own full checkpoint directory.

Successful API responses and prepared contexts are reused during resume. Run
identity includes input, config, execution mode, limit, storage paths,
selection profile, and source fingerprint. The Drive-first Colab notebook
restores SQLite and index artifacts plus materialized model snapshots to local
storage while skipping duplicate Hugging Face blobs. It checkpoints runs under
track-specific Drive directories and exports only allowlisted validated files
plus SHA-256 checksums. Private input and law corpus come from the source-pinned
private repository checkout.

Notebook logic remains thin; reusable behavior belongs in `src/alqac2026`.

## Module map

| Module | Responsibility |
|---|---|
| `schemas.py` | Inference, gold, evidence, label, and result types |
| `data.py` | Load and separate inference/gold data; load law corpus |
| `case_retrieval.py` | Query generation, official API client, throttle, retry, and cache |
| `law_retrieval.py` | BM25, embedding retrieval, RRF, and reranking |
| `prediction.py` | Token-aware Qwen context, structured parser, verifier, repair, and optional adapter |
| `pipeline.py` | Context preparation and prediction orchestration |
| `evaluation.py` | Public outcome/law metrics and error analysis |
| `submission.py` | Official output builder and local validation |
| `runner.py` | Staged execution, checkpointing, resume, and artifacts |
| `artifacts.py` | Drive layout, verified cache restore, directory sync, and safe exports |
| `colab_workflow.py` | Public/Private smoke/full orchestration and Drive contracts |
