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

The current retriever:

- uses two deterministic production queries: `court_decision` and normalized `case_query`;
- calls official `POST /retrieve` with `X-API-Key`;
- enforces a five-second interval;
- permits at most one retry for `429` and `5xx` within a global run cap;
- stops on `403` and `422`;
- caches successful responses in SQLite; and
- de-duplicates results by exact `chunk_id`.

Important retrieval phrases include “chấp nhận yêu cầu khởi kiện”, “không chấp nhận yêu cầu khởi kiện”, “Hội đồng xét xử nhận định”, and “Tuyên xử”.

Every non-mock run requires an explicit `max_network_calls` value. Preflight counts logical queries, cache hits, and cache misses without contacting the official API. A SQLite ledger records safe metadata for every local HTTP attempt, including failed retries and timeouts.

During a non-mock run, `ALQAC_PROGRESS` records safe per-request Case Content API lifecycle events (`request_started`, `request_completed`, `http_error`, `request_exception`, `retry_scheduled`, and `cache_hit`). This distinguishes API waiting/retry time from hybrid retrieval or Qwen inference without exposing query text, response text, headers, or secrets.

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
                 │                     ├→ Vietnamese_Reranker → top 5
Exact citations ───────────────────────┘
```

The candidate uses `AITeamVN/Vietnamese_Embedding`, normalized dot product, Reciprocal Rank Fusion, and `AITeamVN/Vietnamese_Reranker`. It also extracts up to 12 exact `Điều ...` citations associated with named corpus laws from retrieved case evidence and adds them to the reranker pool. Citation extraction never forces an item directly into the final top five. Returned evidence preserves official `law_id` and `aid`.

Status:

- BM25 path: `CPU/mock verified`.
- Hybrid path: `implemented`; full Kaggle verification is pending.

## 4. Outcome prediction

The current predictor uses pinned `Qwen/Qwen3-8B` with NF4 4-bit quantization, FP16 compute, double quantization, deterministic generation, and thinking disabled.

The candidate uses the `decision_first_v2` prompt. It identifies the plaintiff's main claim, prioritizes `Tuyên xử`/`Quyết định` evidence, separates procedural or independent claims, and estimates the accepted proportion before selecting a label. The `>50%` split between the two partial labels remains a Team implementation heuristic, not a confirmed official definition.

Input context contains:

- `case_query`;
- up to two case evidence segments under the current production query policy, with 2,200 characters per segment for the candidate; and
- up to five law articles.

The model must emit:

```json
{
  "reasoning": "Short internal reasoning",
  "label": "A_WIN"
}
```

The parser validates JSON and the official label. One repair attempt is allowed; a second failure creates an explicit failed prediction that cannot be submitted.

Status: `implemented`; not yet supported by a clean recorded `GPU/API verified` run.

## 5. Submission builder

The builder emits only:

- `case_id`;
- `prediction`;
- `case_evidence`; and
- `law_evidence`.

Reasoning, evidence text, API call counts, latency, raw model output, and secrets remain internal.

Status: `CPU/mock verified`.

## 6. Submission validator

The validator checks coverage, duplicate cases, exact labels and fields, strict `law_id`/`aid` types, corpus-valid law pairs, non-empty opaque case identifiers, duplicate evidence, strict JSON serialization, and the 10 MB file limit. It preserves identifiers exactly and never infers `_chunk_` or `_seg_` patterns.

Status for refreshed submissions: `CPU/mock verified`; real refreshed API identifiers and leaderboard acceptance are pending.

## 7. Evaluation

Public evaluation reports:

- outcome accuracy;
- Micro Law Evidence F1;
- law Recall@5;
- format failures; and
- run-local API statistics.

The current Public file has no gold case `chunk_id`, so offline Case Evidence Recall, Penalized Case Recall, and FinalScore are unavailable. Official values must come from the leaderboard.

The local API count is operational metadata only. The official efficiency count is the organizer's cumulative server-side count across all runs.

Status: outcome/law evaluation is `CPU/mock verified`; official scoring is not leaderboard verified.

## 8. Checkpointing and artifacts

Each run records resolved configuration, environment, API preflight, prepared contexts, predictions, API statistics, validation, metrics, errors, and a manifest containing Git/model/corpus identifiers.

Successful API responses and prepared contexts are reused during resume. Run identity includes input, config, mock/limit settings, and source fingerprint. The SQLite ledger binds attempts to that stable run identity, so a restarted process cannot silently reset its total cap. An explicitly increased cap may be supplied when the team approves additional resume attempts. Kaggle notebooks import/export the SQLite cache separately from submission artifacts.

Notebook logic remains thin; reusable behavior belongs in `src/alqac2026`.

## Module map

| Module | Responsibility |
|---|---|
| `schemas.py` | Inference, gold, evidence, label, and result types |
| `data.py` | Load and separate inference/gold data; load law corpus |
| `case_retrieval.py` | Query generation, official API client, throttle, retry, and cache |
| `law_retrieval.py` | BM25, embedding retrieval, RRF, and reranking |
| `prediction.py` | Qwen backend, prompt, parser, and one repair attempt |
| `pipeline.py` | Context preparation and prediction orchestration |
| `evaluation.py` | Public outcome/law metrics and error analysis |
| `submission.py` | Official output builder and local validation |
| `runner.py` | Staged execution, checkpointing, resume, and artifacts |
