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
        ├── LLM structured planner
        │       └── safe deterministic fallback
        │
        ├── two primary Case API queries
        │       └── evidence gate ──→ optional adaptive query 3
        │
        ├───────────────────────────────→ case evidence
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

### Structured planning stage

The planner runs before law retrieval and outcome generation:

1. pinned NF4 4-bit `Qwen/Qwen3-8B` at revision
   `b968826d9c46dd6066d109eabc6255188de91218` attempts a deterministic,
   thinking-disabled structured plan with a 256-token output limit and
   deadline-aware 12-second generation stop;
2. the prompt includes query-derived deterministic lexical hints, and output is
   validated for exact schema, required claim/remedy/object coverage, canonical
   case type, and lexical grounding in `case_query`;
3. load, generation, timeout, JSON, validation, or runtime failure selects the
   deterministic planner without failing the case, while a classified
   `planner_failure_code` is persisted without raw model/query text; and
4. the planner backend is released before law retrieval models load.

The plan contains `case_type`, `main_claim`, `requested_remedies`,
`legal_objects`, and `amounts_or_areas`. The deterministic planner normalizes
Unicode/whitespace and extracts dispute type, requested relief, money/area,
contract and land identifiers, and primary legal objects.

The deterministic composer produces up to three concise de-duplicated queries:
operative verdict plus main claim, remedy/object scope, and an adaptive missing
scope query. It never sends the full normalized `case_query` and never predicts
the winner or acceptance result.

### Retrieval and sufficiency gate

Live retrieval:

- attempts each primary query once, subject to cache and remaining budget,
  before assigning the one case-level retry;
- checks operative markers, court/Hội đồng xét xử source role, claim/object
  overlap, accepted/rejected scope, procedural/party-statement negatives, and
  duplicate `chunk_id`;
- issues the third query only when the gate fails and budget remains;
- calls official `POST /retrieve` with `X-API-Key`;
- enforces a five-second interval;
- permits at most one case-level retry for timeout, `429`, or `5xx`;
- does not retry `400`, `401`, `403`, or `422`;
- records a transient request error as recovered when the query-level retry
  succeeds, without treating that case as degraded;
- caches successful responses in SQLite; and
- de-duplicates results by exact `chunk_id`.

The third network attempt is allocated to a failed primary retry first, or to
adaptive Q3 when both primary requests succeeded and the evidence gate failed.
When the global/per-case budget is exhausted, or a query still has a classified
request failure after its retry policy is exhausted, retrieval records an
unresolved degradation code and continues with cached/partial evidence. A
timeout, `429`, or `5xx` recovered by a successful retry remains visible under
`retrieval_recovered_failure_codes` and `recovered_retrieval_cases`, but does
not increment `degraded_cases`. Token/authentication, malformed
successful-response, SQLite integrity, and external-backup failures remain
fail-fast because they are systemic or safety-critical.

Important retrieval phrases include “chấp nhận yêu cầu khởi kiện”, “không chấp nhận yêu cầu khởi kiện”, “Hội đồng xét xử nhận định”, and “Tuyên xử”.

Execution is explicit: `mock` uses no models or API, `cache-only` runs the real model stack but returns empty evidence for cache misses, and `live` enables the official API. Cache-only never reads the team token or instantiates the HTTP client and fixes the network cap at zero. Every live run requires an explicit `max_network_calls` value.

These are low-level runner execution modes, not user-facing Colab stages. The
canonical Colab notebooks expose only `smoke` and `full`; both use live
retrieval, while smoke automatically executes the zero-API model gate first.

Preflight counts the maximum logical queries plus cache hits/misses without
contacting the official API. The global cap is
`planned_cases × max_network_attempts_per_case`; the default per-case cap is
three, shared by retries and semantic queries. Cache hits spend no attempt.
Run-key/per-case counts are restored from the SQLite ledger when the client is
recreated. A successful response, its attempt ledger row, and a pending-backup
marker are committed in one SQLite transaction. When `cache_backup_db` is
configured, every success, HTTP error, or exception produces a verified atomic
backup before another request is attempted. On resume, a new live client
repairs pending backup state before returning a cache hit or sending a request;
backup failure stops the run.

During retrieval, `ALQAC_PROGRESS` records safe lifecycle events
(`request_started`, `request_completed`, `http_error`, `request_exception`,
`retry_scheduled`, `query_failed`, `query_skipped_budget`, `cache_hit`, and
`cache_miss`) without exposing query text, response text, headers, or secrets.

The query-plan artifact is reused only when its fingerprint matches `case_id`,
raw `case_query` SHA-256, configured planner strategy/model revision, prompt
version, and composer version. It is internal, absent from `ALQAC_PROGRESS`,
submission, and exports.

Status: structured fallback planning, composer, evidence gate, adaptive policy,
cache, budget-aware retry scheduling, fail-soft case retrieval, preflight,
budget guards, resume ledger, and artifact exclusion are `CPU/mock verified`;
the pinned 8B planner and refreshed official API path are not yet
`GPU/API verified`.

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

Prepared contexts retain the top 10 laws. The memory-safe candidate outcome
prompt sees the first three, while its OOM recovery prompt sees the first two;
this does not truncate the law evidence retained for evaluation or submission.
Public evaluation selects submission top-k from 3 through 10 by Micro Law F1,
ties toward the smaller k, and writes a profile that Private consumes as a
scalar only.

Status:

- BM25 path: `CPU/mock verified`.
- Hybrid path: `implemented`; full Colab verification is pending.

## 4. Outcome prediction

The current predictor uses pinned `Qwen/Qwen3-8B` with NF4 4-bit quantization,
FP16 compute, double quantization, deterministic generation, thinking disabled,
SDPA attention, and an offloaded KV cache for the candidate T4 profile.

The candidate uses the `decision_first_v2` prompt. It identifies the plaintiff's main claim, prioritizes `Tuyên xử`/`Quyết định` evidence, separates procedural or independent claims, and estimates the accepted proportion before selecting a label. Its `>50%` boundary for `PARTIAL_A_WIN` versus `PARTIAL_B_WIN` now matches the official competition definition.

Candidate input context is tokenizer-aware with a 4,096-token input cap and a
192-token output cap. System instructions, `case_query`, and the final JSON
instruction are protected; remaining context is allocated approximately 65% to
prioritized case evidence and 35% to the first three laws, with unused space
reallocated. Retrieval still checkpoints ten law results, so this prompt-only
limit does not change submission evidence.

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

The parser validates JSON, ratio range, and the official `>50%` boundary.
Numeric ratios deterministically normalize inconsistent labels. Partial or
inconsistent results receive one deterministic verifier pass; a malformed
first output receives one repair. These operations belong to one case-level
prediction attempt.

If that attempt still raises an exception, the prediction stage performs
garbage collection, clears the CUDA cache, and repeats prediction against the
exact same checkpointed `PreparedCase`. Non-OOM failures may use
`max_case_retries: 3`, for at most four model attempts per case. CUDA OOM is
handled differently: the first OOM activates a 3,072-token, 160-output-token,
two-law compact profile and permits one retry; a second OOM falls back
immediately. Both normal and compact candidate generation use offloaded KV
cache. Re-prediction never repeats query planning, Case API calls, or law
retrieval. Only after the applicable bounded policy is exhausted does the
pipeline use the explicitly configured deterministic fallback.
Operative court language maps clear full acceptance/rejection first;
unquantified partial acceptance maps conservatively to `PARTIAL_B_WIN`; the
configured default is `B_WIN` when no trustworthy operative scope exists.
This produces a complete format-valid result while recording
`PredictionFallback:<error type>` internally. Model-load failure remains
fail-fast. `adapter_path` optionally loads an approved PEFT adapter.

Prediction checkpoints preserve `prediction_attempts` and safe
`prediction_failure_types`. Manifest/validation artifacts report
`prediction_retry_cases`, `recovered_prediction_cases`, and the total
`prediction_attempts`. `ALQAC_PROGRESS` emits `prediction/retry_scheduled` with
only attempt numbers and the exception type. A retry-recovered case is not
degraded; a deterministic prediction fallback remains degraded and blocks
smoke. Repeating a
deterministic prompt may reproduce a persistent malformed output, so bounded
re-prediction primarily protects against transient runtime/generation failure.

Offline fine-tuning may use Public gold to construct labels, accepted/rejected
scope, and reasoning targets. Inputs must remain production-equivalent:
`case_query`, cached Case API evidence, and retrieved law evidence. Evaluation
uses group-preserving stratified five-fold splits by `case_id`; augmentations of
one case remain in one fold. Report out-of-fold accuracy/confusion matrix, lock
hyperparameters, then train the final adapter on all Public cases. A training
notebook is `Not implemented yet`.

A successful deterministic planner fallback is the designed recovery path for
an LLM planner load, timeout, generation, JSON, or grounding failure. It remains
visible through `planner_fallbacks` but does not increment `degraded_cases`.
Smoke rejects unresolved retrieval failures and prediction fallbacks. Full
records `degraded_cases`, `planner_fallbacks`, and `fallback_predictions` in its
manifest and validation artifacts, together with retry/recovery counts, so a
complete submission can be reviewed before manual upload.

Status: fallback completion and the memory-safe/OOM-compact policy are
`CPU/mock verified`; the model path is not yet supported by a clean recorded
`GPU/API verified` run.

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

Each run records resolved configuration, environment, API preflight, internal
query plans, prepared contexts, predictions, API statistics, validation,
metrics, errors, law selection, and a manifest containing Git/model/corpus/input
identifiers. `scripts/check_runtime.py` creates the zero-API
planner/embedding/reranker/Qwen gate before live retrieval. In Colab, smoke
persists one exact source commit and `workflow_config.json` fingerprint per
`RUN_ID`; full requires a completed two-case smoke gate from that exact
commit/config and automatically resumes only its own full checkpoint directory.
Smoke/full differ only by case count, stage directory, case-count-derived
network cap, and gate state. Their shared query-plan store and SQLite cache
reuse overlapping smoke work.

Successful API responses and prepared contexts are reused during resume. Run
identity includes input, config, execution mode, limit, storage paths,
selection profile, and source fingerprint. The Drive-first Colab notebook
restores SQLite and index artifacts plus materialized model snapshots to local
storage while skipping duplicate Hugging Face blobs. It checkpoints runs under
track-specific Drive directories and exports only allowlisted validated files
plus SHA-256 checksums. Private input and law corpus come from the source-pinned
private repository checkout.

Before Private starts, the source Public full directory must contain
`selection_profile.json`, `validation.json`, and `manifest.json`; validation
must be `PASS` and the manifest must record all 50 Public cases completed.
Private consumes only the selected law top-k scalar from that profile.

Notebook logic remains thin; reusable behavior belongs in `src/alqac2026`.

## Module map

| Module | Responsibility |
|---|---|
| `schemas.py` | Inference, gold, evidence, label, and result types |
| `data.py` | Load and separate inference/gold data; load law corpus |
| `query_planning.py` | Structured planners, Transformers backend, composer, validation/store, and evidence gate |
| `case_retrieval.py` | Official API client, adaptive execution, throttle, retry, budget, ledger, and cache |
| `law_retrieval.py` | BM25, embedding retrieval, RRF, and reranking |
| `prediction.py` | Token-aware Qwen context, structured parser, verifier, repair, and optional adapter |
| `pipeline.py` | Context preparation and prediction orchestration |
| `evaluation.py` | Public outcome/law metrics and error analysis |
| `submission.py` | Official output builder and local validation |
| `runner.py` | Staged execution, checkpointing, resume, and artifacts |
| `artifacts.py` | Drive layout, verified cache restore, directory sync, and safe exports |
| `colab_workflow.py` | Public/Private smoke/full orchestration and Drive contracts |
