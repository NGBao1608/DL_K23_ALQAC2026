# Architecture — Synchronized Legacy View

**Last synchronized:** 2026-07-24

**Canonical source:** [`pipeline.md`](pipeline.md)

This filename is retained for existing links. `pipeline.md` is the canonical implementation document and must be updated first when architecture or status changes.

## System overview

```text
case_id + case_query
        │
        ├── structured planner + fallback
        │       └── 2 primary queries + gate + optional query 3
        ├──────────────────────────────────→ case evidence
        ├── BM25 / dense + RRF + reranker ─→ law evidence
        └── Qwen3-8B 4-bit ────────────────→ outcome label
                                                   │
                                                   ▼
                                    builder → validator → submission.json
```

Public development uses the same private-like production pipeline. Public gold
annotations may enter offline training/evaluation code but never production
inference.

## Data boundary

```text
Public JSON ──→ InferenceCase(case_id, case_query) ──→ pipeline
      └──────→ PublicGold ───────────────────────────→ evaluator

Private JSON ─→ InferenceCase(case_id, case_query) ──→ pipeline
```

`verdict_label`, `case_fact`, `judgment_text`, `court_reasoning`, `court_verdict`, and `related_law_provisions` are prohibited as inference features.

Status: `CPU/mock verified`.

## Case evidence stage

Live production retrieval uses a pinned small LLM structured planner with
deterministic fallback, two primary composed queries, and an optional adaptive
third query after the evidence gate fails. The canonical Public quality workflow
uses live retrieval after explicit approval; low-level zero-network
`cache-only` remains available for diagnostics.

The official call count is permanent across runs. Live preflight, a three-attempt
per-case cap shared by retry/semantic queries, and cache/checkpoint reuse are
therefore part of scoring safety. The global cap is planned cases multiplied by
three. A successful response, ledger row, and pending-backup marker share one
transaction. Every attempt is backed up to external SQLite before the next
request; a new live client restores per-case counts and repairs pending state
before any cache hit or request, and backup failure stops the run.

Returned `chunk_id` values are treated as opaque throughout the pipeline and validator. No `_chunk_` or `_seg_` prefix is inferred.

Status: query policy, cache/retry, preflight, hard cap, ledger, and opaque validation are `CPU/mock verified`; refreshed official API verification is pending.

## Law evidence stage

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

The candidate uses a controlled case/evidence query, `AITeamVN/Vietnamese_Embedding`, and `AITeamVN/Vietnamese_Reranker`. Up to 12 corpus-valid citations expand the reranker pool without bypassing reranking. The checkpoint keeps top 10, Qwen reads top five, and Public evaluation selects submission top-k 3–10.

Status: BM25 is `CPU/mock verified`; the hybrid candidate is `implemented`.

## Outcome stage

The predictor uses pinned `Qwen/Qwen3-8B`, NF4 4-bit, FP16 compute, deterministic generation, and thinking disabled. Token-aware context protects the prompt/query/final instruction under a 6,144-token cap and allocates remaining space to prioritized case evidence and five law articles.

The model returns structured main claim, accepted scope, acceptance ratio,
reasoning, and label. Numeric ratios enforce the official boundary; partial or
inconsistent results receive a verifier pass, and malformed output receives one
repair. An approved PEFT adapter may be loaded through `adapter_path`; the
case-grouped five-fold Public training workflow is not implemented yet.

Status: `implemented`; a clean refreshed `GPU/API verified` run is pending.

The selected Qwen3-8B model is open-weight and below the official 10-billion-parameter limit. Proprietary model APIs and externally annotated legal QA/entailment datasets are outside the allowed architecture boundary.

## Submission and evaluation

The validator checks exact fields and labels, non-empty opaque identifiers, corpus-valid law pairs and strict types, duplicates, strict JSON, and the 10 MB limit. Validation and manifest artifacts bind the exact submission SHA-256 and byte length; export rechecks both bindings and the actual case count. It is `CPU/mock verified`; official refreshed-format acceptance remains pending.

Public evaluation supports Outcome Accuracy, Micro Law Evidence F1, law Recall@5, format failures, and run-local API statistics. Public data does not contain gold case `chunk_id`, so official Penalized Case Recall and FinalScore require the leaderboard.

## Runtime and artifacts

The canonical Colab workflow exposes only `smoke` and `full`. Smoke pins one
exact Git commit/config fingerprint, runs `scripts/check_runtime.py` to load
planner, embedding, reranker, and Qwen before any live request, then predicts
two cases. Full requires that gate on the same commit/config and automatically
resumes its own full checkpoints while reusing shared plans/cache.
`colab_public.ipynb` stores Public evaluation artifacts;
`colab_private.ipynb` uses the Private law corpus and stores submission
artifacts. SQLite/model/index working copies stay local while reusable
artifacts are backed by Drive.

Model-cache restoration copies only revision snapshots, refs, and negative
cache metadata. It intentionally skips Hugging Face `blobs` because Drive
backups already contain materialized snapshot files and copying both would
duplicate model storage locally.

Each run records:

- resolved config and environment;
- prepared-context and prediction checkpoints;
- predictions and errors;
- API preflight and API/cache statistics;
- validation and Public metrics; and
- a manifest with Git, model, and corpus identifiers.

Notebook code remains thin; reusable behavior belongs in `src/alqac2026`.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `schemas.py` | Inference, gold, evidence, label, and result types |
| `data.py` | Data boundary and law-corpus loading |
| `case_retrieval.py` | Queries, official API, throttle, retry, and cache |
| `law_retrieval.py` | BM25, dense retrieval, RRF, and reranking |
| `prediction.py` | Qwen backend, prompt, parser, and repair |
| `pipeline.py` | Context preparation and prediction orchestration |
| `evaluation.py` | Public metrics and error analysis |
| `submission.py` | Submission builder and local validator |
| `runner.py` | Staged execution, resume, and artifacts |
| `artifacts.py` | Drive layout, cache restore, directory sync, and safe export |
| `colab_workflow.py` | Source-pinned smoke/full orchestration for Public and Private |

Implementation priorities are maintained in [`plan.md`](plan.md); team choices and rejected alternatives are maintained in [`decisions.md`](decisions.md).
