# Architecture — Synchronized Legacy View

**Last synchronized:** 2026-07-14

**Canonical source:** [`pipeline.md`](pipeline.md)

This filename is retained for existing links. `pipeline.md` is the canonical implementation document and must be updated first when architecture or status changes.

## System overview

```text
case_id + case_query
        │
        ├── Case Content API ───────────────→ case evidence
        ├── BM25 / dense + RRF + reranker ─→ law evidence
        └── Qwen3-8B 4-bit ────────────────→ outcome label
                                                   │
                                                   ▼
                                    builder → validator → submission.json
```

Public development uses the same private-like production pipeline. Public gold annotations enter only the evaluator after prediction.

## Data boundary

```text
Public JSON ──→ InferenceCase(case_id, case_query) ──→ pipeline
      └──────→ PublicGold ───────────────────────────→ evaluator

Private JSON ─→ InferenceCase(case_id, case_query) ──→ pipeline
```

`verdict_label`, `case_fact`, `judgment_text`, `court_reasoning`, `court_verdict`, and `related_law_provisions` are prohibited as inference features.

Status: `CPU/mock verified`.

## Case evidence stage

The production retriever uses exactly two deterministic queries (`court_decision` and normalized `case_query`), calls official `POST /retrieve`, throttles to one request every five seconds, permits one bounded transient retry, caches successful responses in SQLite, and de-duplicates exact `chunk_id` values.

The official call count is permanent across runs. Preflight cache-miss planning, an explicit hard network cap, a safe SQLite attempt ledger, and cache/checkpoint reuse are therefore part of scoring safety, not only performance optimization.

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
                 │                     ├→ Vietnamese_Reranker → top 5
Exact citations ───────────────────────┘
```

The candidate uses `AITeamVN/Vietnamese_Embedding` and `AITeamVN/Vietnamese_Reranker`. Up to 12 corpus-valid citations extracted from case evidence expand the reranker pool without bypassing reranking. Evidence preserves corpus-valid `law_id` and `aid`.

Status: BM25 is `CPU/mock verified`; the hybrid candidate is `implemented`.

## Outcome stage

The predictor uses pinned `Qwen/Qwen3-8B`, NF4 4-bit, FP16 compute, double quantization, deterministic generation, and thinking disabled. The candidate's decision-first prompt prioritizes the main claim and `Tuyên xử`/`Quyết định` evidence. Under the current two-query policy, context includes `case_query`, up to two case segments, and up to five law articles.

The model returns internal `{reasoning, label}` JSON. The parser validates the official label and permits one repair attempt. A second failure creates an explicit failed result that cannot be submitted.

Status: `implemented`; a clean refreshed `GPU/API verified` run is pending.

## Submission and evaluation

The validator checks exact fields and labels, non-empty opaque identifiers, corpus-valid law pairs and strict types, duplicates, strict JSON, and the 10 MB limit. It is `CPU/mock verified`; official refreshed-format acceptance remains pending.

Public evaluation supports Outcome Accuracy, Micro Law Evidence F1, law Recall@5, format failures, and run-local API statistics. Public data does not contain gold case `chunk_id`, so official Penalized Case Recall and FinalScore require the leaderboard.

## Runtime and artifacts

Retrieval contexts are prepared before Qwen loads so embedding/reranker GPU memory can be released. Each run records:

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

Implementation priorities are maintained in [`plan.md`](plan.md); team choices and rejected alternatives are maintained in [`decisions.md`](decisions.md).
