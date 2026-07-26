# ALQAC 2026 Architecture

## Design goals

- A single production pipeline shared across the CLI, Kaggle, and Colab.
- Private inference receives only `case_id` and `case_query`.
- API calls, model outputs, config, and source revision are all traceable.
- Runs stage by stage on a 16 GB T4 GPU.
- Submissions always pass through the strict validator and are never uploaded automatically.

## Data boundary

`load_inference_cases()` converts both public and private JSON into `InferenceCase(case_id, case_query)`. This object carries no gold fields.

Public evaluation reads gold via `load_public_gold()` only after inference completes. The fields `verdict_label`, `case_fact`, `judgment_text`, `court_reasoning`, `court_verdict`, and `related_law_provisions` are never passed into `ALQACPipeline`.

```text
Public JSON ──→ InferenceCase ──→ production pipeline ──→ PredictionResult
      └──────→ PublicGold ─────────────────────────────→ evaluator

Private JSON ─→ InferenceCase ──→ production pipeline ──→ submission
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `schemas.py` | Inference, evidence, label, and result types |
| `data.py` | Load and validate public, private, and law corpus; resolve public law gold |
| `case_retrieval.py` | Query generation, API client, rate limit, retry, SQLite cache |
| `citations.py` | Extract cited law provisions from evidence; procedural-prior articles |
| `law_retrieval.py` | BM25, dense embeddings, RRF, and reranking (fallback search) |
| `prediction.py` | Prompt, Qwen3 backend, JSON parser, and repair (single-pass predictor) |
| `ensemble.py` | Multi-signal outcome ensemble: self-consistency, precedents, thinking |
| `vks.py` | Procuracy (Viện Kiểm Sát / VKS) stance signal |
| `pipeline.py` | Prepare context, predict, checkpoint serialization |
| `evaluation.py` | Public outcome accuracy, law micro F1, and optional case recall |
| `submission.py` | Build official schema and strict validation |
| `runner.py` | Stage orchestration, artifacts, resume, and manifest |

Dependencies flow in one direction only:

```text
scripts/notebooks → runner → pipeline
pipeline → case_retrieval, law_retrieval, prediction
evaluation/submission → schemas
```

Notebooks do not redefine production functions, do not change the working directory in code, do not modify `sys.path`, and do not monkey-patch the API client.

## Case evidence stage

Each case uses up to eight queries, in order:

1. The court's ruling (Quyết định của Tòa án tuyên xử).
2. Acceptance of the plaintiff's claim (chấp nhận yêu cầu khởi kiện của nguyên đơn).
3. Rejection of the plaintiff's claim (không chấp nhận yêu cầu của nguyên đơn).
4. The trial panel's assessment (nhận định của Hội đồng xét xử).
5. Application of legal provisions (áp dụng điều luật).
6. First-instance civil court fees (nghĩa vụ án phí dân sự sơ thẩm).
7. The `tranh chấp ...` ("dispute over ...") phrase extracted from `case_query`, if present.
8. The original, normalized `case_query`.

The client enforces a minimum gap of 5 seconds between network calls, applies exponential retry for `429/5xx`, stops immediately on `403/422`, and caches successful responses keyed by a hash of API version + case ID + normalized query.

Evidence is deduplicated by `chunk_id`; when the same chunk appears multiple times, the highest-scoring hit is kept. `api_calls` only counts the network calls of the run and is not included in the submission.

## Law retrieval stage

The primary path is citation-based, not search-based (`citations.py`, `pipeline.py`):

```text
case evidence → extract cited provisions ("áp dụng Điều 584, 558 ... của Bộ luật Dân sự")
              + near-universal civil-procedure articles (procedural priors)
              → deduplicated law evidence
```

Vietnamese first-instance judgments cite the applied provisions literally in the retrieved evidence, so those citations are extracted directly and merged with a small fixed set of civil-procedure articles that almost every case applies (jurisdiction, court fees, trial-in-absentia, appeal window).

A search-based fallback runs only when no citation or corpus context is available (for example a mock run). Its retriever depends on the config's `strategy`:

`bm25_only` (used by `baseline.yaml` and `ensemble.yaml`):

```text
case query + case evidence → BM25 → top 5
```

`hybrid_rerank` (used by `candidate.yaml`):

```text
BM25 top 50 ─────┐
                 ├→ RRF(k=60) top 30 → Vietnamese_Reranker → top 5
Dense top 50 ────┘
```

Dense embeddings use normalized vectors and dot product. The index covers 3,352 articles and is cached together with the model revision and the list of article keys; a cache with mismatched metadata is rebuilt.

After contexts are prepared for all cases, the embedding and reranker models are released and the CUDA cache is cleared before Qwen3 is loaded.

## Outcome stage

The adjudicator prompt gives Qwen3:

- `case_query`.
- Up to eight case chunks, ordered decision-first (a chunk whose text contains a ruling lead comes first, then by query-type priority and score).
- Up to five law articles.

The model must return `{"reasoning":"...","label":"..."}`. The parser accepts only the four official labels. On a parse error, the model is called exactly once more with a repair prompt.

Two predictor configurations share this prompt:

**Single-pass predictor** (`baseline.yaml`, `candidate.yaml`). One greedy generation with thinking disabled and output up to 384 tokens. If both the initial call and the repair fail to parse, it defers to the procuracy (VKS) stance when unambiguous, otherwise produces `PredictionResult(status="failed")`.

**Ensemble predictor** (`ensemble.yaml`, `use_ensemble: true`). A multi-signal predictor where each signal is an independent config toggle (`ensemble.py`). The winning configuration enables:

- **Self-consistency** (`self_consistency: 5`) — five sampled generations (temperature 0.7) drawn in one batched call; the majority label wins.
- **Precedent case-based reasoning** (`use_precedents: true`, `num_precedents: 3`) — the three most similar labelled public cases, retrieved by embedding similarity, are added to the prompt as reference outcomes. The queried case is always excluded, so a case never sees its own gold label.
- **Thinking adjudicator** (`thinking: true`) — reasoning mode is enabled and the output budget is raised to 1,024 tokens.

Additional toggles exist but are off in the shipped config: an LLM-extracted VKS stance block and a dual-advocate debate. When no signal is enabled the ensemble prompt is identical to the single-pass prompt.

With no signals the base prompt reduces to the single-pass prompt, so each signal adds context on top of that base. Any failure in the ensemble degrades gracefully: it falls back first to a single greedy generation, then to the majority class. Runtime is Qwen3-8B, NF4 4-bit, double quantization, FP16 compute, input up to 7,000 tokens. The submission builder rejects any failed result.

## Cache, checkpoint, and resume

- `cache/case_api.sqlite`: persistent successful API responses.
- `cache/law_index/`: dense embeddings and metadata.
- `<run>/contexts.checkpoint.json`: prepared case and law contexts.
- `<run>/predictions.checkpoint.json`: completed and failed prediction records.
- `--resume-run <directory>`: reuses the checkpoint; the config must match `config.resolved.json` exactly.

On resume, completed predictions are filtered out before retrieval and model loading. Prepared contexts are taken from the checkpoint; API and law retrieval run only for the contexts still missing. Mock mode, limit, input, config, and source fingerprint are part of the run identity and cannot change on resume.

## Evaluation limitations

- Outcome accuracy is computed over all public cases; a failed prediction is counted as incorrect.
- Law evidence uses micro F1 over the gold provisions that resolve to the current corpus.
- The public file currently has no official gold `chunk_id`, so `case_evidence_recall` and `final_score` return `null` offline.
- The official Penalized Case Recall and final score must be taken from the leaderboard, because the organizers do not provide the gold chunks or an executable formula.

## Artifact layout

```text
<run>/
├── config.resolved.json
├── environment.json
├── contexts.checkpoint.json
├── predictions.checkpoint.json
├── predictions.json
├── api_stats.json
├── submission.json
├── validation.json
├── manifest.json
├── metrics.json              # public run only
└── errors.json               # public run only
```

The manifest records model names and revisions, license metadata, corpus SHA-256, Git commit, dirty state, resolved config, case count, and API call count.
