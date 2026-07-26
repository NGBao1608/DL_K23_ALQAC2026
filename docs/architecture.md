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
| `law_retrieval.py` | BM25, dense embeddings, RRF, and reranking |
| `prediction.py` | Prompt, Qwen3 backend, JSON parser, and repair |
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

`baseline.yaml`:

```text
case query + case evidence → BM25 → top 5
```

`candidate.yaml`:

```text
BM25 top 50 ─────┐
                 ├→ RRF(k=60) top 30 → Vietnamese_Reranker → top 5
Dense top 50 ────┘
```

Dense embeddings use normalized vectors and dot product. The index covers 3,352 articles and is cached together with the model revision and the list of article keys; a cache with mismatched metadata is rebuilt.

After contexts are prepared for all cases, the embedding and reranker models are released and the CUDA cache is cleared before Qwen3 is loaded.

## Outcome stage

Qwen3 receives:

- `case_query`.
- Up to eight case chunks, prioritizing decision/accepted/rejected/reasoning.
- Up to five law articles.

Default runtime: Qwen3-8B, NF4 4-bit, double quantization, FP16 compute, thinking disabled, input up to 7,000 tokens and output up to 384 tokens.

The model must return `{"reasoning":"...","label":"..."}`. The parser accepts only the four official labels. On a parse error, the model is called exactly once more with a repair prompt; a second failure produces `PredictionResult(status="failed")`. The submission builder rejects failed results.

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
