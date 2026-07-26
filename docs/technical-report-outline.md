# Technical report outline

This is a source-backed outline. Do not add scores until a reproducible artifact exists.

## 1. Task and constraints

- Legal Case Outcome Prediction with Evidence Retrieval.
- Four outcome labels and official metric weights.
- Private-test boundary, API and open-weight model restriction.

## 2. Data

- Public development set statistics.
- Law corpus statistics.
- Separation of inference fields and public gold annotations.

## 3. System

- Deterministic Case Content API query strategy.
- BM25 + Vietnamese Embedding + RRF + Vietnamese Reranker.
- Qwen3-8B NF4 outcome prediction and structured output validation.
- Cache, checkpoint and staged T4 execution.

## 4. Experiments

- BM25 baseline.
- Hybrid retrieval ablation.
- Outcome accuracy and law evidence metrics.
- API efficiency and failure statistics.

Every table row must include config, Git revision, model revisions and run directory.

## 5. Error analysis

- Confusion between full and partial wins.
- Missing/incorrect case evidence.
- Missing/incorrect law evidence.
- JSON or runtime failures.

## 6. Reproducibility

- Environment and hardware.
- Commands for public/private runs.
- Source bundle contents and known limitations.

