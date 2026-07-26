# ALQAC 2026 — Legal Case Outcome Prediction (Team K23)

A reproducible pipeline for the ALQAC 2026 Legal Case Outcome Prediction with
Evidence Retrieval task. Given a case described only by its `case_id` and a short
`case_query`, the system predicts a four-class trial outcome and returns the
supporting case-evidence chunks and law provisions.

## Pipeline

The pipeline runs in four stages, all under `src/alqac2026/`:

1. **Case-evidence retrieval** (`case_retrieval.py`) — queries the official Case
   Content API with a prioritized bank of section-targeted queries, with a local
   SQLite cache, request throttling, and a saturation early-stop that bounds the
   number of API calls per case.
2. **Law retrieval** (`citations.py`, `law_retrieval.py`) — extracts the law
   provisions the court cited literally in the retrieved evidence, adds a small
   set of near-universal civil-procedure articles as a prior, and falls back to a
   hybrid BM25 + dense-embedding + reranker search when no citation context is
   available.
3. **Outcome prediction** (`prediction.py`, `ensemble.py`, `vks.py`) — predicts
   one of `A_WIN`, `PARTIAL_A_WIN`, `PARTIAL_B_WIN`, `B_WIN` with a Qwen3-8B
   ensemble. Each signal is a configuration toggle: self-consistency majority
   voting, precedent case-based reasoning over the labelled public cases, a
   reasoning ("thinking") adjudicator, an extracted procuracy (Viện Kiểm Sát /
   VKS) stance, and dual-advocate debate.
4. **Submission assembly** (`submission.py`) — builds and strictly validates the
   `submission.json` against the corpus and the input case set.

The prediction step reads no gold label of the case being predicted; precedents
surface other cases' labels only and always exclude the queried case.

## Repository layout

```
src/alqac2026/     Pipeline modules (retrieval, prediction, ensemble, submission, evaluation)
configs/           Run configurations: baseline.yaml, candidate.yaml, ensemble.yaml
scripts/           Entry points: run_public, run_private, validate_submission, and helpers
tests/             Unit tests
notebooks/         colab_submission.ipynb — run harness for public and private phases
docs/              Architecture, competition rules, runbook, and technical-report outline
```

## Installation

Requires Python 3.10+.

```bash
python -m pip install -e .
```

Copy `.env.example` to `.env` and set `ALQAC_TEAM_TOKEN` (the Case Content API
credential). On Colab, provide it through Colab Secrets instead of committing it.

## Running

Both entry points take a `--config` and an `--input` file and write a validated
submission plus a run manifest into the output directory.

```bash
# Public phase
python scripts/run_public.py \
  --config configs/ensemble.yaml \
  --input data/raw/ALQAC2026_public_test.json

# Private phase (input released by the organizers)
python scripts/run_private.py \
  --config configs/ensemble.yaml \
  --input path/to/private_test.json
```

`configs/ensemble.yaml` is the production configuration: it enables the full
outcome ensemble (self-consistency 5, precedents, thinking) and is run on an A100
GPU. `configs/baseline.yaml` and `configs/candidate.yaml` provide lighter
single-pass configurations for ablation. Add `--mock` for a dependency-free smoke
run that uses neither the API nor a GPU, or `--limit N` to run the first `N` cases.

The Colab harness in `notebooks/colab_submission.ipynb` wraps the same commands
for an A100 runtime, including cache persistence and resumable runs.

## Validating a submission

```bash
python scripts/validate_submission.py \
  --input submission.json \
  --test-data data/raw/ALQAC2026_public_test.json \
  --law-corpus data/raw/corpus_law_pub.json
```

The validator checks case coverage, label validity, evidence shapes, chunk-id
prefixes, and law-provision references, and fails loudly on any violation.

## Models

All models are open-weight and under 10B parameters:

- Outcome prediction: `Qwen/Qwen3-8B` (loaded in 4-bit).
- Embedding: `AITeamVN/Vietnamese_Embedding` — used for precedent retrieval in the
  ensemble, and for dense law retrieval in the hybrid fallback.
- Reranker: `AITeamVN/Vietnamese_Reranker` — used only by the `hybrid_rerank` law
  configuration (`candidate.yaml`); the production `ensemble.yaml` uses `bm25_only`,
  so the reranker is not loaded there.

Model revisions are pinned in the config files and recorded in each run manifest
for reproducibility.

## Tests

```bash
pytest -q
```
