# ALQAC 2026 — Legal Case Outcome Prediction

This repository implements the team K23 pipeline for **Legal Case Outcome Prediction with Evidence Retrieval** at ALQAC 2026.

```text
case_id + case_query
  ├─ Case Content API → case evidence
  ├─ BM25 / Vietnamese embedding + reranker → law evidence
  └─ Qwen3-8B 4-bit → outcome prediction
       ↓
validated submission.json
```

## Read first

1. [Competition requirements](docs/competition.md) — official rules, scoring, timeline, and open questions.
2. [Data and API](docs/data_api.md) — raw files, API contract, permanent call accounting, and secret handling.
3. [Submission specification](docs/submission.md) — schema, labels, validation, and upload checklist.
4. [Pipeline](docs/pipeline.md) — current implementation and verification status.
5. [Work plan](docs/plan.md) — P0/P1/P2 priorities.
6. [Technical decisions](docs/decisions.md) and [experiments](docs/experiments.md).

## Critical current warning

The local validator now treats `chunk_id` as an opaque non-empty string and is `CPU/mock verified`. A refreshed real API response and leaderboard result are still pending, so do not claim `GPU/API verified` or `leaderboard verified` yet.

Case Content API calls accumulate permanently across Public and Private runs. Every non-mock run requires an explicit network-attempt cap, and the current baseline/candidate policy uses only `court_decision` plus the normalized `case_query` for each case.

## Repository layout

```text
configs/            Baseline and candidate configurations
data/raw/           Organizer-provided Public Test and law corpus
data/private/       Private Test input; never commit
src/alqac2026/      Production package
scripts/            CLI entry points
notebooks/          Thin Kaggle/Colab orchestration
tests/              Unit and integration smoke tests
docs/               Requirements, architecture, plan, and runbook
cache/              API cache and law index; never commit
outputs/            Public run artifacts; never commit
submissions/        Private run artifacts; never commit
```

Current configurations:

- `configs/baseline.yaml`: Case API + BM25 law retrieval + Qwen3.
- `configs/candidate.yaml`: Case API + citation-aware BM25/dense RRF + Vietnamese reranker + decision-first Qwen3 prompt.

Do not create or promote `final.yaml` until a clean Public comparison and official score support the decision.

## Local setup

Python 3.10+ is required. Qwen3 NF4 inference requires Linux with an NVIDIA CUDA GPU. Local macOS is intended for tests, validation, BM25, and mock runs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Set `ALQAC_TEAM_TOKEN` only in `.env` or approved secret storage. Never commit or print it.

Verify the environment:

```bash
python -m pip check
pytest
```

## CPU/mock smoke test

This test does not call the official API and does not load Qwen3:

```bash
rm -rf outputs/smoke
python scripts/run_public.py \
  --mock \
  --limit 2 \
  --config configs/baseline.yaml \
  --resume-run outputs/smoke
```

`outputs/smoke/validation.json` must report `PASS`. Mock metrics are not model results and the resulting file must never be uploaded.

## Public development

Use `notebooks/public_development.ipynb` on Kaggle T4. Keep `RUN_MODE='smoke'` for the first real two-case run. Only use `RUN_MODE='full'` after the smoke run completes with valid API/model artifacts.

CLI equivalents:

```bash
python scripts/plan_api_calls.py \
  --config configs/baseline.yaml \
  --input data/raw/ALQAC2026_public_test.json \
  --cache-db cache/case_api.sqlite \
  --output outputs/public_baseline_api_plan.json

python scripts/run_public.py \
  --config configs/baseline.yaml \
  --resume-run outputs/public_baseline_full \
  --cache-db cache/case_api.sqlite \
  --max-network-calls APPROVED_BASELINE_BUDGET

python scripts/build_law_index.py --config configs/candidate.yaml
python scripts/run_public.py \
  --config configs/candidate.yaml \
  --resume-run outputs/public_candidate_full \
  --cache-db cache/case_api.sqlite \
  --max-network-calls 0
```

Replace `APPROVED_BASELINE_BUDGET` with the reviewed integer from preflight. The candidate command is valid only when preflight reports zero cache misses.

Do not run these commands against the real API merely to test setup. Each network call permanently affects the official per-case efficiency count.

## Private inference

After the organizer releases the Private Test input:

```bash
python scripts/run_private.py \
  --config configs/candidate.yaml \
  --input data/private/private_test.json \
  --resume-run submissions/private_candidate_full \
  --cache-db cache/private_case_api.sqlite \
  --max-network-calls APPROVED_PRIVATE_BUDGET
```

Validate the exact candidate file:

```bash
python scripts/validate_submission.py \
  --input submissions/private_candidate_full/submission.json \
  --test-data data/private/private_test.json
```

The validator accepts exact opaque API identifiers without inferring `_chunk_` or `_seg_` prefixes.

## Run artifacts

A run directory contains:

- `submission.json` and `validation.json`;
- `config.resolved.json` and `environment.json`;
- `manifest.json` and `api_stats.json`, plus `api_plan.json` for non-mock retrieval;
- context/prediction checkpoints;
- internal predictions; and
- Public-only metrics/error analysis when applicable.

Reasoning, evidence text, API counters, raw model output, and tokens are never included in `submission.json`.

## Leaderboard workflow

The submission owner manually uploads `submission.json` with the organizer-issued team name and token. The official limit is 20 submissions per team per 24 hours, and the public leaderboard displays the best run per team. The source code never uploads automatically.

## Reproducibility and packaging

```bash
pytest
git diff --check
python scripts/package_source.py
```

The default bundle is `artifacts/alqac2026_source.zip`. It excludes data, secrets, caches, weights, logs, outputs, and submissions.

Operational details are in [docs/runbook.md](docs/runbook.md). Source-code and technical-report delivery requirements are currently `Needs confirmation` because the refreshed official pages do not state them.
