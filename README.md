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

The local validator treats `chunk_id` as an opaque non-empty string and is `CPU/mock verified`. A prior incomplete Kaggle candidate smoke recorded two HTTP 200 Case API attempts and two cache rows, but stopped during embedding-model download before completing any case. No clean end-to-end run is `GPU/API verified`, and no current result is `leaderboard verified`.

Case Content API calls accumulate permanently across Public and Private runs. Every live run requires an explicit network-attempt cap; Public model evaluation defaults to zero-network `cache-only`. The live baseline/candidate policy uses only `court_decision` plus the normalized `case_query` for each case.

Official participation rules allow only open-weight models with fewer than 10 billion parameters, prohibit proprietary/non-open model APIs, and prohibit externally annotated legal QA or legal entailment datasets. The current Qwen3-8B stack fits the model contract.

## Repository layout

```text
configs/            Baseline and candidate configurations
data/raw/           Organizer data; the local Private Test file is git-ignored
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

## Google Colab workflow

`notebooks/colab_rag.ipynb` is the canonical Drive-first runner for both tracks. The first `runtime_check` for a `RUN_ID` resolves the current `TuanAnh` head and persists its exact commit SHA; later smoke, full, and resume stages detach at that SHA. Use a new `RUN_ID` to adopt newer branch code. `GITHUB_TOKEN` is required for a private repository, `HF_TOKEN` is optional, and `ALQAC_TEAM_TOKEN` is read only for Private `live` execution. The notebook restores the shared SQLite cache and fingerprinted law index to local Colab storage, verifies embedding/reranker/Qwen before any live request, checkpoints each case to Drive, and exports only validated artifacts. `requirements-colab.txt` preserves Colab's preinstalled Torch/CUDA build.

Recommended Public order:

1. `RUN_MODE='runtime_check'`, `TRACK='public'`, `EXECUTION_MODE='cache-only'`.
2. Keep the same `RUN_ID` and use `RUN_MODE='smoke'` for two cases.
3. Keep the same `RUN_ID` and use `RUN_MODE='full'` for all 50 cases.

Public cache-only execution neither reads `ALQAC_TEAM_TOKEN` nor instantiates the HTTP client and records zero network attempts. It evaluates Outcome Accuracy and Law F1, writes `selection_profile.json`, and does not estimate official Case Recall.

Equivalent Public CLI:

```bash
python scripts/run_public.py \
  --config configs/candidate.yaml \
  --execution-mode cache-only \
  --resume-run outputs/public_candidate_full \
  --cache-db cache/case_api.sqlite \
  --law-index-dir cache/law_index \
  --max-network-calls 0
```

For Private, place the immutable 60-case file at `MyDrive/ALQAC2026/inputs/private/ALQAC_private_test.json`. Use an exact Git commit, run the model-only gate, then a two-case live smoke with cap four. Run all 60 cases in a different directory, reuse the smoke cache, and set the full cap to current cache misses plus the explicitly approved retry reserve.

Equivalent Private CLI:

```bash
python scripts/run_private.py \
  --config configs/candidate.yaml \
  --input data/raw/ALQAC_private_test.json \
  --execution-mode live \
  --resume-run submissions/private_candidate_full \
  --cache-db cache/case_api.sqlite \
  --cache-backup-db /approved/external/case_api.sqlite \
  --law-index-dir cache/law_index \
  --selection-profile /approved/public/selection_profile.json \
  --max-network-calls APPROVED_PRIVATE_BUDGET
```

The validator accepts exact opaque API identifiers without inferring `_chunk_` or `_seg_` prefixes. Fine-tuning is not part of the production path; an optional `--adapter-path` can load a separately approved LoRA adapter.

## Run artifacts

A run directory contains:

- `submission.json` and `validation.json`, cryptographically bound by the
  submission SHA-256 and byte length recorded in validation and manifest;
- `config.resolved.json` and `environment.json`;
- `manifest.json` and `api_stats.json`, plus `api_plan.json` for non-mock retrieval;
- `runtime_check.json` from the zero-API model gate and Public `selection_profile.json` when available;
- context/prediction checkpoints;
- internal predictions; and
- Public-only metrics/error analysis when applicable.

Reasoning, evidence text, API counters, raw model output, and tokens are never included in `submission.json`.

## Leaderboard workflow

The submission owner manually uploads `submission.json` with the organizer-issued team name and token. Current official pages conflict: the competition website says at most three submissions per day, while the leaderboard rules say 20 per team per 24 hours. Until clarified, follow the stricter team limit of at most three submissions in any 24-hour period.

Public Test can be submitted repeatedly within that limit. Private Test permits at most three distinctly named runs in total, requires **Check format** before final confirmation, and uses the best run. The source code never uploads automatically.

## Reproducibility and packaging

```bash
pytest
git diff --check
python scripts/package_source.py
```

The default bundle is `artifacts/alqac2026_source.zip`. It excludes data, secrets, caches, weights, logs, outputs, and submissions.

Operational details are in [docs/runbook.md](docs/runbook.md). The official competition website encourages a short technical report and says the organizers may request source code, configuration, or logs. Whether those become mandatory and their deadlines remains `Needs confirmation`.
