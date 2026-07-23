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

Case Content API calls accumulate permanently for each `case_id`. The reviewed
Public and Private files have no overlapping identifiers, so Public calls should
not directly increase a Private case's `c_i`; whether any additional team-wide
cross-track accounting exists is `Needs confirmation`. Every live run still
requires explicit approval and a hard network-attempt cap. The live
baseline/candidate policy uses only `court_decision` plus the normalized
`case_query` for each case.

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

The canonical workflow is split into two thin notebooks:

- `notebooks/colab_public.ipynb`: live two-case smoke, full 50-case run,
  evaluator, law top-k selection, and validated candidate export.
- `notebooks/colab_private.ipynb`: live two-case smoke, full 60-case inference,
  validation, and submission export without gold evaluation.

Both notebooks expose only `STAGE='smoke'` and `STAGE='full'`. Smoke
automatically runs the zero-API embedding/reranker/Qwen gate before its two live
cases. Full requires the passing smoke gate, uses the same pinned Git commit, and
automatically resumes its own checkpoints. Use a new `RUN_ID` to adopt newer
branch code.

`GITHUB_TOKEN` is required for a private repository, `ALQAC_TEAM_TOKEN` is
required for either live notebook, and `HF_TOKEN` is optional. The workflow
restores the shared SQLite cache and fingerprinted law index to local Colab
storage, then writes checkpoints and safe exports to
`MyDrive/ALQAC2026`. `requirements-colab.txt` preserves Colab's preinstalled
Torch/CUDA build.

The dependency cell removes the unused preinstalled `gradio`, `gradio-client`, and
`hf-gradio` UI stack before installing Transformers 4.x. This avoids its incompatible
Hugging Face Hub requirement while keeping the final `pip check` fail-closed.
It also clears stale `alqac2026` imports after recloning and verifies that the
package and `artifacts` module resolve from the exact source-pinned checkout.

Public order:

1. Set `STAGE='smoke'` and explicitly approve Public API calls after reviewing
   the maximum four-attempt budget.
2. Keep the same `RUN_ID`, set `STAGE='full'`, and run all 50 cases.
3. Review Outcome Accuracy, Law Micro F1, Recall@5, error analysis, and
   `selection_profile.json`.

For Private, place both immutable organizer files under
`MyDrive/ALQAC2026/inputs/private/`: `ALQAC_private_test.json` and
`private_test_60_cases_extracted_corpus.json`. Set the successful Public
`RUN_ID` so Private can read
`runs/public/<RUN_ID>/full/selection_profile.json`. Run Private smoke first,
then keep the same Private `RUN_ID` for full. Smoke uses at most four network
attempts; full persists a budget equal to current cache misses plus the selected
retry reserve.

The validator accepts exact opaque API identifiers without inferring `_chunk_` or `_seg_` prefixes. Fine-tuning is not part of the production path; an optional `--adapter-path` can load a separately approved LoRA adapter.

## Run artifacts

Each `runs/<track>/<RUN_ID>/` workflow contains:

- `source_pin.json`, `runtime_check.json`, `smoke_gate.json`, and API plans;
- separate `smoke/` and `full/` run directories;
- config, environment, contexts, predictions, API statistics, validation, and manifest;
- Public-only metrics, error analysis, and `full/selection_profile.json`; and
- a validated full-run export under `exports/<RUN_ID>/`.

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
