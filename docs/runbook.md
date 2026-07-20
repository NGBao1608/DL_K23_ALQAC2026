# ALQAC 2026 Runbook

**Last synchronized:** 2026-07-20

**Canonical references:** [`competition.md`](competition.md), [`data_api.md`](data_api.md), and [`submission.md`](submission.md)

Read the canonical references before any real API run.

## 1. Pre-flight

```bash
git status --short --branch
python -m pip install -e .
python -m pip check
pytest
```

Confirm:

- the intended branch, commit, input, and config;
- `ALQAC_TEAM_TOKEN` exists only in approved secret storage;
- Public/law files match configured paths;
- every model is open-weight and below 10 billion parameters;
- no proprietary model API or externally annotated legal QA/entailment dataset enters the run;
- Colab T4 and Internet are enabled for GPU/model execution;
- the run directory is new or belongs to the exact same run identity; and
- the team has approved the number and purpose of real API calls.

The Colab notebook records the clean-runtime `pip check` result before installing
project dependencies. It fails only when the ALQAC installation introduces a new
conflict, while reporting unrelated conflicts already present in the Colab image as
warnings. It also fails if the installation changes Colab's preinstalled `torch`
build. Restart the runtime before the first run of an updated notebook so that this
baseline is captured before dependency installation.

The project does not use Colab's preinstalled Gradio UI. Current Gradio 6.x requires
`huggingface-hub>=1`, which conflicts with the selected Transformers 4.x runtime.
The dependency cell therefore removes only `gradio` and `gradio-client` before
installing `requirements-colab.txt`; it still fails on every remaining new
`pip check` conflict.

## 2. CPU/mock smoke test

```bash
rm -rf outputs/smoke
python scripts/run_public.py \
  --mock \
  --limit 2 \
  --config configs/baseline.yaml \
  --resume-run outputs/smoke
```

Require `outputs/smoke/validation.json` to report `PASS`. Mock metrics and mock submissions are not model results and must never be uploaded.

The smoke artifact can be revalidated independently with:

```bash
python scripts/validate_submission.py \
  --input outputs/smoke/submission.json \
  --test-data data/raw/ALQAC2026_public_test.json \
  --limit 2
```

## 3. Colab model-only gate

Use `notebooks/colab_rag.ipynb` on one Google Colab T4 session. Before Run All, configure `GITHUB_TOKEN` for read-only private-repository access. `HF_TOKEN` is optional and only improves Hugging Face download reliability. Configure `ALQAC_TEAM_TOKEN` before a Private live run; Public cache-only stages do not read it. The notebook mounts `MyDrive/ALQAC2026`, resolves the current `TuanAnh` head during the first runtime gate for a `RUN_ID`, persists the exact commit SHA, restores the cache/index, and runs with:

```python
TRACK = "public"
RUN_MODE = "runtime_check"
EXECUTION_MODE = "cache-only"
```

`scripts/check_runtime.py` must load and execute embedding, reranker, and Qwen sequentially, write `runtime_check.json` with `status=PASS`, and report zero ALQAC API attempts. Do not proceed to live retrieval if any model download, CUDA allocation, index, adapter, or generation check fails.

The same `RUN_ID` must be kept for its runtime gate, two-case smoke, full run,
and resume. Later stages detach at the persisted commit and fail if the runtime
gate or smoke artifact belongs to another commit. Create a new `RUN_ID` to test
newer branch code.

## 4. Public cache-only validation

Public API experimentation is disabled because organizer logs are append-only across Public and Private activity. Run two cases first, then all 50 cases with the same source-pinned `RUN_ID`:

```bash
python scripts/run_public.py \
  --config configs/candidate.yaml \
  --execution-mode cache-only \
  --resume-run outputs/public_candidate_full \
  --cache-db cache/case_api.sqlite \
  --law-index-dir cache/law_index \
  --max-network-calls 0
```

Cache hits preserve previously retrieved exact evidence; misses produce an empty `case_evidence` list and never instantiate the HTTP client. Require 50 completed predictions, zero network attempts, `validation=PASS`, Outcome Accuracy, Law Micro F1, and `selection_profile.json`. The evaluator tries submission law top-k values 3 through 10, maximizes Public Micro Law F1, and breaks ties toward the smaller list.

This validates GPU inference, law retrieval, resume, formatting, and outcome/law metrics. It does not estimate official Case Recall or FinalScore. A Public upload remains optional and manual.

## 5. Private live run

The canonical file has SHA-256 `9db83cf98ade7d19df52c60145830bebcc192e064ec830bcd285cefbfddf0252` and 60 unique objects containing exactly `case_id` and `case_query`. Place it at `MyDrive/ALQAC2026/inputs/private/ALQAC_private_test.json`; never copy it into Git, source bundles, or exports.

Set `TRACK='private'`, `EXECUTION_MODE='live'`, a new `RUN_ID`, and the Public `selection_profile.json`. Complete the model-only gate to pin and verify one exact commit before the organizer token is read or exported to the runner. Keep that `RUN_ID` for smoke, full, and resume.

Run a two-case smoke in `<RUN_ID>-smoke` with hard cap four. Each success records the response, attempt, and pending-backup marker atomically, then publishes a verified SQLite backup to Drive before another request is allowed. A resumed live client must repair any pending backup before serving a cache hit or sending a request. Never upload the limited output.

Run all 60 cases in a different directory. Reuse the smoke cache; approve current cache misses plus at most four retry attempts. If that reserve is exhausted, stop and require an explicit new cap before resume. Do not resume the limited run as full.

Equivalent CLI:

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

Require exactly 60 completed cases, `validation=PASS`, exact opaque API identifiers, corpus-valid `{law_id, aid}`, a file below 10 MB, and a submission SHA-256/byte length that match both `validation.json` and `manifest.json`. Export rechecks these bindings and the actual JSON case count.

## 6. Manual leaderboard submission

1. Confirm the candidate is a full completed production run, not mock or limited smoke output.
2. Confirm validation passes under the refreshed rules.
3. Review manifest, config, Git revision, and API statistics.
4. Apply the stricter team limit of at most three submissions in any 24-hour period because the current official pages conflict.
5. For Private Test, confirm that fewer than three distinctly named runs have been spent and select a new run name tied to the manifest.
6. The submission owner manually enters the team name/token and uploads `submission.json`.
7. For Private Test, use **Check format**, inspect the reported case count and remaining slots, then manually confirm.
8. Record official metrics and the run name in `experiments.md` without storing credentials.

The Public leaderboard displays the best run per team. Private rankings are hidden publicly, the team can see its own score on the Submit page, and the best of at most three Private runs counts. Source code must never upload automatically.

## 7. Source bundle

```bash
pytest
git diff --check
python scripts/package_source.py
```

Inspect the archive for README, config, source, scripts, notebooks, tests, and docs. It must exclude `.env`, raw/private data, caches, weights, logs, outputs, submissions, and tokens.

The official competition website encourages a short technical report and says organizers may request source code, configuration files, or logs for verification and reproducibility. `Needs confirmation`: whether either artifact becomes mandatory and what deadline or transfer procedure applies.

## Failure handling

| Failure | Action |
|---|---|
| HTML/ngrok response | Record safe status/content type/error code; do not expose headers or token; wait for organizer infrastructure. |
| `403` JSON response | Check secret name/value and organizer activation; do not retry. |
| `422` | Fix `query`/`case_id`; do not repeat malformed calls. |
| `429` | Wait at least five seconds and use bounded backoff. |
| `503` | Use bounded exponential retry and resume. |
| CUDA OOM during retrieval | Reduce reranker batch size and release retrieval models before Qwen. |
| CUDA OOM during Qwen | Confirm NF4, context budget, and released retrieval models. |
| Invalid model output | Allow one repair attempt; do not invent a fallback label. |
| Resume identity mismatch | Use the original source/config/input or start a new run directory. |
| Validator rejects a non-empty exact API ID | Do not upload; preserve the artifact and investigate the validator/API schema conflict. |
| API budget smaller than cache misses | Stop before retrieval; review preflight and explicitly approve a sufficient cap. |
| API budget exhausted during retry | Preserve cache/checkpoints; review the consumed local attempts before approving a resume budget. |
