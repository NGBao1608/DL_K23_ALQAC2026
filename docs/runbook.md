# ALQAC 2026 Runbook

**Last synchronized:** 2026-07-23

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
The dependency cell therefore removes only the unused `gradio`, `gradio-client`,
and `hf-gradio` UI stack before installing `requirements-colab.txt`; it still
fails on every remaining new `pip check` conflict.

After the editable install, the notebook clears any `alqac2026` modules retained
by the current kernel, puts the checked-out `<project>/src` first on `sys.path`,
invalidates Python's importer cache, and verifies that both `alqac2026` and
`alqac2026.artifacts` resolve from the pinned commit. This provenance gate is
required because Run All deletes and reclones `/content/alqac2026` without
necessarily restarting the Python kernel.

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

## 3. Canonical Colab stages

The user-facing Colab contract has only two stages:

```python
STAGE = "smoke"  # smoke | full
RUN_ID = "one-stable-experiment-id"
```

Use `notebooks/colab_public.ipynb` for Public and
`notebooks/colab_private.ipynb` for Private. Configure `GITHUB_TOKEN` for
read-only private-repository access and `ALQAC_TEAM_TOKEN` for live retrieval.
`HF_TOKEN` is optional and only improves Hugging Face download reliability.

Smoke performs all safety gates in one operation:

1. resolve the selected branch and persist its exact commit in
   `source_pin.json`;
2. restore the shared SQLite cache and the fingerprinted law index;
3. run the query planner, embedding, reranker, and Qwen through
   `scripts/check_runtime.py` with zero Case Content API attempts; and
4. run exactly two live cases with a hard cap of six network attempts
   (`2 cases × 3 attempts`).

Full requires `runtime_check.json` and `smoke_gate.json` from the same pinned
commit and `workflow_config.json` fingerprint. It uses its own `full/`
directory and automatically resumes that directory after interruption. Smoke
checkpoints are never promoted into the full prediction set; the shared query
plans, evidence cache, and law index are reused. Create a new `RUN_ID` to adopt
newer source code or a different workflow configuration.

## 4. Public quality evaluation

Public quality evaluation needs real case evidence, so the canonical Public
notebook uses live retrieval. Set `APPROVE_PUBLIC_API_CALLS=True` only after
reviewing the two-primary-plus-adaptive policy and budget. The reviewed Public and Private files
have zero overlapping `case_id` values. Because the official formula defines
the penalty per case, Public calls should not directly increase a Private
case's `c_i`; `Needs confirmation`: whether the organizer applies any
additional team-wide cross-track accounting.

Run:

1. `STAGE='smoke'` with a new Public `RUN_ID`; require two completed cases,
   `validation=PASS`, a passing zero-call model gate, and no more than six
   network attempts.
2. Keep the same `RUN_ID`, set `STAGE='full'`, and run all 50 cases with the
   same configuration fingerprint and cap `50 × 3 = 150`.
3. Require 50 completed predictions, `validation=PASS`, Outcome Accuracy, Law
   Micro F1, Recall@5, error analysis, and `selection_profile.json`.

The evaluator reads Public gold only after prediction. It tests submission law
top-k values 3 through 10, maximizes Public Micro Law F1, and breaks ties toward
the smaller list. The current Public data lacks gold case `chunk_id`, so local
evaluation cannot report official Case Recall, Penalized Case Recall, or
FinalScore. Public upload remains optional and manual.

## 5. Private live inference

The source-pinned private GitHub checkout must contain both immutable files:

| File | Reviewed contract |
|---|---|
| `ALQAC_private_test.json` | SHA-256 `9db83cf98ade7d19df52c60145830bebcc192e064ec830bcd285cefbfddf0252`; 60 unique two-field cases |
| `private_test_60_cases_extracted_corpus.json` | SHA-256 `9d79379e017ce346cf143a71fa82f5170a755c33a0341048c9baacb28c6119b5`; 14 laws and 2,820 articles |

Both files are permitted only in the verified private repository and may not
enter source bundles, run artifacts, or exports. The current prepared notebook
sets `PUBLIC_RUN_ID=public-candidate-v7`. Private requires that Public full
directory to contain `selection_profile.json`, `validation.json`, and
`manifest.json`, with `validation=PASS` and all 50 cases completed. Private
reads only the selected scalar from
`runs/public/<PUBLIC_RUN_ID>/full/selection_profile.json`.

Run:

1. `STAGE='smoke'` with a new Private `RUN_ID`. The notebook validates both
   source-pinned Private files, builds or restores an index fingerprinted from the Private
   law corpus, runs the zero-call model gate, and predicts exactly two live
   cases with cap six. Never upload this limited output.
2. Keep the same Private `RUN_ID`, set `STAGE='full'`, and run all 60 cases.
   The full run reuses the shared query plans/API cache, derives cap
   `60 × 3 = 180`, and
   automatically resumes its own checkpoints after interruption.
3. Require 60 completed cases, `validation=PASS`, exact opaque API identifiers,
   corpus-valid `{law_id, aid}`, a file below 10 MB, and matching submission
   SHA-256/byte length in validation and manifest.

Private has no local outcome or evidence labels, so the notebook does not run
the Public evaluator. It exports only the complete validated full submission.

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
| CUDA OOM during Qwen | Restart into a clean T4 runtime and resume the same pinned run. Candidate generation uses offloaded KV cache at 4,096/192 tokens; the first OOM retries once with 3,072/160 tokens and two laws, while a second OOM records deterministic fallback. |
| Invalid model output | Allow one repair attempt; do not invent a fallback label. |
| Resume identity mismatch | Use the original source/config/input or start a new run directory. |
| Git clone command contains `[https://...](https://...)` | Use a raw GitHub URL. Current notebooks normalize this accidental Markdown wrapper before clone. |
| Validator rejects a non-empty exact API ID | Do not upload; preserve the artifact and investigate the validator/API schema conflict. |
| API budget smaller than cache misses | Stop before retrieval; review preflight and explicitly approve a sufficient cap. |
| API budget exhausted during retry | Preserve cache/checkpoints; review the consumed local attempts before approving a resume budget. |
