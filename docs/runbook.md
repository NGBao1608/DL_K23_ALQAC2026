# ALQAC 2026 Runbook

**Last synchronized:** 2026-07-14

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
- Kaggle T4 and Internet are enabled for GPU/API execution;
- the run directory is new or belongs to the exact same run identity; and
- the team has approved the number and purpose of real API calls.

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

## 3. Real two-case smoke test

Use `notebooks/public_development.ipynb` with `RUN_MODE='smoke'`.

Set `APPROVED_MAX_NETWORK_CALLS=4`, use the external writable SQLite cache, and review the preflight report before running. After running, verify:

- the Case API returned one segment per successful call;
- exact returned `chunk_id` values were preserved;
- successful calls were cached;
- no successful query was repeated;
- the five-second rate limit was respected;
- Qwen3 loaded without OOM;
- every model output parsed successfully; and
- artifacts contain no secrets.

The local opaque-ID validator must report `PASS`. This is still only `CPU/mock verified` until exact refreshed API identifiers complete a real smoke run.

## 4. Full Public runs

Run baseline first, then candidate using the same cached case evidence:

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

Replace `APPROVED_BASELINE_BUDGET` with the reviewed preflight value. Run the candidate only when its preflight reports zero misses. Export the SQLite cache separately after every real run.

For the candidate two-case smoke, set `EXPERIMENT='candidate'`, restore the baseline cache, and use a separate candidate smoke directory. The expected approved budget is `0` only when preflight confirms zero misses for those two cases.

Do not change config, input, source, mock mode, or limit while resuming. Save manifests, metrics, validation, and API statistics. The organizer's official cumulative call count is not reset and may differ from run-local counts.

Compare runs only after both completed under compatible conditions:

```bash
python scripts/compare_runs.py \
  outputs/public_baseline_full \
  outputs/public_candidate_full
```

## 5. Private run

`Needs confirmation`: verify the organizer's refreshed Private Test schema and deadline before execution.

```bash
python scripts/run_private.py \
  --config configs/candidate.yaml \
  --input data/private/private_test.json \
  --resume-run submissions/private_candidate_full \
  --cache-db cache/private_case_api.sqlite \
  --max-network-calls APPROVED_PRIVATE_BUDGET
```

Validate the completed candidate:

```bash
python scripts/validate_submission.py \
  --input submissions/private_candidate_full/submission.json \
  --test-data data/private/private_test.json
```

Check complete coverage, official labels, exact API-returned case evidence, valid corpus `{law_id, aid}` pairs, no extra fields, and strict JSON.

## 6. Manual leaderboard submission

1. Confirm the candidate is a full non-mock run.
2. Confirm validation passes under the refreshed rules.
3. Review manifest, config, Git revision, and API statistics.
4. Confirm the team is within 20 submissions per 24 hours.
5. The submission owner manually enters the team name/token and uploads `submission.json`.
6. Record official metrics in `experiments.md` without storing credentials.

The leaderboard displays the best run per team. Source code must never upload automatically.

## 7. Source bundle

```bash
pytest
git diff --check
python scripts/package_source.py
```

Inspect the archive for README, config, source, scripts, notebooks, tests, and docs. It must exclude `.env`, raw/private data, caches, weights, logs, outputs, submissions, and tokens.

`Needs confirmation`: the refreshed official pages do not currently state source-code or technical-report delivery requirements or deadlines.

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
