# Runbook

## 1. Pre-flight

```bash
git status --short --branch
python -m pip install -e .
python -m pip check
pytest
```

Confirm:

- The correct branch and run config.
- `ALQAC_TEAM_TOKEN` exists in secret storage and is not in the source.
- Public and law files are at the paths given in the config.
- The A100 GPU and Internet are enabled in the Colab runtime.

## 2. CPU smoke test

```bash
python scripts/run_public.py \
  --mock \
  --limit 2 \
  --config configs/ensemble.yaml \
  --resume-run outputs/smoke
```

Check that `outputs/smoke/validation.json` has `status: PASS`. A mock score is not a model result.

## 3. Colab two-case validation

The notebook defaults to `RUN_MODE='smoke'`, which corresponds to `limit=2`. Run it from a clean A100 runtime and check that:

- Model revisions download successfully.
- There is no CUDA OOM.
- The Case API returns results and the cache holds records.
- Every prediction parses JSON successfully.
- The validator reports PASS.

Once this succeeds, switch to `RUN_MODE='full'`. The notebook uses a different run directory automatically; the runner also refuses to resume if mock, limit, input, or config change.

## 4. Full public run

```bash
python scripts/run_public.py \
  --config configs/ensemble.yaml \
  --resume-run outputs/public_ensemble_full
```

`ensemble.yaml` uses the `bm25_only` law strategy, so no dense index build is required. For a config with the `hybrid_rerank` strategy (such as `candidate.yaml`), build the dense index first:

```bash
python scripts/build_law_index.py --config configs/candidate.yaml
```

Do not change the config while resuming. Store `metrics.json`, `manifest.json`, and the leaderboard score outside Git. `scripts/compare_runs.py` compares two run directories when evaluating config changes.

## 5. Private run

```bash
python scripts/run_private.py \
  --config configs/ensemble.yaml \
  --input data/private/private_test.json \
  --resume-run submissions/private_ensemble_full
```

If the session stops, rerun the same command. After it completes:

```bash
python scripts/validate_submission.py \
  --input submissions/private_ensemble_full/submission.json \
  --test-data data/private/private_test.json
```

Upload only when the validator reports PASS, the case count matches the private input, and the submission owner confirms the attempt.

## 6. Source bundle

```bash
pytest
git diff --check
python scripts/package_source.py
```

Open the zip and check that it contains the README, config, source, scripts, notebooks, tests, and docs, and that it contains no `.env`, data, cache, weights, logs, outputs, or submissions.

## Failure handling

| Failure | Action |
|---|---|
| `403` API | The client sends `ngrok-skip-browser-warning` to bypass the interstitial and reports `ngrok_codes` if the gateway still blocks it. If the response is JSON from the Case API, confirm the secret has no stray quotes or prefix and that the token has been activated by the organizers. The client trims whitespace but does not retry 403. |
| `422` API | Check the case ID and query schema. |
| `429/5xx` API | The client backs off automatically; resume with the same run and cache. |
| CUDA OOM during retrieval | Reduce the reranker batch size, build contexts first, then release models. |
| CUDA OOM during Qwen | Confirm NF4 4-bit, a 7,000-token context, and that no retrieval models remain on the GPU. |
| Model output invalid | The pipeline repairs once; inspect the failed record and do not fix the label manually. |
| Config mismatch on resume | Use a new run directory or restore the exact previous config. |
| Validator FAIL | Do not upload; fix the reported errors and rerun the validator. |
