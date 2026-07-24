# Experiment Registry

**Last updated:** 2026-07-24

Only reproducible artifacts may be recorded as results. Every real API run must include cumulative-call risk in its notes, even though the repository can observe only run-local network calls and the organizer owns the official cumulative count.

## Current results

| Run | Config | Environment | Status | Outcome accuracy | Law F1 | Notes |
|---|---|---|---|---:|---:|---|
| CPU mock | `baseline.yaml` | Local CPU | CPU/mock verified | N/A | N/A | Plumbing only; mock metrics are not model results. |
| BM25 retrieval | `baseline.yaml` | Local CPU | CPU/mock verified | N/A | N/A | 50 cases: Recall@5 `0.02373`, Recall@10 `0.03756`; artifact `outputs/retrieval_bm25.json`. |
| Structured Case Retrieval plumbing | shared baseline/candidate retrieval config | Local CPU/mock | CPU/mock verified | N/A | N/A | 50 Public inputs deterministically produced 150 candidate queries; observed length range 8–25 whitespace tokens. Unit tests cover LLM success/fallback classes, grounded validation, evidence gate, adaptive query 3, retry/case caps, resume, cache-only, safe logs, and artifact exclusion. No scored API call was made. |
| Citation-aware hybrid candidate | `candidate.yaml` | Kaggle T4 | implemented | Not measured | Not measured | Exact citations expand the reranker pool; requires a new two-case smoke run. |
| Decision-first Qwen3 candidate | `candidate.yaml` | Kaggle T4 | implemented | Not measured | N/A | Uses the official partial-label boundary. Bounded prepared-context re-prediction is `CPU/mock verified`: three retries after the initial attempt, no repeated retrieval, and deterministic fallback only after exhaustion. No clean `GPU/API verified` artifact is recorded. |
| Split Drive-first Colab candidate path | `candidate.yaml` | Local tests only | CPU/mock verified | Not measured | Not measured | Public/Private smoke-full orchestration, source pinning, automatic model gate, Private corpus validation, token budgeting, resume-safe pending SQLite backup, submission-hash binding, notebook syntax, and safe export are covered; no live T4 result is claimed. |
| Candidate smoke artifact `334997098` | `candidate.yaml` | Kaggle T4 | implemented | N/A | N/A | Commit `495f178eafe5232ded1be4487f94c5360836be5c`; two Case API attempts returned HTTP 200 and produced two cache rows, then execution stopped during `AITeamVN/Vietnamese_Embedding` download before context preparation or prediction completed. |
| Public candidate diagnostic `public-candidate-v4` | `candidate.yaml` | Colab T4 + live API | implemented | Not measured | Not measured | Commit `ec1675edaebc57998105db64f8a62724c6dbe324`; smoke completed, but full stopped before prediction. Query plans used the LLM for 2/50 cases and deterministic fallback for 48/50, all recorded as `ValueError`. Full recorded 22 attempts, 20 successful calls, 6 cache hits, and nine prepared contexts; `case_8219` consumed 3 attempts with one success before an inline retry raised `ApiBudgetExceeded`. Diagnostic only; not `GPU/API verified` and not submit-ready. |
| Public candidate diagnostic `public-candidate-v5` | `candidate.yaml` | Colab T4 + live API | implemented | 1/2 | Not promoted | Commit `89da008156e42e5cec8e4f81bb03a6af0bda2e3d`; both cases completed and submission validation passed. The run used 6 attempts with 5 successful calls. `case_4337` received one `server_error`, then completed both primary logical queries through the bounded retry, but the pre-fix status logic retained that recovered error as `degraded_cases=1` and rejected the smoke gate. Diagnostic only; the classification fix remains `CPU/mock verified` until a new source-pinned smoke run. |
| Public candidate diagnostic `public-candidate-v6` | `candidate.yaml` | Colab T4 + cached official evidence | implemented | 2/2 | `0.28571` | Commit `178c81f4748467421ffa4caa7ad28d3b8979f21f`; both cases completed, submission validation passed, and all six retrievals were cache hits with zero network attempts. `case_4337` reached the 12-second LLM planner generation deadline and used the deterministic planner successfully. The pre-fix status policy counted that designed fallback as `degraded_cases=1` and rejected the smoke gate. Diagnostic only; not promoted to `GPU/API verified`. |

The incomplete candidate smoke is diagnostic evidence only. Its manifest remains `running` with zero completed cases, so it does not promote the candidate to `GPU/API verified` and must not be submitted.

The structured retrieval row is plumbing/proxy evidence only. Planner model
loading, deadline behavior on T4, real BM25 segment quality, and the
evidence-sufficiency gate against official API responses remain unverified on
GPU/API.

The `public-candidate-v4` artifact motivated prompt/composer v2, the quantized
8B planner candidate, classified planner failures, primary-first retry
allocation, and complete-case degradation. Those replacements are only
`CPU/mock verified` until a new two-case smoke uses a new `RUN_ID`.

The `public-candidate-v5` artifact showed that attempt history and final
degradation state must be separated. A bounded retry that succeeds is now
reported as a recovered retrieval error; an unresolved final failure still
degrades the case and blocks smoke.

The `public-candidate-v6` artifact showed the same distinction for structured
planning. The deterministic planner is the required recovery path for an LLM
planner timeout, so it remains observable under `planner_fallbacks` without
degrading an otherwise complete case.

## Official format revision

The leaderboard announcement rechecked on 2026-07-20 changed/clarified evidence submission:

- `law_evidence` is a list of `{law_id, aid}` objects;
- case evidence identifiers are announced as opaque hashed IDs; and
- Public Test runs should be resubmitted to refresh scores.

Any old leaderboard score must be treated as stale unless it was produced and submitted under the refreshed format. No refreshed result is currently `leaderboard verified` in this registry.

## Required run metadata

Every recorded real run must identify:

- experiment/config name;
- Git revision and dirty state;
- model names and pinned revisions;
- corpus SHA-256;
- environment/GPU;
- input and case count;
- run directory;
- run-local network calls and cache hits;
- validator result; and
- official leaderboard metrics, when submitted.

Never store tokens or request headers.

## Required comparison

```bash
python scripts/evaluate_retrieval.py \
  --config configs/baseline.yaml \
  --output outputs/retrieval_bm25.json

python scripts/evaluate_retrieval.py \
  --config configs/candidate.yaml \
  --output outputs/retrieval_hybrid.json

python scripts/compare_runs.py \
  outputs/public_baseline_full \
  outputs/public_candidate_full
```

Reuse the same cached case evidence for fair comparison and to avoid permanent extra API calls. Promote a final config only after a clean full run and official evidence support the choice.
