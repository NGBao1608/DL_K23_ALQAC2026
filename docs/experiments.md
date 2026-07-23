# Experiment Registry

**Last updated:** 2026-07-23

Only reproducible artifacts may be recorded as results. Every real API run must include cumulative-call risk in its notes, even though the repository can observe only run-local network calls and the organizer owns the official cumulative count.

## Current results

| Run | Config | Environment | Status | Outcome accuracy | Law F1 | Notes |
|---|---|---|---|---:|---:|---|
| CPU mock | `baseline.yaml` | Local CPU | CPU/mock verified | N/A | N/A | Plumbing only; mock metrics are not model results. |
| BM25 retrieval | `baseline.yaml` | Local CPU | CPU/mock verified | N/A | N/A | 50 cases: Recall@5 `0.02373`, Recall@10 `0.03756`; artifact `outputs/retrieval_bm25.json`. |
| Citation-aware hybrid candidate | `candidate.yaml` | Kaggle T4 | implemented | Not measured | Not measured | Exact citations expand the reranker pool; requires a new two-case smoke run. |
| Decision-first Qwen3 candidate | `candidate.yaml` | Kaggle T4 | implemented | Not measured | N/A | Uses the official partial-label boundary; no clean `GPU/API verified` artifact is recorded. |
| Split Drive-first Colab candidate path | `candidate.yaml` | Local tests only | CPU/mock verified | Not measured | Not measured | Public/Private smoke-full orchestration, source pinning, automatic model gate, Private corpus validation, token budgeting, resume-safe pending SQLite backup, submission-hash binding, notebook syntax, and safe export are covered; no live T4 result is claimed. |
| Candidate smoke artifact `334997098` | `candidate.yaml` | Kaggle T4 | implemented | N/A | N/A | Commit `495f178eafe5232ded1be4487f94c5360836be5c`; two Case API attempts returned HTTP 200 and produced two cache rows, then execution stopped during `AITeamVN/Vietnamese_Embedding` download before context preparation or prediction completed. |

The incomplete candidate smoke is diagnostic evidence only. Its manifest remains `running` with zero completed cases, so it does not promote the candidate to `GPU/API verified` and must not be submitted.

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
