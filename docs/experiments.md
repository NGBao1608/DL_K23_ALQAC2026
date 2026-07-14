# Experiment Registry

**Last updated:** 2026-07-14

Only reproducible artifacts may be recorded as results. Every real API run must include cumulative-call risk in its notes, even though the repository can observe only run-local network calls and the organizer owns the official cumulative count.

## Current results

| Run | Config | Environment | Status | Outcome accuracy | Law F1 | Notes |
|---|---|---|---|---:|---:|---|
| CPU mock | `baseline.yaml` | Local CPU | CPU/mock verified | N/A | N/A | Plumbing only; mock metrics are not model results. |
| BM25 retrieval | `baseline.yaml` | Local CPU | CPU/mock verified | N/A | N/A | 50 cases: Recall@5 `0.02373`, Recall@10 `0.03756`; artifact `outputs/retrieval_bm25.json`. |
| Citation-aware hybrid candidate | `candidate.yaml` | Kaggle T4 | implemented | Pending | Pending | Exact citations expand the reranker pool; requires a new two-case smoke run. |
| Decision-first Qwen3 candidate | `candidate.yaml` | Kaggle T4 | implemented | Pending | N/A | Uses larger decision evidence excerpts; no clean GPU/API artifact is recorded. |

## Official format revision

The leaderboard announcement observed on 2026-07-14 changed/clarified evidence submission:

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
