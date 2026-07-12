# Experiment registry

Only reproducible artifacts may be recorded as results. Use the verification levels defined in `AGENTS.md`.

| Run | Config | Environment | Status | Outcome accuracy | Law F1 | Notes |
|---|---|---|---|---:|---:|---|
| CPU mock | `baseline.yaml` | Local CPU | CPU/mock verified | N/A | N/A | Validates plumbing only; mock metrics are not model results |
| BM25 retrieval | `baseline.yaml` | Local CPU | CPU/mock verified | N/A | N/A | 50 cases: Recall@5=0.02373, Recall@10=0.03756; artifact `outputs/retrieval_bm25.json` |
| Hybrid candidate | `candidate.yaml` | Kaggle T4 | implemented | Pending | Pending | Must pass 2-case smoke before full run |
| Qwen3 candidate | `candidate.yaml` | Kaggle T4 | implemented | Pending | N/A | Qwen3/Vietnamese models/API not yet GPU/API verified |

## Required comparison

```bash
python scripts/evaluate_retrieval.py \
  --config configs/baseline.yaml \
  --output outputs/retrieval_bm25.json

python scripts/evaluate_retrieval.py \
  --config configs/candidate.yaml \
  --output outputs/retrieval_hybrid.json

python scripts/compare_runs.py outputs/public_baseline outputs/public_candidate_full
```

Promote `candidate.yaml` to `final.yaml` only after a clean full public run supports the choice and the result is recorded in this table.
