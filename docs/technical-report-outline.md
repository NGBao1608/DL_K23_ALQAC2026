# Technical Report Outline

**Last synchronized:** 2026-07-20

**Canonical references:** [`competition.md`](competition.md), [`pipeline.md`](pipeline.md), and [`experiments.md`](experiments.md)

The official competition website encourages a short technical report and says organizers may request source code, configuration files, or logs for verification and reproducibility. `Needs confirmation`: whether these artifacts become mandatory and what deadlines or transfer procedure apply.

Do not add scores unless a reproducible artifact and, where applicable, an official leaderboard result exist.

## 1. Task and official score

- Legal Case Outcome Prediction with Evidence Retrieval.
- Four official labels.
- Outcome Accuracy, Penalized Case Recall, and Micro Law Evidence F1.
- Permanent cross-run Case API call accounting.
- Official rules source and last-checked date.

## 2. Data

- Public development set statistics and field boundary.
- Official law corpus statistics.
- Private inference over 60 cases using only `case_id` and `case_query`.
- Separation of Public gold annotations from inference.
- Private file integrity, non-overlap, and raw-data protection.
- Official prohibition on externally annotated legal QA/entailment datasets.

## 3. System

- Deterministic Case Content API query strategy and permanent call budget.
- Opaque case-evidence identifier handling.
- BM25 and hybrid Vietnamese law retrieval.
- Qwen3-8B NF4 prediction and structured output validation.
- Open-weight, fewer-than-10B model compliance and absence of proprietary model APIs.
- Cache-only Public evaluation, atomic cache backup, checkpoint/resume, and staged Colab T4 execution.

## 4. Experiments

- BM25 baseline.
- Hybrid retrieval and reranking comparison.
- Outcome accuracy and law evidence metrics.
- Official Penalized Case Recall and FinalScore when available.
- API calls/cache hits and query-budget rationale.

Every result row must reference config, Git revision, model revisions, run directory, corpus hash, and verification status.

## 5. Error analysis

- Confusion among full/partial plaintiff/defendant outcomes.
- Missing or incorrect case evidence.
- Missing or incorrect law evidence.
- API budget inefficiency.
- JSON, runtime, and validator failures.

## 6. Reproducibility

- Environment and hardware.
- Public/private commands and notebook workflow.
- Secret-handling policy.
- Run artifacts and clean-kernel replay.
- Source bundle contents and exclusions.
- Known limitations and unresolved organizer questions.
- Conflicting submission limits and the stricter internal operating rule.
