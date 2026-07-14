# Technical Decisions

This log distinguishes team decisions from official ALQAC 2026 requirements. Official rules live in `docs/competition.md`.

## D-001: Private-like inference boundary

**Status:** Accepted

Production inference receives only `case_id` and `case_query`. Public-only annotations are isolated in the evaluator. This prevents leakage and keeps the Public pipeline compatible with the expected Private workflow.

## D-002: Treat `chunk_id` as opaque

**Status:** Accepted and `CPU/mock verified`

The exact string returned by the Case Content API is the only valid identifier source. The system must not construct or validate `_chunk_`/`_seg_` patterns. This decision is robust to the current inconsistency between official examples and the leaderboard announcement.

## D-003: Shared permanent API-call budget

**Status:** Accepted and `CPU/mock verified`

Official call counts accumulate across all runs and affect Penalized Case Recall. Real API calls require a reviewed purpose, successful responses must be cached, and experiments should reuse a shared evidence registry whenever possible.

The production baseline/candidate policy uses exactly two queries per case (`court_decision` and normalized `case_query`). Every non-mock run requires an explicit network-attempt cap. A preflight report must fit inside that cap, and Kaggle imports/exports the SQLite cache separately from submission artifacts.

Rejected alternatives:

- repeating calls to reproduce an already cached result;
- broad query sweeps on Public Test;
- treating each run as having an independent call budget.

## D-004: Manual leaderboard submission only

**Status:** Accepted

The repository generates and validates `submission.json` but never uploads it. The submission owner reviews the run artifacts and performs the official upload manually.

## D-005: Staged Kaggle T4 execution

**Status:** Accepted

Retrieval models prepare contexts and are released before Qwen3 loads. Every full run must first pass a two-case smoke run in a separate directory.

## D-006: Team model stack

**Status:** Accepted team implementation; not an official restriction

- Outcome: pinned `Qwen/Qwen3-8B`, NF4 4-bit, FP16 compute, thinking disabled.
- Law retrieval baseline: BM25.
- Law retrieval candidate: `AITeamVN/Vietnamese_Embedding` + BM25 + RRF + exact citation candidate expansion + `AITeamVN/Vietnamese_Reranker`.
- Outcome candidate: decision-first prompt with larger case-evidence excerpts and the documented Team heuristic for partial labels.

The refreshed official pages do not currently confirm model-size, open-weight, proprietary-API, or external-data restrictions. The team keeps the current open-weight stack while seeking confirmation.

## D-007: Evidence-backed verification language

**Status:** Accepted

Use only:

- `implemented`;
- `CPU/mock verified`;
- `GPU/API verified`;
- `leaderboard verified`; and
- `Not implemented yet`.

A status may be promoted only when a reproducible artifact supports it.

## D-008: Secret handling

**Status:** Accepted

Tokens are read only from environment variables or approved notebook secret storage. They are never hardcoded, printed, committed, serialized into artifacts, or included in submission files.

## D-009: Canonical documentation layout

**Status:** Accepted

- `competition.md`: official requirements and open questions.
- `data_api.md`: data/API contract and call-budget rules.
- `submission.md`: official output and validation checklist.
- `pipeline.md`: current repository implementation.
- `plan.md`: prioritized work.
- `experiments.md`: reproducible results only.

Legacy names remain as compatibility pointers and must not become competing sources of truth.
