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

The live baseline/candidate policy uses exactly two queries per case (`court_decision` and normalized `case_query`). Every live run requires an explicit network-attempt cap. Public development defaults to `cache-only` with zero network attempts. A preflight report must fit inside the live cap, and Colab keeps a local working database with fail-closed external backup.

Rejected alternatives:

- repeating calls to reproduce an already cached result;
- broad query sweeps on Public Test;
- treating each run as having an independent call budget.

## D-004: Manual leaderboard submission only

**Status:** Accepted

The repository generates and validates `submission.json` but never uploads it. The submission owner reviews the run artifacts and performs the official upload manually.

## D-005: Staged T4 execution

**Status:** Accepted

Retrieval models prepare contexts and are released before Qwen3 loads. Every full run must first pass a two-case smoke run in a separate directory.

## D-006: Team model stack

**Status:** Accepted and aligned with current official restrictions

- Outcome: pinned `Qwen/Qwen3-8B`, NF4 4-bit, FP16 compute, thinking disabled.
- Law retrieval baseline: BM25.
- Law retrieval candidate: `AITeamVN/Vietnamese_Embedding` + BM25 + RRF + exact citation candidate expansion + `AITeamVN/Vietnamese_Reranker`.
- Outcome candidate: decision-first prompt with larger case-evidence excerpts and the official `>50%` partial-label boundary.

The official competition website allows only open-weight models with fewer than 10 billion parameters, prohibits proprietary/non-open model APIs, and prohibits externally annotated legal QA or legal entailment datasets.

The pinned Hugging Face revisions were rechecked on 2026-07-20. All three repositories are public, ungated, Apache-2.0, and report safetensors parameter totals below 10 billion: Qwen3-8B `8,190,735,360`, Vietnamese Embedding `567,754,752`, and Vietnamese Reranker `567,755,777`. The code loads these weights locally rather than calling a hosted model API. Any added model or dataset must be checked and documented before use.

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

## D-010: Conservative submission-limit policy

**Status:** Accepted

The official competition website states a maximum of three submissions per day, while the leaderboard rules state 20 submissions per team per 24 hours. This conflict is `Needs confirmation`.

Until written organizer clarification is recorded, the team follows the stricter operational limit of at most three submissions in any 24-hour period. Private Test additionally permits only three distinctly named runs in total, and the best run counts. Upload remains manual under D-004.

## D-011: Canonical local Private Test contract

**Status:** Accepted and `CPU/mock verified`

The canonical local Private input is `data/raw/ALQAC_private_test.json`. Its observed contract is 60 unique cases containing exactly non-empty string `case_id` and `case_query` fields. The file is raw organizer data: it is immutable, git-ignored, excluded from source bundles, and never used as a source of gold inference features.

## D-012: Drive-first Colab and zero-call Public development

**Status:** Accepted and `CPU/mock verified`

`notebooks/colab_rag.ipynb` is the canonical Public/Private orchestration entry point. It requires all three Colab Secrets up front, clones the latest `TuanAnh` branch, records the resolved commit, keeps source execution thin, restores fingerprinted index/cache artifacts from Google Drive, runs a model-only gate before exporting the organizer token to the runner, separates smoke/full directories, and exports only allowlisted validated files.

Public runs use the real embedding/reranker/Qwen stack in `cache-only` mode and may not create new Case Content API calls. Private live success records and the attempt ledger share one transaction; the next request is forbidden until the SQLite state is backed up successfully. LoRA loading is supported as an optional interface, but Public gold is not enabled as fine-tuning data without a future organizer-confirmed contract change.
