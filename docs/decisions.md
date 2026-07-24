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

The live baseline/candidate policy uses two primary structured queries and at
most one adaptive third query after an evidence-gate failure. Every live run
uses a three-attempt per-case cap shared by semantic queries and the single
allowed retry. The global hard cap is `planned_cases × 3`; cache hits spend no
attempt. Public quality evaluation may use live retrieval only after explicit
approval; low-level `cache-only` remains available for zero-call diagnostics.
SQLite restores per-case counts on client recreation, and Colab keeps a local
working database with fail-closed external backup.

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

The canonical Private input is `data/raw/ALQAC_private_test.json`. Its observed
contract is 60 unique cases containing exactly non-empty string `case_id` and
`case_query` fields. The input and Private law corpus are immutable organizer
data tracked only in the verified private repository. They remain excluded from
source bundles and exports and are never used as gold inference features.

## D-012: Split Drive-first Colab smoke/full workflows

**Status:** Accepted and `CPU/mock verified`

`notebooks/colab_public.ipynb` is the canonical Public evaluation entry point and
`notebooks/colab_private.ipynb` is the canonical Private submission entry point.
Both expose only `smoke` and `full`. Smoke pins an exact Git commit and
`workflow_config.json` fingerprint, runs the zero-API planner/model gate, and
then runs two live cases. Full requires the passing smoke gate on that exact
commit/config, reuses the shared query-plan/cache artifacts, and automatically
resumes only its own checkpoint directory. Stage differences are restricted to
case count, stage directory, case-count-derived network cap, and gate state. A
new `RUN_ID` is required to adopt newer code or configuration.

Public live retrieval requires explicit approval and produces local outcome/law
evaluation. Private uses the organizer-provided Private law corpus and the
Public full run's selected law top-k scalar, but never reads Public gold.
Private input and corpus are loaded from the source-pinned private checkout.
Colab restores Drive model snapshots to local storage without copying duplicate
Hugging Face blob objects.
Successful live responses, the attempt ledger, and pending-backup state share
one transaction; a resumed client repairs pending backup before any cache hit or
request. Validation and manifest bind the exact submission hash and byte length
before export. LoRA loading remains optional through `adapter_path`.

## D-013: Interpret API penalty per exact case identifier

**Status:** Accepted with `Needs confirmation`

The official formula defines `c_i` for each case, and the reviewed Public and
Private inputs have zero overlapping `case_id` values. The working
interpretation is therefore that Public calls do not directly increase a
Private case's `c_i`, even though logs remain append-only across all runs.
Whether the organizer applies any additional team-wide cross-track accounting
is not explicitly documented and requires organizer confirmation. Public live
runs remain budgeted and cached because they still affect Public scoring and
consume shared operational capacity.

## D-014: LLM-assisted structured query planning with safe fallback

**Status:** Accepted and `CPU/mock verified`

The production planner first uses pinned open-weight `Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca` with thinking off, deterministic
generation, a 256-token output bound, and deadline-aware stopping. Output must
match the structured schema, preserve query-grounded spans/terms, include
required claim/remedy/object information, and avoid outcome inference.

Any load, timeout, generation, JSON, grounding, or runtime failure selects the
deterministic structured planner and does not fail the case. The planner runs
as a separate stage and releases GPU resources before law retrieval/outcome
models. Query plans are persisted under a versioned fingerprint and excluded
from safe logs, submissions, and exports.

## D-015: Public gold for offline training, never inference

**Status:** Accepted

Public Test gold may construct offline SFT/QLoRA targets and support evaluation
or error analysis. Production-equivalent training input contains only
`case_query`, cached Case API evidence, and retrieved law evidence; gold
verdict, court text, or derived target fields never enter Private inference.

Model selection uses case-grouped stratified five-fold out-of-fold evaluation,
with all augmentations of one `case_id` in one fold. Reporting training-set
accuracy after fitting all 50 cases is rejected. After hyperparameters are
locked from out-of-fold evidence, a final adapter may be trained on all Public
cases and loaded only through the existing `adapter_path` interface.
