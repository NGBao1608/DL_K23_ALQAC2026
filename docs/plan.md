# ALQAC 2026 Work Plan

**Last updated:** 2026-07-23

Official requirements and open questions are tracked in `docs/competition.md`. No task may be marked complete without the expected artifact or reproducible evidence.

## P0 – Required for a valid submission

| Task name | Purpose | Related files/modules | Expected output | Current status |
|---|---|---|---|---|
| Confirm refreshed identifier behavior | Resolve the conflict between sequential and opaque hashed `chunk_id` examples | Official API response, `docs/submission.md` | One redacted response example and organizer confirmation if possible | Needs confirmation |
| Make validator identifier-opaque | Stop rejecting official hashed evidence identifiers | `submission.py`, submission tests | Validator accepts exact API-returned IDs without guessing a prefix | CPU/mock verified |
| Update evidence fixtures | Prevent tests from encoding stale `_chunk_N` assumptions | Retrieval/submission tests | Fixtures cover opaque hashed IDs and malformed values | CPU/mock verified |
| Synchronize current official contracts | Record partial-label definitions, model/data restrictions, quota conflict, and Private submission flow | `AGENTS.md`, canonical docs, legacy views | Cross-referenced English documentation with unresolved conflicts marked `Needs confirmation` | implemented |
| Protect and validate Private Test input | Establish the exact local path, schema, integrity, and packaging boundary | `data/raw/ALQAC_private_test.json`, `.gitignore`, configs, data loader | 60 unique two-field cases; file ignored and config paths aligned | CPU/mock verified |
| Establish a shared API-call budget | Protect Penalized Case Recall because calls accumulate permanently | Case retrieval config, runbook, team evidence registry | Two-query policy, preflight, hard cap, local ledger, and external cache workflow | CPU/mock verified |
| Implement split Drive-first Colab workflows | Separate Public evaluation from Private submission while keeping reusable logic outside notebooks | `colab_public.ipynb`, `colab_private.ipynb`, `colab_workflow.py` | Two thin notebooks exposing only smoke/full with track-separated artifacts | implemented |
| Add zero-API runtime gate inside smoke | Prove embedding, reranker, index, and Qwen work before retrieval spend without another user-facing mode | `check_runtime.py`, candidate config, Colab helper | `runtime_check.json` with model stages and zero API attempts before two live cases | implemented |
| Add cache-only Public mode | Evaluate model/law pipeline without adding permanent organizer calls | Runner, cache client, Public notebook | 50-case capable path with cache misses converted to empty evidence and zero HTTP | CPU/mock verified |
| Validate and index the Private law corpus | Bind Private law evidence and index reuse to the organizer-provided corpus | Colab helper, Private notebook, Drive input | Reviewed hash, 14 laws, 2,820 articles, fingerprinted index | CPU/mock verified |
| Verify refreshed API with two cases | Confirm auth, rate limit, response schema, opaque IDs, cache, and resume | Public notebook, Case API client | Two-case artifact with safe API stats and no repeated successful calls | Not implemented yet |
| Verify Qwen candidate on Colab T4 | Produce a real two-case outcome artifact after the model-only gate | `candidate.yaml`, Colab notebook | Completed two-case run, valid JSON outputs, no OOM | Not implemented yet |
| Complete Public baseline | Create a comparable 50-case BM25 + Qwen run | Public runner | Reproducible full Public run and validated candidate submission | Not implemented yet |
| Complete Public candidate | Evaluate hybrid law retrieval and reranking | `candidate.yaml`, public runner | Reproducible full Public candidate run | Not implemented yet |
| Validate refreshed submission | Prevent format rejection | Submission builder/validator | `validation.json` PASS under opaque-ID rules | CPU/mock verified |
| Submit manually and record score | Obtain official metric values | Official submission page, experiment registry | Leaderboard result tied to run manifest and Git revision | Not implemented yet |
| Run Private inference | Produce the scored final file from the 60-case Private input | Private runner/notebook | Complete validated `submission.json` and named-run record | Not implemented yet |

## P1 – Score improvement

| Task name | Purpose | Related files/modules | Expected output | Current status |
|---|---|---|---|---|
| Compare BM25 and hybrid retrieval | Measure whether dense retrieval, citation expansion, and reranking improve law evidence | Law retriever, comparison script | Same-input Recall@5/10, Micro Law F1, runtime comparison | Not implemented yet |
| Optimize query allocation | Improve case recall without wasting permanent API calls | Query generator, cached evidence registry | Two-query live policy, shared cache, and reviewed Public/Private budgets | CPU/mock verified |
| Analyze outcome errors | Reduce confusion among four outcome labels | Public metrics/errors artifacts | Label confusion and case-level error categories | Not implemented yet |
| Tune outcome prompt safely | Improve accuracy without data leakage | Predictor/config | Baseline versus `decision_first_v2` comparison on private-like Public input | Not implemented yet |
| Evaluate evidence selection | Improve grounding within the fixed context budget | Token-aware context, law top-k profile | Public Micro Law F1 selection from top-k 3–10 without inference leakage | CPU/mock verified |
| Compare candidate on leaderboard | Select final config using official score | Experiment registry | Best configuration chosen by accuracy, FinalScore, law F1, stability, runtime | Not implemented yet |

## P2 – Reproducibility and reporting

| Task name | Purpose | Related files/modules | Expected output | Current status |
|---|---|---|---|---|
| Clean-kernel replay | Prove setup is reproducible | Colab notebook, Colab requirements | Restart & Run All succeeds on a fresh T4 runtime | Not implemented yet |
| Pin final environment | Freeze dependencies and model revisions | Requirements/config/manifest | Reproducible environment record | implemented |
| Maintain experiment registry | Prevent unsupported score claims | `docs/experiments.md`, run manifests | Every result references config, Git revision, artifacts, and API count | implemented |
| Prepare source bundle | Deliver clean source without secrets/data/cache | Packaging script | Audited source archive | CPU/mock verified |
| Prepare technical report | Document task, system, experiments, and limitations | Technical report outline | Submission-ready report if requested | Not implemented yet |
| Confirm organizer deliverables | Resolve source/report deadlines and restrictions | Organizer communication | Written confirmation recorded in competition docs | Needs confirmation |

## Execution order

1. Run Public `smoke`; it pins the source, restores/builds the Public law index, runs the zero-call model gate, and completes two live cases with cap four.
2. Keep the Public `RUN_ID`, run Public `full`, and review Outcome Accuracy, Law Micro F1, error analysis, and `selection_profile.json`.
3. If Public quality is acceptable, place and validate both Private input files on Drive and choose a new Private `RUN_ID`.
4. Run Private `smoke`; require two completed live cases, valid output, and verified atomic external cache backup.
5. Keep the Private `RUN_ID`, run Private `full`, and allow automatic resume only within its separate `full/` directory.
6. Validate complete coverage and checksums, assign a distinct Private run name, and manually use **Check format**.
7. Record official metrics and promote a final config only with reproducible evidence.
8. Reproduce from a clean Colab kernel, package source, and complete the encouraged technical report.
