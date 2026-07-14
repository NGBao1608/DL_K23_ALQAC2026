# ALQAC 2026 Work Plan

**Last updated:** 2026-07-14

Official requirements and open questions are tracked in `docs/competition.md`. No task may be marked complete without the expected artifact or reproducible evidence.

## P0 – Required for a valid submission

| Task name | Purpose | Related files/modules | Expected output | Current status |
|---|---|---|---|---|
| Confirm refreshed identifier behavior | Resolve the conflict between sequential and opaque hashed `chunk_id` examples | Official API response, `docs/submission.md` | One redacted response example and organizer confirmation if possible | Needs confirmation |
| Make validator identifier-opaque | Stop rejecting official hashed evidence identifiers | `submission.py`, submission tests | Validator accepts exact API-returned IDs without guessing a prefix | CPU/mock verified |
| Update evidence fixtures | Prevent tests from encoding stale `_chunk_N` assumptions | Retrieval/submission tests | Fixtures cover opaque hashed IDs and malformed values | CPU/mock verified |
| Establish a shared API-call budget | Protect Penalized Case Recall because calls accumulate permanently | Case retrieval config, runbook, team evidence registry | Two-query policy, preflight, hard cap, local ledger, and external cache workflow | CPU/mock verified |
| Verify refreshed API with two cases | Confirm auth, rate limit, response schema, opaque IDs, cache, and resume | Public notebook, Case API client | Two-case artifact with safe API stats and no repeated successful calls | Pending |
| Verify Qwen baseline on Kaggle T4 | Produce a real two-case outcome artifact | `baseline.yaml`, public notebook | Completed two-case run, valid JSON outputs, no OOM | Pending |
| Complete Public baseline | Create a comparable 50-case BM25 + Qwen run | Public runner | Reproducible full Public run and validated candidate submission | Pending |
| Complete Public candidate | Evaluate hybrid law retrieval and reranking | `candidate.yaml`, public runner | Reproducible full Public candidate run | Pending |
| Validate refreshed submission | Prevent format rejection | Submission builder/validator | `validation.json` PASS under opaque-ID rules | CPU/mock verified; API/leaderboard pending |
| Submit manually and record score | Obtain official metric values | Official submission page, experiment registry | Leaderboard result tied to run manifest and Git revision | Pending |
| Run Private inference | Produce the scored final file when Private data is released | Private runner/notebook | Complete validated `submission.json` | Waiting for Private Test |

## P1 – Score improvement

| Task name | Purpose | Related files/modules | Expected output | Current status |
|---|---|---|---|---|
| Compare BM25 and hybrid retrieval | Measure whether dense retrieval, citation expansion, and reranking improve law evidence | Law retriever, comparison script | Same-input Recall@5/10, Micro Law F1, runtime comparison | Citation extraction CPU/mock verified; full hybrid pending |
| Optimize query allocation | Improve case recall without wasting permanent API calls | Query generator, cached evidence registry | Small reviewed query policy using existing cache first | Pending |
| Analyze outcome errors | Reduce confusion among four outcome labels | Public metrics/errors artifacts | Label confusion and case-level error categories | Pending real run |
| Tune outcome prompt safely | Improve accuracy without data leakage | Predictor/config | Baseline versus `decision_first_v2` comparison on private-like Public input | Candidate implemented; GPU comparison pending |
| Evaluate evidence selection | Improve grounding within the fixed context budget | Case/law evidence ranking | Ablation using cached evidence only | Pending |
| Compare candidate on leaderboard | Select final config using official score | Experiment registry | Best configuration chosen by accuracy, FinalScore, law F1, stability, runtime | Pending |

## P2 – Reproducibility and reporting

| Task name | Purpose | Related files/modules | Expected output | Current status |
|---|---|---|---|---|
| Clean-kernel replay | Prove setup is reproducible | Kaggle notebooks, requirements | Restart & Run All succeeds for winning config | Pending |
| Pin final environment | Freeze dependencies and model revisions | Requirements/config/manifest | Reproducible environment record | Model revisions pinned; dependencies range-pinned |
| Maintain experiment registry | Prevent unsupported score claims | `docs/experiments.md`, run manifests | Every result references config, Git revision, artifacts, and API count | In progress |
| Prepare source bundle | Deliver clean source without secrets/data/cache | Packaging script | Audited source archive | CPU/mock verified |
| Prepare technical report | Document task, system, experiments, and limitations | Technical report outline | Submission-ready report if required | Requirement needs confirmation |
| Confirm organizer deliverables | Resolve source/report deadlines and restrictions | Organizer communication | Written confirmation recorded in competition docs | Needs confirmation |

## Execution order

1. Review the two-query preflight and preserve the external SQLite cache.
2. Run a new two-case Public smoke test with an explicit four-attempt cap.
3. Run full baseline, then full candidate with the same cached case evidence and zero candidate misses.
4. Validate and manually submit both comparable runs.
5. Record official metrics and promote a final config only with evidence.
6. Reproduce from a clean kernel, package source, and complete required reporting.
