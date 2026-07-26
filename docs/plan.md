# ALQAC 2026 Work Plan

**Last updated:** 2026-07-25

Official requirements and open questions are tracked in `docs/competition.md`. No task may be marked complete without the expected artifact or reproducible evidence.

## P0 – Required for a valid submission

| Task name | Purpose | Related files/modules | Expected output | Current status |
|---|---|---|---|---|
| Confirm refreshed identifier behavior | Resolve the conflict between sequential and opaque hashed `chunk_id` examples | Official API response, `docs/submission.md` | One redacted response example and organizer confirmation if possible | Needs confirmation |
| Make validator identifier-opaque | Stop rejecting official hashed evidence identifiers | `submission.py`, submission tests | Validator accepts exact API-returned IDs without guessing a prefix | CPU/mock verified |
| Update evidence fixtures | Prevent tests from encoding stale `_chunk_N` assumptions | Retrieval/submission tests | Fixtures cover opaque hashed IDs and malformed values | CPU/mock verified |
| Synchronize current official contracts | Record partial-label definitions, model/data restrictions, quota conflict, and Private submission flow | `AGENTS.md`, canonical docs, legacy views | Cross-referenced English documentation with unresolved conflicts marked `Needs confirmation` | implemented |
| Protect and validate Private Test input | Establish the exact local path, schema, integrity, and packaging boundary | `data/raw/ALQAC_private_test.json`, `.gitignore`, configs, data loader | 60 unique two-field cases; file ignored and config paths aligned | CPU/mock verified |
| Establish a shared API-call budget | Protect Penalized Case Recall because calls accumulate permanently | Case retrieval config, runbook, team evidence registry | Three-attempt per-case cap shared by retry/semantic queries, preflight, resume ledger, and external cache workflow | CPU/mock verified |
| Implement structured adaptive Case Retrieval | Improve operative-verdict evidence without sending the full query | `query_planning.py`, `case_retrieval.py`, configs/tests | Pinned LLM planner with deterministic fallback, two primary queries, sufficiency gate, optional query 3, reusable plan artifact | CPU/mock verified |
| Implement split Drive-first Colab workflows | Separate Public evaluation from Private submission while keeping reusable logic outside notebooks | `colab_public.ipynb`, `colab_private.ipynb`, `colab_workflow.py` | Two thin notebooks exposing only smoke/full with track-separated artifacts | implemented |
| Add zero-API runtime gate inside smoke | Prove embedding, reranker, index, and Qwen work before retrieval spend without another user-facing mode | `check_runtime.py`, candidate config, Colab helper | `runtime_check.json` with model stages and zero API attempts before two live cases | implemented |
| Add cache-only Public mode | Evaluate model/law pipeline without adding permanent organizer calls | Runner, cache client, Public notebook | 50-case capable path with cache misses converted to empty evidence and zero HTTP | CPU/mock verified |
| Validate and index the Private law corpus | Bind Private law evidence and index reuse to the organizer-provided corpus | Colab helper, Private notebook, private Git checkout | Reviewed hash, 14 laws, 2,820 articles, fingerprinted index | CPU/mock verified |
| Verify refreshed API with two cases | Confirm auth, rate limit, response schema, opaque IDs, cache, and resume | Public notebook, Case API client | Two-case artifact with safe API stats and no repeated successful calls | Not implemented yet |
| Verify memory-safe Qwen candidate on Colab T4 | Confirm the Private outcome profile avoids the Public v7 OOM failure mode | `candidate.yaml`, Private Colab artifacts | 60/60 completed cases, zero OOM, zero prediction fallback | GPU/API verified |
| Complete Public baseline | Create a comparable 50-case BM25 + Qwen run | Public runner | Reproducible full Public run and validated candidate submission | Not implemented yet |
| Complete Public candidate | Evaluate hybrid law retrieval and reranking | `candidate.yaml`, public runner | Reproducible 50-case Public candidate with metrics/errors/profile | GPU/API verified |
| Validate refreshed submission | Prevent format rejection | Submission builder/validator | `validation.json` PASS under opaque-ID rules | CPU/mock verified |
| Add bounded outcome re-prediction | Recover transient per-case generation failures without repeating retrieval | Prediction pipeline, checkpoints, candidate/baseline config | At most three retries using one prepared context, then deterministic fallback | CPU/mock verified |
| Submit manually and record score | Obtain official metric values | Official submission page, experiment registry | Leaderboard result tied to run manifest and Git revision | Not implemented yet |
| Run Private inference | Produce the scored final file from the 60-case Private input | Private runner/notebook | Complete validated `private-candidate-v2` 60-case artifact | GPU/API verified |

## P1 – Score improvement

| Task name | Purpose | Related files/modules | Expected output | Current status |
|---|---|---|---|---|
| Compare BM25 and hybrid retrieval | Measure whether dense retrieval, citation expansion, and reranking improve law evidence | Law retriever, comparison script | Same-input Recall@5/10, Micro Law F1, runtime comparison | Not implemented yet |
| Optimize query allocation | Improve case recall without wasting permanent API calls | Structured planner, evidence gate, cached evidence registry | Primary-first scheduler, adaptive query 3 only on gate failure, one shared retry, fail-soft case completion | CPU/mock verified |
| Analyze outcome errors | Reduce confusion among four outcome labels | Public/Private metrics, checkpoints, and case status | Full confusion matrix plus OOM, repair, verifier, evidence, and class-bias analysis | GPU/API verified |
| Tune outcome prompt safely | Improve accuracy without data leakage | Predictor/config/rescore | Evidence-grounded `decision_first_v3` plus zero-HTTP prepared-context rescore | CPU/mock verified |
| Validate cached rescore on Public | Prove quality before spending another Private slot | `candidate_rescore_v1.yaml`, Public v7 prepared contexts | 50-case cache-only accuracy/confusion, zero network, no verifier failure | Not implemented yet |
| Build Public-gold adapter training workflow | Improve four-label calibration after retrieval | Training notebook/module, `adapter_path`, experiment registry | Production-equivalent inputs, case-grouped stratified five-fold OOF metrics, final locked adapter | Not implemented yet |
| Evaluate evidence selection | Improve grounding within the fixed context budget | Token-aware context, law top-k profile | Public Micro Law F1 selection from top-k 3–10 without inference leakage | CPU/mock verified |
| Compare candidate on leaderboard | Select final config using official score | Experiment registry | Best configuration chosen by accuracy, FinalScore, law F1, stability, runtime | Not implemented yet |

## P2 – Reproducibility and reporting

| Task name | Purpose | Related files/modules | Expected output | Current status |
|---|---|---|---|---|
| Clean-kernel replay | Prove setup is reproducible | Colab notebook, Colab requirements | Restart & Run All succeeds on a fresh T4 runtime | Not implemented yet |
| Run Private candidate | Use `public-candidate-v7` law-top-k profile and the memory-safe Qwen3-8B candidate | Private Colab notebook, Drive artifacts | `private-candidate-v2` two-case smoke PASS, then 60 completed validated cases under the same Private `RUN_ID` | GPU/API verified |
| Reuse warm Colab runtime from smoke to full | Avoid unnecessary reclone, dependency install, and model snapshot restore | Public/Private notebooks, artifact restore | Exact-clean pinned checkout, bootstrap fingerprint, and missing-file-only snapshot restore | CPU/mock verified |
| Pin final environment | Freeze dependencies and model revisions | Requirements/config/manifest | Reproducible environment record | implemented |
| Maintain experiment registry | Prevent unsupported score claims | `docs/experiments.md`, run manifests | Every result references config, Git revision, artifacts, and API count | implemented |
| Prepare source bundle | Deliver clean source without secrets/data/cache | Packaging script | Audited source archive | CPU/mock verified |
| Prepare technical report | Document task, system, experiments, and limitations | Technical report outline | Submission-ready report if requested | Not implemented yet |
| Confirm organizer deliverables | Resolve source/report deadlines and restrictions | Organizer communication | Written confirmation recorded in competition docs | Needs confirmation |

## Execution order

1. Rescore all 50 Public v7 prepared contexts with
   `candidate_rescore_v1.yaml`; require zero network attempts, no verifier
   failure, and a complete accuracy/confusion comparison.
2. Lock the outcome profile only if Public improves materially over `0.46` and
   the gain is not caused by evaluating fitted targets on the same cases.
3. Rescore the immutable Private v2 prepared contexts under a new Private
   `RUN_ID`; do not change planner/retrieval and do not call the Case API.
4. Validate 60-case coverage, checksums, class distribution, repair/verifier
   counts, and degradation before considering manual submission 2.
5. Reserve submission 3 for a Public-validated, meaningfully different
   evidence-selection/OOF-calibration strategy; do not spend it on an
   unmeasured model or random prompt variant.
6. Record any manually observed official metric without claiming uploaded-file
   hash verification, then reproduce checks and package source.
