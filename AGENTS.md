# ALQAC 2026 Agent Rules

## Communication and language

- Communicate with the user in Vietnamese.
- Write repository documentation, plans, code comments, module descriptions, API notes, submission rules, and experiment logs in English.
- Preserve official field names, labels, filenames, API names, and important Vietnamese legal phrases exactly.

## Sources of truth

Use this order: current official organizer pages, organizer-provided raw data, tracked code/configuration, reproducible run artifacts, then repository notes. Report conflicts as `Needs confirmation`; never guess.

Current official organizer pages include both the ALQAC 2026 website at `https://sites.google.com/view/alqac2026` and the leaderboard pages at `https://alqac2026-leaderboard.ngrok.app/`. If those pages disagree, document both statements and follow the stricter rule until the organizers clarify it.

Read only the relevant canonical documents:

- `docs/competition.md` — official requirements and open questions.
- `docs/data_api.md` — data/API contract, token, rate limit, and call accounting.
- `docs/submission.md` — schema and validation checklist.
- `docs/pipeline.md` — current implementation.
- `docs/plan.md` — priorities and status.
- `docs/experiments.md` and `docs/decisions.md` — evidence and team decisions.

## Data boundary

- Production inference may consume only `case_id` and `case_query`.
- Public-only gold fields may be read by offline training, target-construction,
  evaluation, and error-analysis code. They must never enter production inference.
- Never use `verdict_label`, `case_fact`, `judgment_text`, `court_reasoning`, `court_verdict`, or `related_law_provisions` as inference features.
- The Private Test contract is `data/raw/ALQAC_private_test.json`; it must contain only `case_id` and `case_query`.
- `data/raw/ALQAC_private_test.json` and
  `data/raw/private_test_60_cases_extracted_corpus.json` are immutable organizer
  data tracked only in the verified private GitHub repository. They must remain
  excluded from source bundles, run artifacts, and exports. Stop before push if
  the repository is no longer private.

## API and submission safety

- Case Content API calls accumulate for each `case_id` across team runs and affect official scoring. Public and Private identifiers do not overlap in the reviewed files, but whether the organizer applies any additional cross-track accounting is `Needs confirmation`. Prefer cache/checkpoint reuse and do not make exploratory real calls without a reviewed budget.
- Production baseline/candidate runs use the same structured query-planning
  policy: two primary queries and at most one adaptive third query when the
  evidence-sufficiency gate fails.
- Every live run requires API preflight, an explicit hard network-attempt cap, and the shared external SQLite cache. The canonical Public quality workflow uses live retrieval only after explicit approval; low-level `cache-only` remains available for zero-call diagnostics.
- Before any live run, complete the model-only runtime check. A successful API response must be committed with its ledger row in one SQLite transaction and backed up to external storage before the next request.
- Treat `chunk_id` as opaque and preserve the exact API value; never construct or validate a guessed prefix.
- Use only open-weight models with fewer than 10 billion parameters. Proprietary models and non-open model APIs are prohibited.
- Do not use externally annotated datasets created for legal question answering or legal entailment. Online legal-database lookup is permitted by the official rules, but every external source must be documented and must not cross the inference-data boundary.
- Read tokens only from environment or approved secret storage. Never hardcode, print, log, commit, or serialize secrets.
- Never modify raw official data. Before pushing tracked Private data, verify
  the GitHub repository visibility is private.
- Never upload a leaderboard submission automatically.
- Do not upload mock, partial, failed, or unvalidated output.
- A Colab `RUN_ID` pins one exact Git commit during `smoke`. Its `smoke` and
  `full` stages must execute that same commit; internal checkpoint resume is
  automatic. Use a new `RUN_ID` to adopt newer branch code.
- Smoke and full must use the same `workflow_config.json` and configuration
  fingerprint. They may differ only in case count, stage/run directory,
  case-count-derived network cap, and smoke/full gate state.
- The Case API hard cap is `planned_cases × max_network_attempts_per_case`.
  Retry attempts and semantic queries share the same per-case cap, which is
  restored from the SQLite ledger on resume.
- Primary queries receive one attempt each before the remaining per-case
  attempt is allocated to one retry or adaptive query 3. A case-scoped
  timeout, `429`, `5xx`, malformed query, or exhausted per-case budget must
  degrade to cached/partial case evidence rather than fail the whole run.
  Authentication, cache-integrity, backup, and model-load failures remain
  fail-fast.
- The configured LLM planner is pinned `Qwen/Qwen3-8B` in NF4 4-bit mode and
  runs in a separate stage. Planner fallback and prediction fallback counts
  must be recorded in internal run artifacts. Smoke may not pass with a
  degraded case; full may finish and validate a complete submission with
  explicitly reported degraded cases for manual review.
- The official submission limits conflict: the competition website says three submissions per day, while the leaderboard rules say 20 per team per 24 hours. Treat this as `Needs confirmation` and enforce the stricter team limit of at most three submissions in any 24-hour period.
- Private Test permits at most three distinctly named runs in total; run names cannot be reused and the best run counts.

## Working rules

- Inspect `git status` before editing and preserve unrelated user changes.
- Do not modify source code when the task requests only documentation or analysis.
- Keep notebooks thin; reusable logic belongs in `src/alqac2026`.
- Use `notebooks/colab_public.ipynb` for Public smoke/evaluation and
  `notebooks/colab_private.ipynb` for Private smoke/submission generation. The
  older Kaggle notebooks are historical entry points, not canonical Colab runners.
- Do not commit, push, submit, or change external state unless explicitly requested.
- Keep changes small, consistent, reviewable, and easy to revert.

Use these verification statuses exactly:

- `implemented`;
- `CPU/mock verified`;
- `GPU/API verified`;
- `leaderboard verified`; and
- `Not implemented yet`.

Never promote a status without reproducible evidence.

## Required code checks

Before claiming a code change complete, run:

```bash
python -m pip check
pytest
git diff --check
rm -rf outputs/smoke
python scripts/run_public.py --mock --limit 2 \
  --config configs/baseline.yaml --resume-run outputs/smoke
python scripts/package_source.py
```

Before a full Public or Private GPU run, run exactly two cases in a new run directory. Never resume a mock or limited run as a full production run.

## Definition of done

- Requested files are updated and cross-references are consistent.
- Official requirements, team implementation, and open questions are separated.
- Uncertain claims are marked `Needs confirmation`.
- No secrets or raw-data modifications are present.
- Relevant checks pass.
- The final user report is in Vietnamese and states changes, verification, remaining uncertainty, and next priorities.
