# ALQAC 2026 Agent Rules

## Communication and language

- Communicate with the user in Vietnamese.
- Write repository documentation, plans, code comments, module descriptions, API notes, submission rules, and experiment logs in English.
- Preserve official field names, labels, filenames, API names, and important Vietnamese legal phrases exactly.

## Sources of truth

Use this order: current official organizer pages, organizer-provided raw data, tracked code/configuration, reproducible run artifacts, then repository notes. Report conflicts as `Needs confirmation`; never guess.

Read only the relevant canonical documents:

- `docs/competition.md` — official requirements and open questions.
- `docs/data_api.md` — data/API contract, token, rate limit, and call accounting.
- `docs/submission.md` — schema and validation checklist.
- `docs/pipeline.md` — current implementation.
- `docs/plan.md` — priorities and status.
- `docs/experiments.md` and `docs/decisions.md` — evidence and team decisions.

## Data boundary

- Production inference may consume only `case_id` and `case_query`.
- Public-only gold fields may be read only by evaluation/error-analysis code.
- Never use `verdict_label`, `case_fact`, `judgment_text`, `court_reasoning`, `court_verdict`, or `related_law_provisions` as inference features.

## API and submission safety

- Case Content API calls accumulate across every team run and affect official scoring. Prefer cache/checkpoint reuse and do not make exploratory real calls without a reviewed budget.
- Production baseline/candidate runs use exactly two deterministic queries per case: `court_decision` and normalized `case_query`. Queries 3–8 require separate budget approval.
- Every non-mock run requires API preflight, an explicit hard network-attempt cap, and the shared external SQLite cache.
- Treat `chunk_id` as opaque and preserve the exact API value; never construct or validate a guessed prefix.
- Read tokens only from environment or approved secret storage. Never hardcode, print, log, commit, or serialize secrets.
- Never modify raw official data.
- Never upload a leaderboard submission automatically.
- Do not upload mock, partial, failed, or unvalidated output.

## Working rules

- Inspect `git status` before editing and preserve unrelated user changes.
- Do not modify source code when the task requests only documentation or analysis.
- Keep notebooks thin; reusable logic belongs in `src/alqac2026`.
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
