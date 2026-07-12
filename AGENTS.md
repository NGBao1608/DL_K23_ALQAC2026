# ALQAC 2026 Engineering Rules

## Communication

- Communicate with the user in Vietnamese.
- Keep code identifiers, commands, schemas, and metric names in English.

## Sources of truth

Use this priority order: official organizer material, official raw data, tracked code/configuration, reproducible run artifacts, then README/notebook notes. Report conflicts instead of guessing.

## Data boundary

- Production inference may consume only `case_id` and `case_query`.
- Public-only gold fields may be read only by evaluation and error-analysis code.
- Never use `verdict_label`, `case_fact`, `judgment_text`, `court_reasoning`, `court_verdict`, or `related_law_provisions` as inference features.
- Public evaluation must build private-like inputs before calling the inference pipeline.

## Competition constraints

- Use only open-weight models with fewer than 10 billion parameters.
- Do not use proprietary model APIs in the competition pipeline.
- Respect the official Case Content API rate limit.
- Do not upload leaderboard submissions automatically.

## Locked implementation stack

- Outcome model: `Qwen/Qwen3-8B`, pinned revision, NF4 4-bit, FP16 compute, thinking disabled.
- Law retrieval candidate: BM25 top 50 + `AITeamVN/Vietnamese_Embedding` top 50 + RRF top 30 + `AITeamVN/Vietnamese_Reranker` top 5.
- Case evidence: at most eight deterministic queries through the official Case Content API.
- Production runtime: Kaggle T4; local CPU is only for tests, validation, and BM25/mock runs.
- `configs/baseline.yaml` is CPU/mock verified for plumbing and BM25 retrieval only; its Qwen/API path is not yet GPU/API verified.
- `configs/candidate.yaml` is the hybrid candidate. Do not create or call a configuration `final` until a clean full public run and comparison support promotion.

## Security

- Read the team token from environment or notebook secret storage.
- Never hard-code or print the token.
- Do not commit `.env`, private test data, caches, model weights, logs, or submission artifacts.

## Working method

- Inspect `git status` before editing and preserve unrelated user changes.
- Keep notebooks thin; reusable behavior belongs in `src/alqac2026`.
- Add tests for data boundaries, identifiers, cache behavior, parsers, and submission validation.
- Report separately what was statically checked, locally tested, and not run because GPU/API access was unavailable.
- Do not commit, push, or submit unless the user explicitly requests it.

## Verification language

Use exactly these status categories in reports:

- `implemented`: code exists but may not have run in the target environment.
- `CPU/mock verified`: covered by local tests or mock execution only.
- `GPU/API verified`: executed successfully with the pinned models and official API.
- `leaderboard verified`: confirmed by an official leaderboard result.

Never promote an earlier category to a later one without reproducible evidence.

## Required checks

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

Before a full public/private GPU run, first run exactly two cases in a new run directory. Never resume a mock or limited run as a full production run.
