# Qwen3 Cached Scope Rescore v1

## Purpose

`qwen3_cached_rescore_scope_v1` is an isolated, outcome-only candidate. It
keeps the pinned `Qwen/Qwen3-8B` model and consumes complete immutable
`PreparedCase` artifacts from an earlier run. It does not execute query
planning, Case Content API retrieval, or law retrieval.

The candidate does not modify the source run. It copies the prepared-context
checkpoint into a new run directory and verifies the copy by SHA-256 and byte
length before inference. The new run uses a local empty SQLite database,
`execution_mode=cache-only`, and `max_network_calls=0`. It does not require or
read the Case API token.

## Candidate-only behavior

- Order unique case-evidence chunks using operative, court-role, scope,
  negative-source-role, lexical-overlap, and numeric-overlap signals.
- Keep case evidence ahead of law evidence in the prompt.
- Include at most one law passage in the normal outcome prompt and none after
  OOM compaction. Submission law evidence remains unchanged.
- Generate the compact `scope-enum-v1` schema and derive the official label in
  deterministic code.
- Run a targeted verifier only for unclear, untrusted, inconsistent,
  near-boundary, or conflicting outcomes.
- Record safe per-case token, EOS, peak CUDA-memory, repair, verification,
  scope, and source-role telemetry in `scope_diagnostics.json`.

The predictor factory override exists only inside the candidate process and is
restored in a `finally` block. The default pipeline and predictor remain
unchanged when this script is not invoked.

## Public evaluation first

Run on all 50 Public prepared contexts before any Private rescore:

```bash
python scripts/rescore_scope_candidate.py \
  --input data/raw/ALQAC2026_public_test.json \
  --public-gold data/raw/ALQAC2026_public_test.json \
  --prepared-contexts /content/drive/MyDrive/ALQAC2026/runs/public/public-candidate-v7/full/contexts.checkpoint.json \
  --run-dir /content/drive/MyDrive/ALQAC2026/runs/public/public-scope-rescore-v1/full
```

Require:

- 50 completed cases and `validation=PASS`;
- zero network attempts, OOMs, and prediction fallbacks;
- at least 48 first/repair outputs that are strict JSON;
- no unresolved verifier failure;
- at least `27/50` outcome accuracy;
- a paired net gain of at least four cases over `public-candidate-v7`; and
- no loss of `B_WIN` or `PARTIAL_B_WIN` recall.

## Private rescore

Only after the Public gate passes, use a new Private run directory:

```bash
python scripts/rescore_scope_candidate.py \
  --input data/raw/ALQAC_private_test.json \
  --corpus data/raw/private_test_60_cases_extracted_corpus.json \
  --selection-profile /content/drive/MyDrive/ALQAC2026/runs/public/public-candidate-v7/full/selection_profile.json \
  --prepared-contexts /content/drive/MyDrive/ALQAC2026/runs/private/private-candidate-v2/full/contexts.checkpoint.json \
  --run-dir /content/drive/MyDrive/ALQAC2026/runs/private/private-scope-rescore-v1/full
```

Never reuse or overwrite `private-candidate-v2`. The script never uploads a
submission. Manual upload remains the submission owner's responsibility after
full artifact and hash review.

