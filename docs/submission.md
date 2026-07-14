# Submission Specification and Checklist

**Last checked:** 2026-07-14

**Sources:** [task rules](https://alqac2026-leaderboard.ngrok.app/about), [submission page](https://alqac2026-leaderboard.ngrok.app/submit), and the leaderboard update announcement.

## Required file

```text
submission.json
```

The file must be valid UTF-8 JSON, no larger than 10 MB, with a JSON array at the root and exactly one object per test case.

## Required schema

```json
[
  {
    "case_id": "case_4101",
    "prediction": "A_WIN",
    "case_evidence": ["opaque_identifier_returned_by_the_api"],
    "law_evidence": [
      {
        "law_id": "47/2010/QH12",
        "aid": 270
      }
    ]
  }
]
```

No additional fields are allowed by the repository validator.

## Field rules

### `case_id`

- Must exist in the test set.
- Must appear exactly once.
- The submission must cover every test case.

### `prediction`

Must be one of:

```text
A_WIN
B_WIN
PARTIAL_A_WIN
PARTIAL_B_WIN
```

### `case_evidence`

- Required and must be a JSON list of strings.
- May be empty.
- Each value must be an exact `chunk_id` returned by the Case Content API.
- Identifiers are static across runs and may be reused after earlier retrieval.
- Do not construct, guess, normalize, or rename identifiers.
- Avoid duplicates; the organizer de-duplicates them before scoring.

`Needs confirmation`: current official examples conflict between sequential `..._chunk_N` and opaque hashed `..._seg_<hash>` forms. Treat the value as opaque rather than validating either prefix.

### `law_evidence`

- Required and must be a JSON list.
- Each item must contain exactly `law_id` and `aid`.
- The pair must exist in `corpus_law_pub.json`.
- `aid` is the article's corpus `aid`, not its position and not free text.
- Avoid duplicate `{law_id, aid}` pairs.

## Leaderboard workflow

1. Generate a candidate `submission.json` from a completed run.
2. Run the local validator against the exact test input and law corpus.
3. Inspect coverage, labels, evidence identifiers, JSON serialization, and absence of secrets/debug fields.
4. The submission owner opens the official submission page.
5. Enter the organizer-issued team name and secret token.
6. Upload `submission.json` and submit it for scoring.
7. Record the official result with the config, Git revision, model revisions, and run manifest.

The public leaderboard shows the best run per team. The official limit is 20 submissions per team per 24 hours. This repository must never submit automatically.

## Validation checklist

- [ ] Root is a JSON array.
- [ ] Object count equals the test-case count.
- [ ] Every expected `case_id` appears exactly once.
- [ ] No unknown or duplicate `case_id` exists.
- [ ] Every object has exactly the four required fields.
- [ ] Every `prediction` is an official label.
- [ ] `case_evidence` is `list[str]` and contains no guessed IDs.
- [ ] `case_evidence` contains no duplicates.
- [ ] `law_evidence` is a list of exact `{law_id, aid}` objects.
- [ ] Every law pair exists in the official corpus.
- [ ] No duplicate law pair exists.
- [ ] No reasoning, evidence text, API counters, logs, or secrets are present.
- [ ] All predictions completed successfully.
- [ ] JSON serialization is strict and contains no NaN values.
- [ ] The local opaque-ID validator reports `PASS`.

## Common rejection or scoring risks

- Missing, duplicate, or unknown cases.
- Wrong label spelling or casing.
- Stale sequential or guessed case-evidence identifiers.
- Evidence identifiers copied from non-official data.
- `law_evidence` expressed as strings instead of objects.
- Using article position instead of `aid`.
- Extra internal fields such as `api_calls` or `reasoning`.
- Uploading a partial smoke-run file.
- Uploading a mock prediction file.
- Reusing an old-format Public submission after the schema update.
- Trusting the current local `_chunk_` prefix check before it is fixed.

## Current implementation status

- Submission builder: implemented and CPU/mock verified.
- Coverage/label/law-pair validation: implemented and CPU/mock verified.
- Opaque `chunk_id`, strict schema/type, and 10 MB validation: `CPU/mock verified`.
- Manual upload: supported operationally, never automated.
- Updated-format leaderboard result: not leaderboard verified.
