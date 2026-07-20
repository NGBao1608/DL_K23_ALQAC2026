# Submission Specification and Checklist

**Last checked:** 2026-07-20

**Sources:** [task rules](https://alqac2026-leaderboard.ngrok.app/about), [submission page](https://alqac2026-leaderboard.ngrok.app/submit), [official competition website](https://sites.google.com/view/alqac2026), and the leaderboard update announcement.

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

Official partial-label boundary:

- `PARTIAL_A_WIN`: the court accepts more than 50% of the plaintiff's main claim;
- `PARTIAL_B_WIN`: the court accepts 50% or less of the plaintiff's main claim.

For multiple-claim cases, focus on the main claim described in `case_query`.

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

## Public leaderboard workflow

1. Generate a candidate `submission.json` from a completed run.
2. Run the local validator against the exact test input and law corpus.
3. Inspect coverage, labels, evidence identifiers, JSON serialization, and absence of secrets/debug fields.
4. The submission owner opens the official submission page.
5. Enter the organizer-issued team name and secret token.
6. Upload `submission.json` and submit it for scoring.
7. Record the official result with the config, Git revision, model revisions, and run manifest.

The public leaderboard shows the best run per team and accepts repeated Public submissions.

## Private submission workflow

1. Select a completed, full, non-mock Private run with exactly 60 cases.
2. Validate the exact file against `data/raw/ALQAC_private_test.json`.
3. Choose a distinct, reproducible run name tied to the local manifest.
4. Select **Private Test**, upload the file, enter the run name, and use **Check format**.
5. Confirm the case count and remaining-run message before final submission.
6. Manually confirm submission; a run name cannot be reused.
7. Record the team's visible score and submission identifier locally without recording credentials.

Private Test permits at most three named runs in total and the best run counts. Private rankings are hidden from the public leaderboard until the organizers reveal them.

## Submission-limit conflict

`Needs confirmation`: the official competition website says at most three submissions per day, while the leaderboard rules say 20 submissions per team per 24 hours. Follow the stricter team rule: at most three submissions in any 24-hour period, while also respecting the separate three-run lifetime limit for Private Test. This repository must never submit automatically.

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
- [ ] The run respects the stricter three-submissions-per-24-hours team limit.
- [ ] For Private Test, the run name is distinct and the team has fewer than three prior Private runs.
- [ ] For Private Test, the website's **Check format** step succeeds before final confirmation.

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
- Spending a Private run slot on an unreviewed configuration or a reused run name.
- Following the leaderboard's 20-per-24-hours platform cap while violating the competition website's stricter three-per-day rule.

## Current implementation status

- Submission builder: implemented and CPU/mock verified.
- Coverage/label/law-pair validation: implemented and CPU/mock verified.
- Opaque `chunk_id`, strict schema/type, and 10 MB validation: `CPU/mock verified`.
- Validation/export binding by submission SHA-256, byte length, and actual case
  count: `CPU/mock verified`.
- Manual upload: supported operationally, never automated.
- Updated-format official result: no current run is `leaderboard verified`.
