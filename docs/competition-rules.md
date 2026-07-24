# Competition Rules — Synchronized Legacy View

**Last synchronized:** 2026-07-20

**Canonical source:** [`competition.md`](competition.md)

This filename is retained for existing links and older workflows. The canonical document remains `competition.md`; update that file first whenever organizer requirements change.

## Task

ALQAC 2026 uses **Legal Case Outcome Prediction with Evidence Retrieval**. For each short Vietnamese case query, the system predicts one exact outcome label and submits supporting case and law evidence.

Valid labels:

- `A_WIN` — Plaintiff wins.
- `PARTIAL_A_WIN` — More than 50% of the plaintiff's main claim is accepted.
- `B_WIN` — Defendant wins.
- `PARTIAL_B_WIN` — 50% or less of the plaintiff's main claim is accepted.

For multiple-claim cases, focus on the main claim described in `case_query`.

## Score

```text
FinalScore = 0.70 × OutcomeAccuracy
           + 0.20 × PenalizedCaseRecall
           + 0.10 × LawF1_micro
```

For case `i`:

```text
E_i = max(0, 1 − max(0, c_i − 2·n_i) / (3·n_i))
```

Case Content API calls receive full efficiency through `2·n_i` calls and decay to zero at `5·n_i`. The organizer obtains `c_i` from append-only server logs.

Critical rule: calls accumulate for each `case_id` across runs and are never
reset. The reviewed Public and Private identifiers are disjoint, so Public
calls should not directly increase a Private case's `c_i`. `Needs confirmation`:
whether any additional team-wide cross-track accounting exists. Real API
experiments must use a reviewed team budget and reuse cached successful
responses.

## Submission

Upload one `submission.json` containing one object per test case with exactly:

```json
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
```

- `case_evidence` is required and may be empty.
- `law_evidence` uses corpus-valid `{law_id, aid}` pairs.
- Every test case appears exactly once.
- Upload size is at most 10 MB.
- The public leaderboard shows the best run per team.
- Upload is always manual.

`Needs confirmation`: the competition website says at most three submissions per day, while the leaderboard rules say 20 per team per 24 hours. The team follows the stricter limit of at most three submissions in any 24-hour period.

Public submissions may be repeated within that limit. Private Test allows at most three distinctly named runs in total, requires a format-check step before final confirmation, and uses the team's best Private run.

## Evidence identifiers

The leaderboard announcement says `chunk_id` is now an opaque hashed identifier that remains stable across runs. Other current official pages still show sequential `_chunk_N` examples.

`Needs confirmation`: the exact accepted naming shape is inconsistent across official pages. Always preserve the exact API-returned string; never construct or validate a guessed prefix.

## Data and API

- Public data in the repository: 50 labeled cases.
- Law corpus: 18 documents and 3,352 articles.
- Case API: `POST https://alqac-api.ngrok.pro/retrieve`.
- Authentication: `X-API-Key` with the organizer-issued team token.
- Request: `{query, case_id}`.
- Successful response: one top-ranked `{score, text, chunk_id}` result.
- Rate limit: one request every five seconds per team.

Production inference in this repository consumes only `case_id` and
`case_query`; Public gold is restricted to offline training/target construction,
evaluation, and error analysis.

The private-repository `data/raw/ALQAC_private_test.json` contains 60 unique
cases with exactly those two fields. It remains excluded from source bundles
and exports.

## Timeline and restrictions

The official ALQAC landing page lists paper submission on 2026-07-15, notification on 2026-08-31, camera-ready on 2026-09-10, and the conference on 2026-11-11 through 2026-11-14.

These are not assumed to be Public/Private Test or leaderboard deadlines.

Official restrictions:

- only open-weight models with fewer than 10 billion parameters;
- no proprietary or non-open model APIs; and
- no externally annotated legal QA or legal entailment datasets.

Online legal-database lookup is permitted. A short technical report is encouraged, and organizers may request source, configuration, or logs for verification.

`Needs confirmation`:

- Public/Private Test and leaderboard deadlines;
- which conflicting general submission limit governs ranking eligibility; and
- whether source or report artifacts become mandatory and, if so, their deadlines.

## Team safeguards

- Use only `case_id` and `case_query` for inference.
- Keep secrets in environment or approved secret storage.
- Cache successful API responses and resume checkpoints.
- Validate every candidate submission.
- Never upload automatically.
- Use only evidence-backed verification statuses from `AGENTS.md`.
