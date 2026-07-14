# ALQAC 2026 Competition Requirements

**Last checked:** 2026-07-14

**Official sources:**

- [Task and scoring rules](https://alqac2026-leaderboard.ngrok.app/about)
- [Retrieval API](https://alqac2026-leaderboard.ngrok.app/api-docs)
- [Submission page](https://alqac2026-leaderboard.ngrok.app/submit)
- [ALQAC 2026 landing page](https://alqac.github.io/)

This document separates current official statements from repository decisions and unresolved questions. When sources conflict, do not guess: preserve the conflict as `Needs confirmation`.

## Official requirement: task

The task is **Legal Case Outcome Prediction with Evidence Retrieval**. Given a short Vietnamese case query, a system must:

1. predict the case outcome;
2. retrieve supporting case-content segments through the Case Content API; and
3. identify relevant legal provisions from the official law corpus.

The official outcome labels are:

- `A_WIN` — Plaintiff wins.
- `PARTIAL_A_WIN` — Plaintiff partially wins.
- `B_WIN` — Defendant wins.
- `PARTIAL_B_WIN` — Defendant partially wins.

Outcome evaluation is exact match over these four labels. The current rules do not define partial outcomes using numeric percentages. The older team interpretation of “more than 50%” versus “not more than 50%” is therefore `Needs confirmation`.

## Official requirement: score

```text
FinalScore = 0.70 × OutcomeAccuracy
           + 0.20 × PenalizedCaseRecall
           + 0.10 × LawF1_micro
```

For case `i`, the API-efficiency factor is:

```text
E_i = max(0, 1 − max(0, c_i − 2·n_i) / (3·n_i))
```

Where:

- `c_i` is the Case Content API call count recorded by the server for the case;
- `n_i` is the number of content segments in the case;
- there is no penalty through `2·n_i` calls; and
- the factor reaches zero at `5·n_i` calls.

`PenalizedCaseRecall` is case-evidence recall multiplied by this efficiency factor. Teams do not submit `c_i`; the organizer calculates it from server logs.

### Critical clarification: calls are cumulative

Case Content API calls accumulate across every run and experiment. Logs are append-only and are not reset between Public Test and Private Test activity. Public experimentation can therefore reduce the efficiency component of later Private submissions. Every real API call must be treated as part of a shared team budget.

## Official requirement: submission and leaderboard

- Upload one file named `submission.json`.
- The root value is a JSON array with exactly one object per test case.
- Every object contains `case_id`, `prediction`, `case_evidence`, and `law_evidence`.
- `case_evidence` is required and may be an empty list.
- `law_evidence` contains `{law_id, aid}` objects from `corpus_law_pub.json`.
- Duplicate evidence identifiers are de-duplicated before scoring.
- The upload page accepts files up to 10 MB.
- Submission requires the organizer-issued team name and secret token.
- The public leaderboard displays the best run per team.
- The current limit is 20 submissions per team per 24 hours.
- Upload is a manual team-owner action; this repository must not upload automatically.

## Official requirement: evidence identifiers

The leaderboard update announcement states that case evidence identifiers are opaque hashed values, for example:

```text
case_1959_seg_9b2898839a509e4f
```

The identifier for a segment remains stable across runs. A segment discovered in an earlier run may be cited in a later submission.

### Open question: inconsistent examples

`Needs confirmation`: the current rules, API documentation, and submission page still contain examples such as `case_4101_chunk_3` and `case_1087_0037_chunk_2`, while the leaderboard announcement says sequential `..._chunk_N` identifiers were replaced by opaque `..._seg_<hash>` identifiers.

Until the organizer resolves this inconsistency:

- treat `chunk_id` as an opaque string;
- preserve the exact value returned by the Case Content API;
- never construct or guess an identifier; and
- do not validate identifiers using a `_chunk_` or `_seg_` naming assumption.

## Provided data and resources

The organizer-provided files currently present in this repository are:

- `ALQAC2026_public_test.json`: 50 labeled Public Test cases;
- `corpus_law_pub.json`: 18 law documents containing 3,352 articles.

The current official web pages describe the input as a short Vietnamese case query and require a valid `case_id` for API retrieval. They do not publish a complete new Private Test JSON example.

`Needs confirmation`: whether the refreshed Private Test release still contains only `case_id` and `case_query`.

## Timeline

The official ALQAC landing page lists these paper and event dates, all distinct from leaderboard/test deadlines:

| Event | Date |
|---|---|
| Paper submission | 2026-07-15 |
| Acceptance notification | 2026-08-31 |
| Camera-ready | 2026-09-10 |
| Conference | 2026-11-11 to 2026-11-14 |

`Needs confirmation`: Public Test, Private Test, leaderboard freeze, source-code delivery, and technical-report deadlines are not stated on the refreshed leaderboard documentation.

## Restrictions and deliverables

The refreshed leaderboard pages do not state:

- model parameter or licensing restrictions;
- whether proprietary model APIs are prohibited;
- external-data or external-annotation restrictions;
- source-code delivery requirements; or
- technical-report delivery requirements.

These were present in earlier team notes but are not confirmed by the refreshed pages. They remain `Needs confirmation` and must not be presented as current official requirements without an organizer source.

## Team implementation

This repository currently adopts stricter internal safeguards:

- production inference receives only `case_id` and `case_query`;
- public-only gold fields are isolated in evaluation code;
- the current model stack uses open-weight models;
- secrets are read from environment or notebook secret storage;
- API responses are cached and runs are checkpointed;
- submission validation is mandatory; and
- leaderboard upload is always manual.

These are team implementation decisions unless explicitly identified above as official requirements.

## Open questions summary

1. What is the exact accepted shape of current `chunk_id` values?
2. Does the refreshed Private Test contain only `case_id` and `case_query`?
3. How should `PARTIAL_A_WIN` and `PARTIAL_B_WIN` be distinguished beyond the party description?
4. What model, API, and external-data restrictions apply?
5. Are source code and a technical report mandatory deliverables?
6. What are the Public/Private Test and leaderboard deadlines?

