# ALQAC 2026 Competition Requirements

**Last checked:** 2026-07-20

**Official sources:**

- [Task and scoring rules](https://alqac2026-leaderboard.ngrok.app/about)
- [Retrieval API](https://alqac2026-leaderboard.ngrok.app/api-docs)
- [Submission page](https://alqac2026-leaderboard.ngrok.app/submit)
- [Official ALQAC 2026 competition website](https://sites.google.com/view/alqac2026)
- [ALQAC 2026 landing page](https://alqac.github.io/)

This document separates current official statements from repository decisions and unresolved questions. When sources conflict, do not guess: preserve the conflict as `Needs confirmation`.

## Official requirement: task

The task is **Legal Case Outcome Prediction with Evidence Retrieval**. Given a short Vietnamese case query, a system must:

1. predict the case outcome;
2. retrieve supporting case-content segments through the Case Content API; and
3. identify relevant legal provisions from the official law corpus.

The official outcome labels are:

- `A_WIN` — Plaintiff wins.
- `PARTIAL_A_WIN` — The court partially accepts the plaintiff's claims, and the accepted portion is greater than 50%.
- `B_WIN` — Defendant wins.
- `PARTIAL_B_WIN` — The court partially accepts the plaintiff's claims, and the accepted portion is 50% or less.

Outcome evaluation is exact match over these four labels. When a case contains multiple claims, the official competition website instructs teams to focus on the main claim described in `case_query`.

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
- Upload is a manual team-owner action; this repository must not upload automatically.

### Submission-limit conflict

`Needs confirmation`: two current official organizer pages state different general limits:

- the competition website allows at most **3 submissions per day**; and
- the leaderboard rules state **20 submissions per team per 24 hours**.

Until the organizers clarify the conflict, this repository uses the stricter operational limit of at most three submissions in any 24-hour period.

The submission page additionally defines track-specific behavior:

- Public Test may be submitted repeatedly, subject to the stricter operational limit above;
- Private Test permits at most **3 named runs in total**;
- every Private run name must be distinct and cannot be reused;
- the Private submission flow performs **Check format** before final confirmation; and
- the best Private run counts, while Private rankings remain hidden until the organizers reveal them.

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
- `corpus_law_pub.json`: 18 law documents containing 3,352 articles; and
- local ignored `ALQAC_private_test.json`: 60 unlabeled Private Test cases.

The official competition website defines test input as a JSON array containing only `case_id` and `case_query`; it explicitly excludes the gold verdict, court reasoning, court decision, and gold evidence.

The local Private Test file `data/raw/ALQAC_private_test.json` conforms to that contract: 60 cases, exactly the two allowed fields, non-empty string values, no duplicate `case_id`, and no overlap with the 50 Public Test identifiers. Its integrity details are recorded in `docs/data_api.md`.

## Timeline

The official ALQAC landing page lists these paper and event dates, all distinct from leaderboard/test deadlines:

| Event | Date |
|---|---|
| Paper submission | 2026-07-15 |
| Acceptance notification | 2026-08-31 |
| Camera-ready | 2026-09-10 |
| Conference | 2026-11-11 to 2026-11-14 |

`Needs confirmation`: Public Test, Private Test, leaderboard freeze, source-verification, and technical-report deadlines are not stated on the current official pages.

## Official restrictions and reproducibility

The official competition website states:

- closed or proprietary systems and non-open model APIs, including ChatGPT, GPT-4, Claude, and Gemini, are prohibited;
- only open-weight models with fewer than 10 billion parameters are allowed;
- online legal-database queries are permitted;
- externally annotated datasets created specifically for legal question answering or legal entailment are prohibited; and
- submissions that violate these rules are disregarded from the final ranking.

Participating teams are encouraged, but not explicitly required, to submit a short technical report describing the Case Content API retrieval strategy, law-corpus retrieval strategy, outcome model, training/fine-tuning data, and final configuration. The organizers may request source code, configuration files, or logs for verification and reproducibility.

`Needs confirmation`: whether a technical report or source package becomes a mandatory deliverable, and the deadline or transfer procedure for any requested verification artifacts.

## Team implementation

This repository currently adopts stricter internal safeguards:

- production inference receives only `case_id` and `case_query`;
- public-only gold fields are isolated in evaluation code;
- all pinned outcome, embedding, and reranking models are public open-weight models below 10 billion parameters;
- proprietary model APIs and prohibited externally annotated legal datasets are excluded;
- secrets are read from environment or notebook secret storage;
- API responses are cached and runs are checkpointed;
- submission validation is mandatory; and
- leaderboard upload is always manual.

These are team implementation decisions unless explicitly identified above as official requirements.

## Open questions summary

1. Will the stale sequential `chunk_id` examples be removed from the rules/API pages now that the leaderboard announces opaque hashed identifiers?
2. Which general submission limit governs ranking eligibility: three per day or 20 per 24 hours?
3. Are source code and a technical report mandatory deliverables, and if so, what are their deadlines?
4. What are the Public/Private Test and leaderboard deadlines?
