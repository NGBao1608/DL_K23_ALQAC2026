# Competition rules used by the implementation

Primary sources: the [ALQAC 2026 official website](https://sites.google.com/view/alqac2026) and organizer announcements sent to the team. When sources conflict, direct confirmation from the organizers takes precedence and this document must be updated.

## Task

Input for each case:

```json
{"case_id":"case_4101","case_query":"Mô tả ngắn tranh chấp..."}
```

The private test does not provide case judgment content or gold annotations. Case content must be retrieved through the Case Content API.

Prediction labels:

- `A_WIN`: the plaintiff's main claim is fully accepted.
- `PARTIAL_A_WIN`: a portion greater than 50% is accepted.
- `PARTIAL_B_WIN`: a portion of at most 50% is accepted.
- `B_WIN`: the main claim is fully rejected.

When a case has multiple claims, the prediction focuses on the main claim described in `case_query`.

## Resources and API

- Public development set: 50 labeled cases provided by the organizers.
- Private test: only `case_id` and `case_query`.
- Law corpus: 18 legal documents, 3,352 articles; submissions identify a provision by `law_id` and `aid`.
- Case API: `POST https://alqac-api.ngrok.pro/retrieve` with an `X-API-Key` header and body `{"query":"...","case_id":"..."}`.
- Each call returns the top-1 segment; the rate limit is one request every 5 seconds per team.

## Evaluation

The official score is:

```text
0.70 × Outcome Accuracy
+ 0.20 × Penalized Case Evidence Recall
+ 0.10 × Micro Law Evidence F1
```

Case evidence recall is penalized by API efficiency: full efficiency holds up to `2n` calls and decays to 0 at `5n`, where `n` is the number of segments in the case. The code does not infer the detailed part of the formula that the organizers have not provided in executable form.

## Submission

Submit a single JSON array with exactly one object per test case, containing:

- `case_id`.
- `prediction`.
- `case_evidence`: a list of official `chunk_id` values, which may be empty.
- `law_evidence`: a list of valid `{law_id, aid}` entries.

No duplicate, missing, or unknown case IDs or evidence identifiers. Leaderboard upload is manual, with at most three submissions per day per team.

## Restrictions

- Only open-weight models under 10B parameters may be used.
- ChatGPT/GPT, Claude, Gemini, and other proprietary model APIs must not be used in the pipeline.
- Externally annotated datasets created specifically for legal QA or legal entailment must not be used.
- Online legal databases may be queried under organizer rules, but submitted evidence must use official identifiers.
- The organizers may request source code, config, and logs to check reproducibility.

## Project enforcement

- The inference schema contains only `case_id` and `case_query`.
- Tokens are read only from the environment or from Colab Secrets.
- The source code never uploads a submission automatically.
- The validator rejects extra fields such as `api_calls`, failed predictions, and law identifiers not present in the corpus.
