# Official Data and Case Content API

**Last checked:** 2026-07-14

**Sources:** [Retrieval API](https://alqac2026-leaderboard.ngrok.app/api-docs), [task rules](https://alqac2026-leaderboard.ngrok.app/about), and organizer-provided raw files in `data/raw/`.

## Official data files

### `ALQAC2026_public_test.json`

The repository copy contains 50 Public Test cases. It is a JSON array. The observed fields are:

```text
A_description, A_role, B_description, B_role, annotation_id,
case_fact, case_id, case_query, case_type, court, court_level,
court_reasoning, court_verdict, judgment_date, judgment_number,
judgment_text, raw_sha256, related_law_provisions, source_filename,
verdict_label
```

Only `case_id` and `case_query` may enter production inference. All other fields are Public Test metadata or gold annotations and may be read only by evaluation and error-analysis code.

Expected private-like inference item:

```json
{
  "case_id": "case_1087_0037",
  "case_query": "A short Vietnamese case query"
}
```

`Needs confirmation`: the refreshed official pages do not publish the complete new Private Test schema.

### `corpus_law_pub.json`

The repository copy contains 18 law documents and 3,352 articles.

Observed document shape:

```json
{
  "id": 1,
  "law_id": "47/2010/QH12",
  "content": [
    {
      "aid": 270,
      "content_Article": "Article text"
    }
  ]
}
```

`aid` is the corpus article identifier. It is not the list position, an article number inferred from text, or a free-text citation.

## Case Content API

Base URL:

```text
https://alqac-api.ngrok.pro
```

Interactive documentation:

```text
https://alqac-api.ngrok.pro/docs
```

### Authentication

Every request must contain the organizer-issued team token:

```http
X-API-Key: <team-secret-token>
```

The same organizer-issued token is used for leaderboard submission. Never hardcode, print, serialize, or commit it. Read it from `ALQAC_TEAM_TOKEN` or approved notebook secret storage.

### `POST /retrieve`

Request:

```json
{
  "query": "tranh chấp quyền sử dụng đất",
  "case_id": "case_1087_0037"
}
```

Response `200`:

```json
{
  "results": [
    {
      "score": 0.886,
      "text": "The retrieved case segment",
      "chunk_id": "opaque_identifier_returned_by_the_api"
    }
  ]
}
```

- Exactly one top-ranked segment is returned per successful call.
- `score` is the API BM25 relevance score.
- Preserve `chunk_id` exactly as returned.
- Issue multiple queries only when their expected evidence value justifies the permanent call cost.

### Rate limit and errors

The official limit is one request every five seconds per team.

| Status | Meaning | Team behavior |
|---|---|---|
| `200` | Success | Validate, cache, and checkpoint the response. |
| `403` | Missing or invalid `X-API-Key` | Stop; do not retry automatically. |
| `422` | Missing or malformed `query`/`case_id` | Stop and fix the request. |
| `429` | Rate limit exceeded | Wait at least five seconds, then retry with bounded backoff. |
| `503` | Team database temporarily unavailable | Retry with bounded exponential backoff. |

Infrastructure error pages from ngrok are not Case API JSON responses. Log only safe metadata such as HTTP status, content type, request ID, and ngrok error code. Never log headers or token values.

## Permanent call accounting

The organizer counts every Case Content API call ever made by the team for each case, across Public and Private runs. Logs are not reset.

Team rules:

1. Cache every successful response by API version, `case_id`, and normalized query.
2. Never repeat a successful query for the same cache key.
3. Resume from checkpoints instead of restarting retrieval.
4. Use mock runs for plumbing tests.
5. Review the proposed query count before any real API experiment.
6. Share one team cache or evidence registry when operationally possible.
7. Record network calls and cache hits in run artifacts, but never in `submission.json`.
8. Require an explicit hard network-attempt cap for every non-mock run.
9. Treat local ledger totals as a lower-bound operational record, never the official BTC count.

## Current repository behavior

- `case_retrieval.py` implements throttling, bounded retry, SQLite caching, de-duplication, and safe error diagnostics.
- The current baseline/candidate configs use exactly two deterministic queries per case: `court_decision` and normalized `case_query`.
- `scripts/plan_api_calls.py` reports cache hits/misses without contacting the API.
- Every non-mock runner requires `--max-network-calls`; the client refuses the next request after the cap is reached.
- SQLite stores successful evidence responses and a safe attempt ledger containing query hashes rather than query text or credentials.
- API responses are checkpointed into prepared contexts for resume.
- `chunk_id` is stored as a string and can carry opaque hashed values.
- During a non-mock run, the runner emits safe `ALQAC_PROGRESS` events for cache hits, HTTP request start/completion, HTTP errors, exceptions, and scheduled retries. Events include only `case_id`, query type, attempt metadata, status metadata, and latency; they never include the query text, response text, headers, or token.

## Verification status

Opaque identifier storage/validation, two-query generation, cache planning, hard budgets, retries, and the local call ledger are `CPU/mock verified`. The refreshed official API path remains pending `GPU/API verified` evidence.
