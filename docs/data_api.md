# Official Data and Case Content API

**Last checked:** 2026-07-23

**Sources:** [Retrieval API](https://alqac2026-leaderboard.ngrok.app/api-docs), [task rules](https://alqac2026-leaderboard.ngrok.app/about), [official competition website](https://sites.google.com/view/alqac2026), the live read-only [OpenAPI schema](https://alqac-api.ngrok.pro/openapi.json), and organizer-provided raw files in `data/raw/`.

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

Only `case_id` and `case_query` may enter the production data loader. All other
fields are Public Test metadata or gold annotations. They may be used only by
offline training/target construction, evaluation, and error-analysis code, and
must never enter Private production inference.

Official inference item:

```json
{
  "case_id": "case_1087_0037",
  "case_query": "A short Vietnamese case query"
}
```

The official competition website states that the test input contains only `case_id` and `case_query` and does not include the gold verdict, court reasoning, court decision, or gold evidence.

### `ALQAC_private_test.json`

The Private Test copy is stored at `data/raw/ALQAC_private_test.json` and is
tracked only in the verified private GitHub repository. Observed integrity
contract:

| Property | Observed value |
|---|---|
| SHA-256 | `9db83cf98ade7d19df52c60145830bebcc192e064ec830bcd285cefbfddf0252` |
| Cases | 60 |
| Exact fields per item | `case_id`, `case_query` |
| Field types | Non-empty strings |
| Duplicate Private `case_id` | 0 |
| Public/Private `case_id` overlap | 0 |
| Gold or prohibited inference fields | 0 |

This file is the canonical Private input for the source-pinned checkout. It may
be read by `load_inference_cases()` because that loader projects only `case_id`
and `case_query`. It must never be edited, included in source bundles, or copied
into run/export artifacts other than through derived identifiers and
predictions allowed by the submission contract. A push containing it is allowed
only while repository visibility is verified private.

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

### `private_test_60_cases_extracted_corpus.json`

The organizer-provided Private law corpus is stored at
`data/raw/private_test_60_cases_extracted_corpus.json` and is tracked only in
the verified private GitHub repository. Observed integrity contract:

| Property | Observed value |
|---|---|
| SHA-256 | `9d79379e017ce346cf143a71fa82f5170a755c33a0341048c9baacb28c6119b5` |
| Law documents | 14 |
| Articles | 2,820 |

Private law retrieval and validation must use this corpus rather than
`corpus_law_pub.json`. The file is immutable organizer data and must never be
included in source bundles or copied into exports.

## Case Content API

Base URL:

```text
https://alqac-api.ngrok.pro
```

Interactive documentation:

```text
https://alqac-api.ngrok.pro/docs
```

The live read-only OpenAPI document reports API version `1.0.0`. Checking that schema does not call `POST /retrieve` and does not spend the team's scored retrieval budget.

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

The organizer counts every Case Content API call ever made by the team for each
case. Logs are not reset. The reviewed Public and Private inputs have zero
overlapping `case_id` values. Because the published formula defines `c_i` per
case, Public calls should not directly increase Private `c_i`. `Needs
confirmation`: whether the organizer applies any additional team-wide
cross-track accounting.

Team rules:

1. Cache every successful response by API version, `case_id`, and normalized query.
2. Never repeat a successful query for the same cache key.
3. Resume from checkpoints instead of restarting retrieval.
4. Use mock runs for plumbing tests.
5. Review the proposed query count before any real API experiment.
6. Share one team cache or evidence registry when operationally possible.
7. Record network calls and cache hits in run artifacts, but never in `submission.json`.
8. Require an explicit hard network-attempt cap for every live run; Public live
   quality evaluation additionally requires explicit approval. Use cache-only
   with cap zero for diagnostics that do not need new case evidence.
9. Treat local ledger totals as a lower-bound operational record, never the official BTC count.

## Current repository behavior

- `query_planning.py` implements the structured deterministic planner, pinned
  `Qwen/Qwen3-0.6B` LLM-assisted planner, validation/fallback, deterministic
  query composer, evidence-sufficiency gate, and fingerprinted query-plan store.
- Planner output must copy query spans/lexical terms, must not infer an outcome,
  and falls back safely on load, timeout, generation, JSON, schema, or grounding
  failure.
- Baseline and candidate use two primary structured queries. A third adaptive
  query is issued only when the first two results fail the evidence gate and
  per-case budget remains.
- `case_retrieval.py` implements throttling, at most one retry, SQLite caching,
  exact `chunk_id` de-duplication, global/per-case hard caps, and safe error
  diagnostics.
- `scripts/plan_api_calls.py` reports a structured upper-bound plan and
  cache hits/misses without contacting the API.
- Every live runner requires `--max-network-calls`; cache-only never constructs the HTTP client and records zero network attempts.
- The default hard cap is three network attempts per case. Semantic queries and
  retries share it; cache hits spend no attempt. A recreated client restores
  per-case counts from the SQLite ledger for the same run key.
- SQLite commits a successful response, its safe attempt ledger row, and a pending-backup marker atomically. With external backup configured, every attempt must be backed up successfully before the next request. A resumed live client repairs pending state before returning a cache hit or sending another request; restore preserves a newer local database while that marker is pending.
- API responses are checkpointed into prepared contexts for resume. Query plans
  are separately persisted by a fingerprint containing `case_id`, the raw
  `case_query` SHA-256, configured planner strategy/model revision, planner
  prompt version, and composer version.
- `chunk_id` is stored as a string and can carry opaque hashed values.
- During retrieval, the runner emits safe `ALQAC_PROGRESS` events for cache hits/misses, HTTP request start/completion, HTTP errors, exceptions, and scheduled retries. Events include only `case_id`, query type, attempt metadata, status metadata, and latency; they never include the query text, response text, headers, or token.

## Verification status

Private input loading, opaque identifier storage/validation, structured
fallback planning, adaptive-query gating, cache planning, hard budgets, retries,
resume-safe per-case ledger counts, and safe logging are `CPU/mock verified`.
The pinned planner and end-to-end retrieval path still lack a completed
`GPU/API verified` artifact.
