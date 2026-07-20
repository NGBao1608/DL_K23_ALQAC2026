import pytest
import requests

from alqac2026.case_retrieval import (
    ApiBudgetExceeded,
    CachedCaseContentClient,
    CaseContentClient,
    EvidenceQueryGenerator,
    SQLiteEvidenceCache,
    build_api_plan,
    cache_key,
)
from alqac2026.law_retrieval import extract_law_citations, law_index_fingerprint
from alqac2026.schemas import InferenceCase, LawArticle
from alqac2026.law_retrieval import reciprocal_rank_fusion


class FakeResponse:
    def __init__(self, status_code, payload, text="", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}
        self.url = "https://example.test/retrieve"

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(str(self.status_code))


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_rrf_is_deterministic_and_rewards_overlap():
    ranked = reciprocal_rank_fusion([[1, 2, 3], [3, 2, 4]], rrf_k=60)
    identifiers = [identifier for identifier, _ in ranked]
    assert identifiers[:2] == [3, 2]


def test_law_index_fingerprint_changes_with_corpus_or_model_revision():
    articles = [LawArticle("law", 1, 1, "Nghĩa vụ thanh toán")]
    config = {"embedding_model": "model", "embedding_revision": "rev-1"}
    first = law_index_fingerprint(config, articles)
    assert first == law_index_fingerprint(config, articles)
    assert first != law_index_fingerprint(
        {**config, "embedding_revision": "rev-2"}, articles
    )
    assert first != law_index_fingerprint(
        config, [LawArticle("law", 1, 1, "Nghĩa vụ hoàn trả")]
    )


def test_exact_law_citations_are_extracted_in_document_order():
    articles = [
        LawArticle("91/2015/QH13", 1001, 584, "Article 584"),
        LawArticle("91/2015/QH13", 1002, 590, "Article 590"),
        LawArticle("92/2015/QH13", 2001, 147, "Article 147"),
    ]
    text = (
        "Căn cứ các Điều 584, Điều 590 của Bộ luật Dân sự năm 2015; "
        "Điều 147 Bộ luật Tố tụng dân sự."
    )
    assert extract_law_citations(text, articles) == [
        ("91/2015/QH13", 584),
        ("91/2015/QH13", 590),
        ("92/2015/QH13", 147),
    ]


def test_law_citation_extractor_ignores_articles_outside_corpus():
    articles = [LawArticle("91/2015/QH13", 1001, 584, "Article 584")]
    assert extract_law_citations(
        "Điều 999 và Điều 584 Bộ luật Dân sự", articles
    ) == [("91/2015/QH13", 584)]


def test_law_citation_extractor_skips_explicit_wrong_law_year():
    articles = [LawArticle("91/2015/QH13", 1001, 131, "Article 131")]
    assert extract_law_citations(
        "Áp dụng Điều 131 của Bộ luật Dân sự năm 1995", articles
    ) == []


def test_cache_key_normalizes_whitespace_and_case():
    assert cache_key("case_1", "  Nhận định   Tòa án ") == cache_key(
        "case_1", "nhận định tòa án"
    )


def test_two_query_policy_is_deterministic():
    generator = EvidenceQueryGenerator()
    queries = generator.generate("  Tranh chấp   Hợp đồng  ", max_queries=2)
    assert [(item.query_type, item.text) for item in queries] == [
        ("court_decision", "quyết định của tòa án tuyên xử"),
        ("original", "tranh chấp hợp đồng"),
    ]


def test_api_retry_then_cache_hit(tmp_path):
    cache = SQLiteEvidenceCache(tmp_path / "cache.sqlite")
    session = FakeSession(
        [
            FakeResponse(503, {}),
            FakeResponse(
                200,
                {
                    "results": [
                        {
                            "chunk_id": "case_1_seg_9b2898839a509e4f",
                            "score": 1.0,
                            "text": "x",
                        }
                    ]
                },
            ),
        ]
    )
    client = CaseContentClient(
        token="test-token",
        base_url="https://example.test",
        cache=cache,
        retries=2,
        request_interval_seconds=0,
        session=session,
        sleep=lambda _: None,
        clock=lambda: 1.0,
    )
    first = client.retrieve("case_1", "query")
    second = client.retrieve("case_1", "query")
    assert first == second
    assert client.network_calls == 2
    assert client.cache_hits == 1
    cache.close()


def test_sqlite_backup_refuses_to_replace_live_database(tmp_path):
    cache_path = tmp_path / "cache.sqlite"
    cache = SQLiteEvidenceCache(cache_path)
    with pytest.raises(ValueError, match="must differ"):
        cache.backup_to(cache_path)
    cache.close()


def test_api_progress_events_are_safe_and_cover_retry_and_cache_hit(tmp_path):
    cache = SQLiteEvidenceCache(tmp_path / "cache.sqlite")
    events = []
    client = CaseContentClient(
        token="secret-token",
        base_url="https://example.test",
        cache=cache,
        retries=2,
        request_interval_seconds=0,
        session=FakeSession(
            [
                FakeResponse(503, {}),
                FakeResponse(200, {"results": []}),
            ]
        ),
        sleep=lambda _: None,
        clock=lambda: 1.0,
        progress_callback=events.append,
    )

    client.retrieve("case_1", "sensitive query", query_type="original")
    client.retrieve("case_1", "sensitive query", query_type="original")

    assert [event["status"] for event in events] == [
        "request_started",
        "http_error",
        "retry_scheduled",
        "request_started",
        "request_completed",
        "cache_hit",
    ]
    assert events[0]["timeout_seconds"] == 30
    assert events[1]["status_code"] == 503
    assert events[-1]["result_count"] == 0
    assert all("query" not in event for event in events)
    assert all("secret-token" not in str(event) for event in events)
    cache.close()


def test_api_strips_secret_and_redacts_403_diagnostics(tmp_path):
    cache = SQLiteEvidenceCache(tmp_path / "cache.sqlite")
    session = FakeSession(
        [
            FakeResponse(
                403,
                {},
                text="Forbidden for secret-token",
                headers={"Content-Type": "application/json", "Server": "test"},
            )
        ]
    )
    client = CaseContentClient(
        token="  secret-token\n",
        base_url="https://example.test",
        cache=cache,
        request_interval_seconds=0,
        session=session,
        sleep=lambda _: None,
        clock=lambda: 1.0,
    )

    try:
        client.retrieve("case_1", "query")
    except PermissionError as error:
        message = str(error)
    else:
        raise AssertionError("Expected a 403 PermissionError")

    assert session.calls[0][1]["headers"]["X-API-Key"] == "secret-token"
    assert (
        session.calls[0][1]["headers"]["ngrok-skip-browser-warning"]
        == "alqac2026-api-client"
    )
    assert "secret-token" not in message
    assert "response_body=<redacted>" in message
    cache.close()


def test_api_budget_caps_retry_attempts(tmp_path):
    cache = SQLiteEvidenceCache(tmp_path / "cache.sqlite")
    session = FakeSession(
        [FakeResponse(503, {}), FakeResponse(200, {"results": []})]
    )
    client = CaseContentClient(
        token="test-token",
        base_url="https://example.test",
        cache=cache,
        retries=2,
        max_network_calls=1,
        request_interval_seconds=0,
        session=session,
        sleep=lambda _: None,
        clock=lambda: 1.0,
    )
    with pytest.raises(ApiBudgetExceeded, match="budget exhausted"):
        client.retrieve("case_1", "query", query_type="original")
    assert client.network_calls == 1
    assert len(session.calls) == 1
    assert cache.known_cumulative_attempts() == 1
    cache.close()


def test_api_budget_persists_across_client_resume(tmp_path):
    cache = SQLiteEvidenceCache(tmp_path / "cache.sqlite")
    first_session = FakeSession([FakeResponse(200, {"results": []})])
    first_client = CaseContentClient(
        token="test-token",
        base_url="https://example.test",
        cache=cache,
        retries=1,
        max_network_calls=1,
        run_key="stable-run",
        request_interval_seconds=0,
        session=first_session,
        sleep=lambda _: None,
        clock=lambda: 1.0,
    )
    first_client.retrieve("case_1", "first query", query_type="original")

    resumed_session = FakeSession([FakeResponse(200, {"results": []})])
    resumed_client = CaseContentClient(
        token="test-token",
        base_url="https://example.test",
        cache=cache,
        retries=1,
        max_network_calls=1,
        run_key="stable-run",
        request_interval_seconds=0,
        session=resumed_session,
        sleep=lambda _: None,
        clock=lambda: 1.0,
    )
    with pytest.raises(ApiBudgetExceeded, match="budget exhausted"):
        resumed_client.retrieve("case_1", "second query", query_type="original")
    assert resumed_client.network_calls == 1
    assert resumed_session.calls == []
    assert cache.run_attempt_stats("stable-run")["network_attempts"] == 1
    cache.close()


def test_timeout_is_counted_and_logged_without_secret(tmp_path):
    path = tmp_path / "cache.sqlite"
    cache = SQLiteEvidenceCache(path)
    client = CaseContentClient(
        token="secret-token",
        base_url="https://example.test",
        cache=cache,
        retries=1,
        max_network_calls=1,
        request_interval_seconds=0,
        session=FakeSession([requests.Timeout("network timeout")]),
        sleep=lambda _: None,
        clock=lambda: 1.0,
    )
    with pytest.raises(requests.Timeout):
        client.retrieve("case_1", "sensitive query", query_type="original")
    row = cache.connection.execute(
        "SELECT query_type, query_hash, exception_type FROM api_call_log"
    ).fetchone()
    assert row == (
        "original",
        cache_key("case_1", "sensitive query"),
        "Timeout",
    )
    database_bytes = path.read_bytes()
    assert b"secret-token" not in database_bytes
    assert b"sensitive query" not in database_bytes
    cache.close()


def test_malformed_success_response_is_not_cached(tmp_path):
    cache = SQLiteEvidenceCache(tmp_path / "cache.sqlite")
    client = CaseContentClient(
        token="test-token",
        base_url="https://example.test",
        cache=cache,
        retries=1,
        max_network_calls=1,
        request_interval_seconds=0,
        session=FakeSession(
            [FakeResponse(200, {"results": [{"text": "missing id"}]})]
        ),
        sleep=lambda _: None,
        clock=lambda: 1.0,
    )
    with pytest.raises(ValueError, match="non-empty opaque chunk_id"):
        client.retrieve("case_1", "query", query_type="original")
    assert not cache.contains("case_1", "query")
    assert cache.known_cumulative_attempts() == 1
    assert cache.run_attempt_stats("unscoped")["successful_calls"] == 0
    cache.close()


def test_api_plan_counts_cache_hits_without_network_calls(tmp_path):
    cache = SQLiteEvidenceCache(tmp_path / "cache.sqlite")
    case = InferenceCase("case_1", "Tranh chấp hợp đồng")
    first_query = EvidenceQueryGenerator().generate(case.case_query, 2)[0]
    cache.put(case.case_id, first_query.text, [])
    report = build_api_plan(
        [case], cache, max_queries=2, approved_max_network_calls=1
    )
    assert report["logical_queries"] == 2
    assert report["cache_hits"] == 1
    assert report["cache_misses"] == 1
    assert report["approved_max_network_calls"] == 1
    assert report["known_local_cumulative_attempts"] == 0
    cache.close()


def test_cache_only_client_returns_hits_and_empty_misses_without_http(tmp_path):
    cache = SQLiteEvidenceCache(tmp_path / "cache.sqlite")
    cache.put("case_1", "known", [{"chunk_id": "opaque", "text": "x"}])
    events = []
    client = CachedCaseContentClient(cache, progress_callback=events.append)
    assert client.retrieve("case_1", "known", "court_decision")[0]["chunk_id"] == "opaque"
    assert client.retrieve("case_1", "missing", "original") == []
    assert client.network_calls == 0
    assert client.cache_hits == 1
    assert client.cache_misses == 1
    assert [event["status"] for event in events] == ["cache_hit", "cache_miss"]
    cache.close()


def test_success_response_and_attempt_are_backed_up_atomically(tmp_path):
    cache = SQLiteEvidenceCache(tmp_path / "local.sqlite")
    backup_path = tmp_path / "drive" / "case_api.sqlite"
    client = CaseContentClient(
        token="test-token",
        base_url="https://example.test",
        cache=cache,
        retries=1,
        max_network_calls=1,
        request_interval_seconds=0,
        session=FakeSession(
            [FakeResponse(200, {"results": [{"chunk_id": "opaque", "text": "x"}]})]
        ),
        sleep=lambda _: None,
        clock=lambda: 1.0,
        backup_path=backup_path,
    )
    client.retrieve("case_1", "query", query_type="original")
    backup = SQLiteEvidenceCache(backup_path)
    assert backup.contains("case_1", "query")
    assert backup.known_cumulative_attempts() == 1
    backup.close()
    cache.close()


def test_backup_failure_stops_before_retry(tmp_path, monkeypatch):
    cache = SQLiteEvidenceCache(tmp_path / "local.sqlite")
    session = FakeSession(
        [FakeResponse(503, {}), FakeResponse(200, {"results": []})]
    )
    monkeypatch.setattr(
        cache,
        "backup_to",
        lambda path: (_ for _ in ()).throw(OSError("drive unavailable")),
    )
    client = CaseContentClient(
        token="test-token",
        base_url="https://example.test",
        cache=cache,
        retries=2,
        max_network_calls=2,
        request_interval_seconds=0,
        session=session,
        sleep=lambda _: None,
        clock=lambda: 1.0,
        backup_path=tmp_path / "drive.sqlite",
    )
    with pytest.raises(OSError, match="drive unavailable"):
        client.retrieve("case_1", "query", query_type="original")
    assert len(session.calls) == 1
    assert cache.known_cumulative_attempts() == 1
    assert cache.has_pending_backup()
    cache.close()


def test_new_live_client_repairs_pending_backup_before_cache_hit(
    tmp_path, monkeypatch
):
    cache = SQLiteEvidenceCache(tmp_path / "local.sqlite")
    backup_path = tmp_path / "drive" / "case_api.sqlite"
    first_session = FakeSession(
        [FakeResponse(200, {"results": [{"chunk_id": "opaque", "text": "x"}]})]
    )
    first_client = CaseContentClient(
        token="test-token",
        base_url="https://example.test",
        cache=cache,
        retries=1,
        max_network_calls=1,
        request_interval_seconds=0,
        session=first_session,
        sleep=lambda _: None,
        clock=lambda: 1.0,
        backup_path=backup_path,
    )
    original_backup = cache.backup_to
    monkeypatch.setattr(
        cache,
        "backup_to",
        lambda path: (_ for _ in ()).throw(OSError("drive unavailable")),
    )
    with pytest.raises(OSError, match="drive unavailable"):
        first_client.retrieve("case_1", "query", query_type="original")
    assert cache.has_pending_backup()

    monkeypatch.setattr(cache, "backup_to", original_backup)
    resumed_session = FakeSession([])
    resumed_client = CaseContentClient(
        token="test-token",
        base_url="https://example.test",
        cache=cache,
        retries=1,
        max_network_calls=1,
        request_interval_seconds=0,
        session=resumed_session,
        sleep=lambda _: None,
        clock=lambda: 1.0,
        backup_path=backup_path,
    )
    result = resumed_client.retrieve("case_1", "query", query_type="original")
    assert result[0]["chunk_id"] == "opaque"
    assert resumed_session.calls == []
    assert not cache.has_pending_backup()
    backup = SQLiteEvidenceCache(backup_path)
    assert backup.contains("case_1", "query")
    assert not backup.has_pending_backup()
    backup.close()
    cache.close()
