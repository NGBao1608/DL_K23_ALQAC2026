from alqac2026.case_retrieval import (
    CaseContentClient,
    SQLiteEvidenceCache,
    cache_key,
)
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
        return next(self.responses)


def test_rrf_is_deterministic_and_rewards_overlap():
    ranked = reciprocal_rank_fusion([[1, 2, 3], [3, 2, 4]], rrf_k=60)
    identifiers = [identifier for identifier, _ in ranked]
    assert identifiers[:2] == [3, 2]


def test_cache_key_normalizes_whitespace_and_case():
    assert cache_key("case_1", "  Nhận định   Tòa án ") == cache_key(
        "case_1", "nhận định tòa án"
    )


def test_api_retry_then_cache_hit(tmp_path):
    cache = SQLiteEvidenceCache(tmp_path / "cache.sqlite")
    session = FakeSession(
        [
            FakeResponse(503, {}),
            FakeResponse(
                200,
                {
                    "results": [
                        {"chunk_id": "case_1_chunk_2", "score": 1.0, "text": "x"}
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
    assert "response=Forbidden for <redacted>" in message
    cache.close()
