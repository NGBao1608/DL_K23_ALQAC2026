import pytest

from alqac2026.case_retrieval import (
    CaseContentClient,
    EvidenceQueryGenerator,
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


def test_generator_core_templates_first_at_small_budget():
    gen = EvidenceQueryGenerator()
    queries = gen.generate("Tranh chấp đất đai giữa ông A và bà B.", max_queries=6)
    assert len(queries) == 6
    assert [q.query_type for q in queries[:6]] == [
        "court_decision",
        "accepted_claim",
        "rejected_claim",
        "court_reasoning",
        "applied_law",
        "court_fee",
    ]


def test_generator_scales_with_budget_and_stays_unique():
    gen = EvidenceQueryGenerator()
    query = (
        "Ông Trần H khởi kiện bà Hà Thị T tranh chấp hợp đồng chuyển nhượng "
        "quyền sử dụng đất thửa 238, tờ bản đồ số 07, trị giá 600.000.000 đồng."
    )
    queries = gen.generate(query, max_queries=40)
    texts = [q.query_type + "|" + q.text for q in queries]
    assert len(texts) == len(set(texts))  # no duplicates
    assert len(queries) >= 20
    kinds = {q.query_type for q in queries}
    assert "parcel" in kinds and "amount" in kinds and "party_name" in kinds


def test_generator_emits_ruling_rescue_queries_at_budget_14():
    gen = EvidenceQueryGenerator()
    queries = gen.generate("Tranh chấp hợp đồng vay tài sản.", max_queries=14)
    kinds = [q.query_type for q in queries]
    rescue = [k for k in kinds if k.startswith("rescue_")]
    assert len(rescue) == 6, kinds
    # Positions 9-14 so a budget of 14 reproduces exactly the cached query set.
    assert kinds[8:14] == [
        "rescue_decision",
        "rescue_reject",
        "rescue_order",
        "rescue_reasoning",
        "rescue_dismiss",
        "rescue_therefore",
    ]


def test_case_specific_queries_need_budget_above_14():
    gen = EvidenceQueryGenerator()
    query = "Ông A tranh chấp thửa 238 trị giá 600.000.000 đồng với bà B."
    assert gen._specific_queries(query), "extractor must fire on a rich query"
    small = {q.query_type for q in gen.generate(query, max_queries=14)}
    assert not (small & {"parcel", "amount", "party_name"})
    large = {q.query_type for q in gen.generate(query, max_queries=24)}
    assert large & {"parcel", "amount", "party_name"}


def test_generator_respects_budget_cap():
    gen = EvidenceQueryGenerator()
    assert len(gen.generate("Tranh chấp vay tiền.", max_queries=3)) == 3


def test_saturation_early_stop_and_cap():
    from alqac2026.case_retrieval import CaseEvidenceRetriever
    from alqac2026.schemas import InferenceCase

    class DupClient:
        # New chunk for the first 3 network calls, duplicates afterwards.
        def __init__(self):
            self.network_calls = 0
            self.cache_hits = 0

        def retrieve(self, case_id, query):
            self.network_calls += 1
            key = f"seg{self.network_calls}" if self.network_calls <= 3 else "seg_dup"
            return [{"chunk_id": key, "text": "x", "score": 1.0}]

    retriever = CaseEvidenceRetriever(
        DupClient(), max_queries=30, saturation_patience=4, max_network_calls=40
    )
    evidence, calls = retriever.retrieve(
        InferenceCase("c1", "Nguyên đơn tranh chấp hợp đồng vay với bị đơn.")
    )
    # 3 unique + seg_dup = 4 chunks; stops after 4 consecutive no-gain calls.
    assert len(evidence) == 4
    assert calls == 8

    class FreshClient:
        def __init__(self):
            self.network_calls = 0
            self.cache_hits = 0

        def retrieve(self, case_id, query):
            self.network_calls += 1
            return [{"chunk_id": f"s{self.network_calls}", "text": "x", "score": 1.0}]

    capped = CaseEvidenceRetriever(
        FreshClient(), max_queries=30, saturation_patience=4, max_network_calls=10
    )
    _, calls2 = capped.retrieve(
        InferenceCase("c2", "Nguyên đơn tranh chấp đất thửa 5 với bị đơn ông A bà B.")
    )
    assert calls2 == 10  # hard ceiling respected


def test_law_query_strips_parties_and_kinship():
    from alqac2026.pipeline import _law_query

    masked = _law_query(
        "Ông Trần H và vợ là bà Nguyễn Thị B tranh chấp quyền sử dụng đất với con ông C."
    )
    assert "Trần H" not in masked and "Nguyễn Thị B" not in masked
    assert "vợ" not in masked and "con" not in masked
    assert "quyền sử dụng đất" in masked


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
                        {"chunk_id": "case_1_seg_2", "score": 1.0, "text": "x"}
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


def test_api_retries_transient_404_then_succeeds(tmp_path):
    cache = SQLiteEvidenceCache(tmp_path / "cache.sqlite")
    session = FakeSession(
        [
            FakeResponse(404, {}, text="ngrok tunnel blip"),
            FakeResponse(
                200,
                {"results": [{"chunk_id": "case_1_seg_ab", "score": 1.0, "text": "x"}]},
            ),
        ]
    )
    client = CaseContentClient(
        token="test-token",
        base_url="https://example.test",
        cache=cache,
        retries=4,
        request_interval_seconds=0,
        session=session,
        sleep=lambda _: None,
        clock=lambda: 1.0,
    )
    results = client.retrieve("case_1", "query")
    assert results == [{"chunk_id": "case_1_seg_ab", "score": 1.0, "text": "x"}]
    assert client.network_calls == 2
    cache.close()


def test_api_persistent_404_raises(tmp_path):
    cache = SQLiteEvidenceCache(tmp_path / "cache.sqlite")
    session = FakeSession([FakeResponse(404, {}, text="not found") for _ in range(2)])
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
    with pytest.raises(RuntimeError):
        client.retrieve("case_1", "query")
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
