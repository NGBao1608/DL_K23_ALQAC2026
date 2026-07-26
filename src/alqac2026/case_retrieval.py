from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

from .schemas import CaseEvidence, InferenceCase


API_VERSION = "retrieve-v1"


@dataclass(frozen=True, slots=True)
class EvidenceQuery:
    text: str
    query_type: str


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip().lower()


def cache_key(case_id: str, query: str, api_version: str = API_VERSION) -> str:
    value = f"{api_version}\0{case_id}\0{normalize_query(query)}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class EvidenceQueryGenerator:
    # Decision- and law-bearing sections first: these drive outcome accuracy and
    # citation-based law evidence, so they must be issued even at small budgets.
    TEMPLATES = (
        ("quyết định của tòa án tuyên xử", "court_decision"),
        ("chấp nhận yêu cầu khởi kiện của nguyên đơn", "accepted_claim"),
        ("không chấp nhận yêu cầu của nguyên đơn", "rejected_claim"),
        ("nhận định của hội đồng xét xử", "court_reasoning"),
        ("áp dụng điều luật", "applied_law"),
        ("nghĩa vụ chịu án phí dân sự sơ thẩm", "court_fee"),
    )
    # Ruling-section rescue queries. Reworded phrasings of the decision section
    # that recover the ruling text for cases where the primary templates return no
    # decision segment. Issued after the primary templates (positions 9-14).
    RESCUE_TEMPLATES = (
        ("quyết định tuyên xử chấp nhận yêu cầu", "rescue_decision"),
        ("không chấp nhận toàn bộ yêu cầu khởi kiện", "rescue_reject"),
        ("buộc bị đơn phải trả cho nguyên đơn", "rescue_order"),
        ("nhận định của hội đồng xét xử xét thấy", "rescue_reasoning"),
        ("đình chỉ giải quyết yêu cầu", "rescue_dismiss"),
        ("vì các lẽ trên quyết định", "rescue_therefore"),
    )
    # Procuracy (VKS) recommendation queries. The API serves the VKS opinion, which
    # strongly predicts the verdict. Issued after the rescue templates (positions
    # 15-17) so smaller query budgets keep the higher-priority sections. See vks.py.
    VKS_TEMPLATES = (
        ("đại diện viện kiểm sát phát biểu quan điểm giải quyết vụ án", "vks_opinion"),
        ("kiểm sát viên đề nghị hội đồng xét xử chấp nhận yêu cầu khởi kiện", "vks_accept"),
        ("viện kiểm sát đề nghị bác hoặc chấp nhận một phần yêu cầu", "vks_partial"),
    )
    # Additional structural sections of a first-instance civil judgment. Each is
    # lexically distinct so it ranks a different segment first; near-synonymous
    # queries would retrieve overlapping segments and add little coverage.
    SECTION_TEMPLATES = (
        ("đơn khởi kiện và quá trình hòa giải", "filing"),
        ("lời trình bày của nguyên đơn", "plaintiff_statement"),
        ("yêu cầu phản tố của bị đơn", "counterclaim"),
        ("người có quyền lợi nghĩa vụ liên quan", "third_party"),
        ("lời khai của người làm chứng", "witness"),
        ("giấy chứng nhận quyền sử dụng đất", "land_certificate"),
        ("hợp đồng chuyển nhượng công chứng chứng thực", "notarized_contract"),
        ("kết quả xem xét thẩm định tại chỗ", "site_assessment"),
        ("kết quả định giá tài sản", "valuation"),
        ("hồ sơ địa chính đo đạc thửa đất", "cadastral"),
        ("ý kiến của đại diện viện kiểm sát", "procuracy"),
        ("hợp đồng vô hiệu hậu quả pháp lý", "contract_invalid"),
        ("công sức đóng góp duy trì tôn tạo tài sản", "contribution"),
        ("lãi suất chậm trả nợ gốc", "interest"),
        ("quyền kháng cáo bản án sơ thẩm", "appeal"),
    )

    # High-IDF, case-specific spans that pull the one segment mentioning them.
    _SPECIFIC_PATTERNS = (
        (r"thửa\s+(?:đất\s+)?(?:số\s+)?\d+[a-zA-Z]?", "parcel"),
        (r"tờ\s+bản\s+đồ\s+(?:số\s+)?\d+", "map_sheet"),
        (r"[\d][\d.,]*\s*(?:m2|m²|mét vuông)", "area"),
        (r"[\d][\d.,]*\s*đồng", "amount"),
        (r"[Gg]iấy\s+chứng\s+nhận[^.,;:?]{0,40}", "certificate"),
        (r"[Hh]ợp\s+đồng\s+[^.,;:?]{0,40}", "contract_ref"),
    )
    _PARTY_PATTERN = re.compile(
        r"(?:ông|bà|anh|chị|Ông|Bà|Anh|Chị)\s+"
        r"([A-ZÀ-Ỹ][\wÀ-ỹ]*(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ]*){0,3})"
    )

    def _specific_queries(self, case_query: str) -> list[EvidenceQuery]:
        found: list[EvidenceQuery] = []
        seen: set[str] = set()
        for pattern, query_type in self._SPECIFIC_PATTERNS:
            for match in re.finditer(pattern, case_query):
                text = re.sub(r"\s+", " ", match.group(0)).strip()
                key = normalize_query(text)
                if key and key not in seen:
                    seen.add(key)
                    found.append(EvidenceQuery(text, query_type))
        for match in self._PARTY_PATTERN.finditer(case_query):
            text = re.sub(r"\s+", " ", match.group(1)).strip()
            key = normalize_query(text)
            if key and key not in seen and len(text) >= 3:
                seen.add(key)
                found.append(EvidenceQuery(text, "party_name"))
        return found

    def generate(self, case_query: str, max_queries: int = 8) -> list[EvidenceQuery]:
        queries = [EvidenceQuery(text, query_type) for text, query_type in self.TEMPLATES]
        match = re.search(r"tranh chấp[^.,;?]*", case_query, flags=re.IGNORECASE)
        if match:
            queries.append(EvidenceQuery(match.group(0).strip(), "dispute_type"))
        # The first 8 queries (6 templates + dispute + original) are the highest
        # priority; the diverse spans below only add coverage at larger budgets.
        queries.append(
            EvidenceQuery(re.sub(r"\s+", " ", case_query).strip()[:500], "original")
        )
        # Positions 9-14: ruling rescue.
        queries.extend(
            EvidenceQuery(text, query_type) for text, query_type in self.RESCUE_TEMPLATES
        )
        # Positions 15-17: procuracy (VKS) recommendation — strong outcome signal.
        queries.extend(
            EvidenceQuery(text, query_type) for text, query_type in self.VKS_TEMPLATES
        )
        queries.extend(self._specific_queries(case_query))
        queries.extend(
            EvidenceQuery(text, query_type) for text, query_type in self.SECTION_TEMPLATES
        )
        result: list[EvidenceQuery] = []
        seen: set[str] = set()
        for query in queries:
            normalized = normalize_query(query.text)
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(query)
            if len(result) >= max_queries:
                break
        return result


class SQLiteEvidenceCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence_cache (
                cache_key TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                query TEXT NOT NULL,
                api_version TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self.connection.commit()

    def get(self, case_id: str, query: str) -> list[dict] | None:
        key = cache_key(case_id, query)
        row = self.connection.execute(
            "SELECT response_json FROM evidence_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, case_id: str, query: str, response: list[dict]) -> None:
        key = cache_key(case_id, query)
        self.connection.execute(
            """
            INSERT OR REPLACE INTO evidence_cache
            (cache_key, case_id, query, api_version, response_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                case_id,
                normalize_query(query),
                API_VERSION,
                json.dumps(response, ensure_ascii=False),
                time.time(),
            ),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


class CaseContentClient:
    def __init__(
        self,
        token: str,
        base_url: str,
        cache: SQLiteEvidenceCache,
        request_interval_seconds: float = 5.0,
        timeout_seconds: int = 30,
        retries: int = 4,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        token = token.strip()
        if not token:
            raise ValueError("ALQAC_TEAM_TOKEN is required")
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        self.request_interval_seconds = request_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.session = session or requests.Session()
        self.sleep = sleep
        self.clock = clock
        self._last_call = 0.0
        self.network_calls = 0
        self.cache_hits = 0

    def _throttle(self) -> None:
        remaining = self.request_interval_seconds - (self.clock() - self._last_call)
        if remaining > 0:
            self.sleep(remaining)

    def _rate_limit_pause(self, response: object, attempt: int) -> float:
        header = getattr(response, "headers", {}).get("Retry-After")
        if header:
            try:
                return max(float(header), self.request_interval_seconds)
            except (TypeError, ValueError):
                pass
        return min(60.0, 15.0 * (2**attempt))

    def retrieve(self, case_id: str, query: str) -> list[dict]:
        cached = self.cache.get(case_id, query)
        if cached is not None:
            self.cache_hits += 1
            return cached

        payload = {"case_id": case_id, "query": query}
        headers = {
            "X-API-Key": self.token,
            "Content-Type": "application/json",
            # The official API is exposed through a free ngrok tunnel. This header
            # prevents ngrok's browser-warning HTML from intercepting API requests.
            "ngrok-skip-browser-warning": "alqac2026-api-client",
        }
        for attempt in range(self.retries):
            self._throttle()
            response = self.session.post(
                f"{self.base_url}/retrieve",
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            self._last_call = self.clock()
            self.network_calls += 1
            if response.status_code == 200:
                results = response.json().get("results", [])
                if not isinstance(results, list):
                    raise ValueError("Case API results must be a list")
                self.cache.put(case_id, query, results)
                return results
            if response.status_code == 403:
                content_type = response.headers.get("Content-Type", "unknown")
                server = response.headers.get("Server", "unknown")
                request_id = response.headers.get(
                    "X-Request-ID", response.headers.get("X-Amzn-Trace-Id", "unknown")
                )
                ngrok_codes = sorted(set(re.findall(r"ERR_NGROK_\d+", response.text)))
                response_preview = re.sub(r"\s+", " ", response.text).strip()[:300]
                if self.token:
                    response_preview = response_preview.replace(self.token, "<redacted>")
                raise PermissionError(
                    "Case API rejected ALQAC_TEAM_TOKEN (403); "
                    f"url={response.url}; server={server}; content_type={content_type}; "
                    f"request_id={request_id}; ngrok_codes={ngrok_codes or 'none'}; "
                    f"response={response_preview or '<empty>'}"
                )
            if response.status_code == 422:
                raise ValueError(f"Malformed Case API request: {payload}")
            # 429 (rate limit) needs a much longer pause than 5s spacing: the free
            # ngrok tunnel enforces a cooldown of tens of seconds. Honor Retry-After
            # when present, else back off 15s→30s→60s.
            if response.status_code == 429:
                if attempt + 1 < self.retries:
                    self.sleep(self._rate_limit_pause(response, attempt))
                    continue
            # 404 is treated as transient: the tunnel intermittently returns "Not
            # Found" for a live endpoint. Retry it (and 5xx) with backoff so a single
            # blip cannot abort a long, mostly-cached run. If it persists past all
            # retries it falls through to raise_for_status(), so a genuine outage
            # still fails loudly and resumably (never silently empty).
            elif response.status_code == 404 or 500 <= response.status_code < 600:
                if attempt + 1 < self.retries:
                    self.sleep(self.request_interval_seconds * (2**attempt))
                    continue
            response.raise_for_status()
        raise RuntimeError(f"Case API failed after {self.retries} attempts: {case_id}")


class CaseEvidenceRetriever:
    def __init__(
        self,
        client: CaseContentClient,
        query_generator: EvidenceQueryGenerator | None = None,
        max_queries: int = 8,
        saturation_patience: int = 4,
        max_network_calls: int = 40,
    ):
        self.client = client
        self.query_generator = query_generator or EvidenceQueryGenerator()
        self.max_queries = max_queries
        # Saturation early-stop: issue queries in priority order and stop once
        # `saturation_patience` consecutive *network* queries add no new chunk, or a
        # hard per-case ceiling of `max_network_calls` (~2n) is hit. Cached queries
        # are always issued (free) and never trip the stop. This lets the query bank
        # be large for coverage without wasting calls on exhausted-pool cases.
        self.saturation_patience = saturation_patience
        self.max_network_calls = max_network_calls

    def retrieve(self, case: InferenceCase) -> tuple[list[CaseEvidence], int]:
        before = self.client.network_calls
        evidence: dict[str, CaseEvidence] = {}
        no_gain = 0
        for query in self.query_generator.generate(case.case_query, self.max_queries):
            if self.client.network_calls - before >= self.max_network_calls:
                break  # hard per-case call ceiling
            net_before = self.client.network_calls
            gained = 0
            for hit in self.client.retrieve(case.case_id, query.text):
                item = CaseEvidence(
                    chunk_id=str(hit["chunk_id"]),
                    text=str(hit.get("text", "")),
                    score=float(hit.get("score", 0.0)),
                    query_type=query.query_type,
                )
                previous = evidence.get(item.chunk_id)
                if previous is None:
                    gained += 1
                if previous is None or item.score > previous.score:
                    evidence[item.chunk_id] = item
            if self.client.network_calls > net_before:  # this query hit the network
                no_gain = 0 if gained else no_gain + 1
                if no_gain >= self.saturation_patience:
                    break
        return list(evidence.values()), self.client.network_calls - before
