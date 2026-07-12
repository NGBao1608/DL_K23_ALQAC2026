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
    TEMPLATES = (
        ("quyết định của tòa án tuyên xử", "court_decision"),
        ("chấp nhận yêu cầu khởi kiện của nguyên đơn", "accepted_claim"),
        ("không chấp nhận yêu cầu của nguyên đơn", "rejected_claim"),
        ("nhận định của hội đồng xét xử", "court_reasoning"),
        ("áp dụng điều luật", "applied_law"),
        ("nghĩa vụ chịu án phí dân sự sơ thẩm", "court_fee"),
    )

    def generate(self, case_query: str, max_queries: int = 8) -> list[EvidenceQuery]:
        queries = [EvidenceQuery(text, query_type) for text, query_type in self.TEMPLATES]
        match = re.search(r"tranh chấp[^.,;?]*", case_query, flags=re.IGNORECASE)
        if match:
            queries.append(EvidenceQuery(match.group(0).strip(), "dispute_type"))
        queries.append(
            EvidenceQuery(re.sub(r"\s+", " ", case_query).strip()[:500], "original")
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

    def retrieve(self, case_id: str, query: str) -> list[dict]:
        cached = self.cache.get(case_id, query)
        if cached is not None:
            self.cache_hits += 1
            return cached

        payload = {"case_id": case_id, "query": query}
        headers = {"X-API-Key": self.token, "Content-Type": "application/json"}
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
                raise PermissionError("Case API rejected ALQAC_TEAM_TOKEN (403)")
            if response.status_code == 422:
                raise ValueError(f"Malformed Case API request: {payload}")
            if response.status_code == 429 or 500 <= response.status_code < 600:
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
    ):
        self.client = client
        self.query_generator = query_generator or EvidenceQueryGenerator()
        self.max_queries = max_queries

    def retrieve(self, case: InferenceCase) -> tuple[list[CaseEvidence], int]:
        before = self.client.network_calls
        evidence: dict[str, CaseEvidence] = {}
        for query in self.query_generator.generate(case.case_query, self.max_queries):
            for hit in self.client.retrieve(case.case_id, query.text):
                item = CaseEvidence(
                    chunk_id=str(hit["chunk_id"]),
                    text=str(hit.get("text", "")),
                    score=float(hit.get("score", 0.0)),
                    query_type=query.query_type,
                )
                previous = evidence.get(item.chunk_id)
                if previous is None or item.score > previous.score:
                    evidence[item.chunk_id] = item
        return list(evidence.values()), self.client.network_calls - before

