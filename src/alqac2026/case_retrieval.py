from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

from .schemas import CaseEvidence, InferenceCase


API_VERSION = "retrieve-v1"


class ApiBudgetExceeded(RuntimeError):
    """Raised before an HTTP request would exceed the approved run budget."""


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
        if max_queries < 0:
            raise ValueError("max_queries must be non-negative")
        original = EvidenceQuery(normalize_query(case_query)[:500], "original")
        queries = [
            EvidenceQuery(*self.TEMPLATES[0]),
            original,
            *(
                EvidenceQuery(text, query_type)
                for text, query_type in self.TEMPLATES[1:]
            ),
        ]
        match = re.search(r"tranh chấp[^.,;?]*", case_query, flags=re.IGNORECASE)
        if match:
            queries.append(EvidenceQuery(match.group(0).strip(), "dispute_type"))
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
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS api_call_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                run_key TEXT,
                case_id TEXT NOT NULL,
                query_type TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                status_code INTEGER,
                exception_type TEXT,
                success INTEGER NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            INSERT OR IGNORE INTO cache_state (key, value)
            VALUES ('backup_pending', '0')
            """
        )
        columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(api_call_log)")
        }
        if "run_key" not in columns:
            self.connection.execute("ALTER TABLE api_call_log ADD COLUMN run_key TEXT")
        self.connection.commit()

    def get(self, case_id: str, query: str) -> list[dict] | None:
        key = cache_key(case_id, query)
        row = self.connection.execute(
            "SELECT response_json FROM evidence_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, case_id: str, query: str, response: list[dict]) -> None:
        key = cache_key(case_id, query)
        with self.connection:
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

    def contains(self, case_id: str, query: str) -> bool:
        key = cache_key(case_id, query)
        row = self.connection.execute(
            "SELECT 1 FROM evidence_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        return row is not None

    def log_attempt(
        self,
        case_id: str,
        query: str,
        query_type: str,
        *,
        run_key: str = "unscoped",
        status_code: int | None = None,
        exception_type: str | None = None,
        success: bool = False,
    ) -> None:
        with self.connection:
            self._insert_attempt(
                case_id,
                query,
                query_type,
                run_key=run_key,
                status_code=status_code,
                exception_type=exception_type,
                success=success,
            )
            self._set_backup_pending(True)

    def record_success(
        self,
        case_id: str,
        query: str,
        query_type: str,
        response: list[dict],
        *,
        run_key: str = "unscoped",
        status_code: int = 200,
    ) -> None:
        """Commit the successful call ledger and reusable response atomically."""
        key = cache_key(case_id, query)
        with self.connection:
            self._insert_attempt(
                case_id,
                query,
                query_type,
                run_key=run_key,
                status_code=status_code,
                success=True,
            )
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
            self._set_backup_pending(True)

    def _insert_attempt(
        self,
        case_id: str,
        query: str,
        query_type: str,
        *,
        run_key: str,
        status_code: int | None = None,
        exception_type: str | None = None,
        success: bool = False,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO api_call_log
            (created_at, run_key, case_id, query_type, query_hash, status_code,
             exception_type, success)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                run_key,
                case_id,
                query_type,
                cache_key(case_id, query),
                status_code,
                exception_type,
                int(success),
            ),
        )

    def integrity_check(self) -> None:
        row = self.connection.execute("PRAGMA integrity_check").fetchone()
        if row is None or row[0] != "ok":
            raise ValueError(f"SQLite cache integrity check failed: {row}")

    def has_pending_backup(self) -> bool:
        row = self.connection.execute(
            "SELECT value FROM cache_state WHERE key = 'backup_pending'"
        ).fetchone()
        return row is not None and row[0] == "1"

    def _set_backup_pending(self, pending: bool) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO cache_state (key, value)
            VALUES ('backup_pending', ?)
            """,
            ("1" if pending else "0",),
        )

    def backup_to(self, destination: str | Path) -> Path:
        """Create a consistent SQLite backup and atomically publish it."""
        target = Path(destination)
        if target.resolve() == self.path.resolve():
            raise ValueError("SQLite backup destination must differ from the live DB")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_handle = tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        )
        temporary = Path(temporary_handle.name)
        temporary_handle.close()
        try:
            backup_connection = sqlite3.connect(temporary)
            try:
                self.connection.backup(backup_connection)
                backup_connection.execute(
                    """
                    INSERT OR REPLACE INTO cache_state (key, value)
                    VALUES ('backup_pending', '0')
                    """
                )
                backup_connection.commit()
            finally:
                backup_connection.close()
            verified = sqlite3.connect(temporary)
            try:
                row = verified.execute("PRAGMA integrity_check").fetchone()
            finally:
                verified.close()
            if row is None or row[0] != "ok":
                raise ValueError(f"SQLite backup integrity check failed: {row}")
            temporary.replace(target)
            with self.connection:
                self._set_backup_pending(False)
            return target
        finally:
            if temporary.exists():
                temporary.unlink()

    def known_cumulative_attempts(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM api_call_log").fetchone()
        return int(row[0])

    def run_attempt_stats(self, run_key: str) -> dict:
        rows = self.connection.execute(
            """
            SELECT case_id, COUNT(*), SUM(success)
            FROM api_call_log
            WHERE run_key = ?
            GROUP BY case_id
            """,
            (run_key,),
        ).fetchall()
        per_case = {
            str(case_id): {
                "network_attempts": int(attempts),
                "successful_calls": int(successes or 0),
            }
            for case_id, attempts, successes in rows
        }
        return {
            "network_attempts": sum(
                item["network_attempts"] for item in per_case.values()
            ),
            "successful_calls": sum(
                item["successful_calls"] for item in per_case.values()
            ),
            "per_case": per_case,
        }

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
        max_network_calls: int | None = None,
        run_key: str = "unscoped",
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        backup_path: str | Path | None = None,
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
        if max_network_calls is not None and max_network_calls < 0:
            raise ValueError("max_network_calls must be non-negative")
        self.max_network_calls = max_network_calls
        self.run_key = run_key
        self.session = session or requests.Session()
        self.sleep = sleep
        self.clock = clock
        self.progress_callback = progress_callback
        self.backup_path = Path(backup_path) if backup_path is not None else None
        self.cache.integrity_check()
        if self.backup_path is not None and self.cache.has_pending_backup():
            # Publish prior committed state before any cache hit or request.
            self.cache.backup_to(self.backup_path)
        self._last_call = 0.0
        existing_stats = self.cache.run_attempt_stats(run_key)
        self.network_calls = int(existing_stats["network_attempts"])
        self.successful_calls = int(existing_stats["successful_calls"])
        self.cache_hits = 0
        self.per_case_attempts: dict[str, int] = defaultdict(int)
        self.per_case_successes: dict[str, int] = defaultdict(int)
        self.per_case_cache_hits: dict[str, int] = defaultdict(int)

    def _emit_progress(self, **event: object) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(event)
        except Exception:
            # Observability must never interrupt a budgeted retrieval request.
            return

    def _throttle(self) -> None:
        remaining = self.request_interval_seconds - (self.clock() - self._last_call)
        if remaining > 0:
            self.sleep(remaining)

    def _claim_network_budget(self, case_id: str) -> None:
        if (
            self.max_network_calls is not None
            and self.network_calls >= self.max_network_calls
        ):
            raise ApiBudgetExceeded(
                "Approved Case Content API network-call budget exhausted before "
                f"requesting {case_id}: {self.max_network_calls}"
            )
        self.network_calls += 1
        self.per_case_attempts[case_id] += 1

    def _backup_cache(self) -> None:
        if self.backup_path is not None:
            self.cache.backup_to(self.backup_path)

    def retrieve(
        self, case_id: str, query: str, query_type: str = "unknown"
    ) -> list[dict]:
        cached = self.cache.get(case_id, query)
        if cached is not None:
            self.cache_hits += 1
            self.per_case_cache_hits[case_id] += 1
            self._emit_progress(
                status="cache_hit",
                case_id=case_id,
                query_type=query_type,
                result_count=len(cached),
            )
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
            self._claim_network_budget(case_id)
            network_attempt = self.network_calls
            self._emit_progress(
                status="request_started",
                case_id=case_id,
                query_type=query_type,
                logical_attempt=attempt + 1,
                network_attempt=network_attempt,
                max_network_calls=self.max_network_calls,
                timeout_seconds=self.timeout_seconds,
            )
            request_started = time.perf_counter()
            try:
                response = self.session.post(
                    f"{self.base_url}/retrieve",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as error:
                latency_seconds = round(time.perf_counter() - request_started, 3)
                self._last_call = self.clock()
                self.cache.log_attempt(
                    case_id,
                    query,
                    query_type,
                    run_key=self.run_key,
                    exception_type=type(error).__name__,
                )
                self._backup_cache()
                self._emit_progress(
                    status="request_exception",
                    case_id=case_id,
                    query_type=query_type,
                    logical_attempt=attempt + 1,
                    network_attempt=network_attempt,
                    exception_type=type(error).__name__,
                    latency_seconds=latency_seconds,
                )
                if attempt + 1 < self.retries:
                    retry_delay_seconds = self.request_interval_seconds * (2**attempt)
                    self._emit_progress(
                        status="retry_scheduled",
                        case_id=case_id,
                        query_type=query_type,
                        logical_attempt=attempt + 1,
                        next_logical_attempt=attempt + 2,
                        retry_delay_seconds=retry_delay_seconds,
                    )
                    self.sleep(retry_delay_seconds)
                    continue
                raise
            self._last_call = self.clock()
            latency_seconds = round(time.perf_counter() - request_started, 3)
            if response.status_code == 200:
                try:
                    payload_value = response.json()
                    if not isinstance(payload_value, dict):
                        raise ValueError("Case API response must be a JSON object")
                    results = payload_value.get("results", [])
                    if not isinstance(results, list):
                        raise ValueError("Case API results must be a list")
                    if any(
                        not isinstance(item, dict)
                        or not isinstance(item.get("chunk_id"), str)
                        or not item["chunk_id"].strip()
                        for item in results
                    ):
                        raise ValueError(
                            "Every Case API result must contain a non-empty opaque "
                            "chunk_id"
                        )
                except Exception as error:
                    self.cache.log_attempt(
                        case_id,
                        query,
                        query_type,
                        run_key=self.run_key,
                        status_code=response.status_code,
                        exception_type=type(error).__name__,
                    )
                    self._backup_cache()
                    self._emit_progress(
                        status="invalid_response",
                        case_id=case_id,
                        query_type=query_type,
                        logical_attempt=attempt + 1,
                        network_attempt=network_attempt,
                        status_code=response.status_code,
                        exception_type=type(error).__name__,
                        latency_seconds=latency_seconds,
                    )
                    raise
                self.cache.record_success(
                    case_id,
                    query,
                    query_type,
                    results,
                    run_key=self.run_key,
                    status_code=response.status_code,
                )
                self._backup_cache()
                self.successful_calls += 1
                self.per_case_successes[case_id] += 1
                self._emit_progress(
                    status="request_completed",
                    case_id=case_id,
                    query_type=query_type,
                    logical_attempt=attempt + 1,
                    network_attempt=network_attempt,
                    status_code=response.status_code,
                    result_count=len(results),
                    latency_seconds=latency_seconds,
                )
                return results
            self.cache.log_attempt(
                case_id,
                query,
                query_type,
                run_key=self.run_key,
                status_code=response.status_code,
            )
            self._backup_cache()
            self._emit_progress(
                status="http_error",
                case_id=case_id,
                query_type=query_type,
                logical_attempt=attempt + 1,
                network_attempt=network_attempt,
                status_code=response.status_code,
                latency_seconds=latency_seconds,
            )
            if response.status_code == 403:
                content_type = response.headers.get("Content-Type", "unknown")
                server = response.headers.get("Server", "unknown")
                request_id = response.headers.get(
                    "X-Request-ID", response.headers.get("X-Amzn-Trace-Id", "unknown")
                )
                ngrok_codes = sorted(set(re.findall(r"ERR_NGROK_\d+", response.text)))
                raise PermissionError(
                    "Case API rejected ALQAC_TEAM_TOKEN (403); "
                    f"url={response.url}; server={server}; content_type={content_type}; "
                    f"request_id={request_id}; ngrok_codes={ngrok_codes or 'none'}; "
                    "response_body=<redacted>"
                )
            if response.status_code == 422:
                raise ValueError("Malformed Case API request (422); payload redacted")
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt + 1 < self.retries:
                    retry_delay_seconds = self.request_interval_seconds * (2**attempt)
                    self._emit_progress(
                        status="retry_scheduled",
                        case_id=case_id,
                        query_type=query_type,
                        logical_attempt=attempt + 1,
                        next_logical_attempt=attempt + 2,
                        retry_delay_seconds=retry_delay_seconds,
                    )
                    self.sleep(retry_delay_seconds)
                    continue
            response.raise_for_status()
        raise RuntimeError(f"Case API failed after {self.retries} attempts: {case_id}")


class CachedCaseContentClient:
    """Read-only cache client used for zero-network model evaluation."""

    def __init__(
        self,
        cache: SQLiteEvidenceCache,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ):
        self.cache = cache
        self.progress_callback = progress_callback
        self.network_calls = 0
        self.successful_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.per_case_cache_hits: dict[str, int] = defaultdict(int)

    def _emit_progress(self, **event: object) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(event)
        except Exception:
            return

    def retrieve(
        self, case_id: str, query: str, query_type: str = "unknown"
    ) -> list[dict]:
        cached = self.cache.get(case_id, query)
        if cached is None:
            self.cache_misses += 1
            self._emit_progress(
                status="cache_miss",
                case_id=case_id,
                query_type=query_type,
                result_count=0,
            )
            return []
        self.cache_hits += 1
        self.per_case_cache_hits[case_id] += 1
        self._emit_progress(
            status="cache_hit",
            case_id=case_id,
            query_type=query_type,
            result_count=len(cached),
        )
        return cached


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
            for hit in self.client.retrieve(
                case.case_id, query.text, query_type=query.query_type
            ):
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


def build_api_plan(
    cases: list[InferenceCase],
    cache: SQLiteEvidenceCache,
    *,
    max_queries: int,
    approved_max_network_calls: int | None = None,
    query_generator: EvidenceQueryGenerator | None = None,
) -> dict:
    generator = query_generator or EvidenceQueryGenerator()
    per_case: dict[str, dict[str, int]] = {}
    logical_queries = 0
    cache_hits = 0
    for case in cases:
        case_queries = generator.generate(case.case_query, max_queries=max_queries)
        hits = sum(cache.contains(case.case_id, query.text) for query in case_queries)
        logical_queries += len(case_queries)
        cache_hits += hits
        per_case[case.case_id] = {
            "logical_queries": len(case_queries),
            "cache_hits": hits,
            "cache_misses": len(case_queries) - hits,
        }
    return {
        "logical_queries": logical_queries,
        "cache_hits": cache_hits,
        "cache_misses": logical_queries - cache_hits,
        "approved_max_network_calls": approved_max_network_calls,
        "known_local_cumulative_attempts": cache.known_cumulative_attempts(),
        "official_cumulative_calls": None,
        "per_case": per_case,
    }
