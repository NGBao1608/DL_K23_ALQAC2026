"""
Client for the ALQAC 2026 *Case Evidence Retrieval API*.

Base URL : https://alqac-api.ngrok.pro
Auth     : header  X-API-Key: <team token>   (same token as leaderboard)
Endpoint : POST /retrieve   body {"query": str, "case_id": str}
Response : {"results": [{"score": float, "text": str, "chunk_id": str}]}  (exactly 1)

Why this exists: in the official (private) phase you can't read case files directly.
You issue keyword queries per case to pull the most relevant segment, and collect the
`chunk_id`s as `case_evidence` for your submission.

Scoring caution (Penalized Case Recall = 20%): evidence recall is multiplied by an
API-efficiency factor — full credit up to 2*n calls per case, decaying to 0 at 5*n
(n = number of segments in that case). So gather thoroughly but economically.

Rate limit: 1 request / 5 seconds per team -> we self-throttle and back off on 429.
"""
from __future__ import annotations
import os
import re
import time

import requests

API_BASE = os.getenv("ALQAC_API_BASE_RETRIEVE", "https://alqac-api.ngrok.pro").rstrip("/")
RATE_LIMIT_SECONDS = 5.0


class EvidenceRetriever:
    def __init__(self, token: str | None = None, base: str = API_BASE,
                 min_interval: float = RATE_LIMIT_SECONDS, timeout: int = 30):
        self.token = token or os.getenv("ALQAC_TEAM_TOKEN")
        if not self.token or "xxxx" in self.token:
            raise ValueError("Set ALQAC_TEAM_TOKEN (the team secret token).")
        self.base = base.rstrip("/")
        self.min_interval = min_interval
        self.timeout = timeout
        self._last_call = 0.0
        self.n_calls = 0  # track total calls for the efficiency penalty

    def _throttle(self):
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    def retrieve(self, query: str, case_id: str, max_retries: int = 4) -> list[dict]:
        """One /retrieve call -> list with (at most) one hit dict."""
        headers = {"X-API-Key": self.token, "Content-Type": "application/json"}
        payload = {"query": query, "case_id": case_id}
        for attempt in range(max_retries):
            self._throttle()
            r = requests.post(f"{self.base}/retrieve", headers=headers,
                              json=payload, timeout=self.timeout)
            self._last_call = time.time()
            self.n_calls += 1
            if r.status_code == 200:
                return r.json().get("results", [])
            if r.status_code == 429:                     # rate limited
                time.sleep(self.min_interval); continue
            if r.status_code == 503:                     # temp unavailable
                time.sleep(self.min_interval * (attempt + 1)); continue
            if r.status_code == 403:
                raise PermissionError("403: invalid/missing X-API-Key.")
            if r.status_code == 422:
                raise ValueError(f"422: malformed request {payload}")
            r.raise_for_status()
        return []


# Turn a case_query into a small set of evidence-seeking sub-queries.
# Keep the count low — it directly drives the API-efficiency penalty.
_GENERIC_ASPECTS = [
    "yêu cầu khởi kiện của nguyên đơn",
    "ý kiến trình bày của bị đơn",
    "nhận định của Hội đồng xét xử",
    "quyết định của Tòa án",
    "người có quyền lợi nghĩa vụ liên quan",
]


def make_subqueries(case_query: str, max_q: int = 5) -> list[str]:
    qs = []
    # 1) the case query itself (best single signal)
    qs.append(re.sub(r"\s+", " ", case_query)[:200])
    # 2) the dispute-type phrase, if present ("tranh chấp ...")
    m = re.search(r"tranh chấp[^.,;]*", case_query, flags=re.IGNORECASE)
    if m:
        qs.append(m.group(0).strip())
    # 3) generic procedural aspects useful for outcome prediction
    qs.extend(_GENERIC_ASPECTS)
    # dedupe, cap
    seen, out = set(), []
    for q in qs:
        k = q.lower()
        if k and k not in seen:
            seen.add(k); out.append(q)
        if len(out) >= max_q:
            break
    return out


def gather_evidence(retriever: EvidenceRetriever, case_id: str, case_query: str,
                    max_calls: int = 5) -> dict[str, dict]:
    """Issue several sub-queries; return {chunk_id: {score, text}} (deduped)."""
    evidence: dict[str, dict] = {}
    for q in make_subqueries(case_query, max_q=max_calls):
        for hit in retriever.retrieve(q, case_id):
            cid = hit["chunk_id"]
            # keep the higher-scoring text if a chunk recurs
            if cid not in evidence or hit["score"] > evidence[cid]["score"]:
                evidence[cid] = {"score": hit["score"], "text": hit["text"]}
    return evidence


if __name__ == "__main__":
    # Connectivity smoke test — uses the doc's example case_id.
    from dotenv import load_dotenv
    load_dotenv()
    rc = EvidenceRetriever()
    ev = gather_evidence(rc, case_id="case_1087_0037",
                         case_query="tranh chấp quyền sử dụng đất", max_calls=3)
    print(f"calls={rc.n_calls}  unique chunks={len(ev)}")
    for cid, h in ev.items():
        print(" ", cid, round(h["score"], 3), h["text"][:80])
