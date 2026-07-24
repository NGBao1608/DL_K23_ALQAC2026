from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class OutcomeLabel(str, Enum):
    A_WIN = "A_WIN"
    PARTIAL_A_WIN = "PARTIAL_A_WIN"
    PARTIAL_B_WIN = "PARTIAL_B_WIN"
    B_WIN = "B_WIN"


@dataclass(frozen=True, slots=True)
class InferenceCase:
    case_id: str
    case_query: str


@dataclass(frozen=True, slots=True)
class PublicGold:
    case_id: str
    verdict_label: OutcomeLabel
    law_evidence: tuple[tuple[str, int], ...] = ()
    case_evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LawArticle:
    law_id: str
    aid: int
    article_number: int
    text: str

    @property
    def key(self) -> str:
        return f"{self.law_id}:{self.aid}"


@dataclass(frozen=True, slots=True)
class CaseEvidence:
    chunk_id: str
    text: str
    score: float
    query_type: str = "unknown"


@dataclass(frozen=True, slots=True)
class LawEvidence:
    law_id: str
    aid: int
    text: str
    score: float
    article_number: int | None = None


@dataclass(slots=True)
class PredictionResult:
    case_id: str
    prediction: OutcomeLabel | None
    case_evidence: list[CaseEvidence] = field(default_factory=list)
    law_evidence: list[LawEvidence] = field(default_factory=list)
    reasoning: str | None = None
    raw_output: str | None = None
    api_calls: int = 0
    latency_seconds: float = 0.0
    status: str = "completed"
    error: str | None = None
    prediction_attempts: int = 1
    prediction_failure_types: list[str] = field(default_factory=list)
