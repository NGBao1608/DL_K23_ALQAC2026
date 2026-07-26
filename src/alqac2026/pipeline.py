from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .citations import extract_law_citations, procedural_prior_evidence
from .config import write_json
from .schemas import (
    CaseEvidence,
    InferenceCase,
    LawEvidence,
    OutcomeLabel,
    PredictionResult,
)


# Honorific + name and kinship terms bias BM25 toward the Marriage & Family law
# (whose articles are saturated with them). Strip them so the law query keeps the
# legal issue, not the parties. Used only for the BM25 fallback; citation-based
# law evidence does not go through here.
_HONORIFIC_NAME = re.compile(
    r"\b(?:ông|bà|anh|chị|cụ|bé|cháu)\s+"
    r"[A-ZÀ-Ỹ][\wÀ-ỹ]*(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ]*){0,3}",
    flags=re.IGNORECASE,
)
_KINSHIP = re.compile(
    r"\b(?:vợ|chồng|con|cha|mẹ|bố|anh|chị|em|ông|bà|cháu|cụ)\b",
    flags=re.IGNORECASE,
)


def _law_query(case_query: str) -> str:
    text = _HONORIFIC_NAME.sub(" ", case_query)
    text = _KINSHIP.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip() or case_query


@dataclass(slots=True)
class PreparedCase:
    case: InferenceCase
    case_evidence: list[CaseEvidence]
    law_evidence: list[LawEvidence]
    api_calls: int


class CheckpointStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: dict[str, dict] = {}
        if self.path.exists():
            import json

            self.records = json.loads(self.path.read_text(encoding="utf-8"))

    def contains(self, case_id: str) -> bool:
        return case_id in self.records

    def put(self, result: PredictionResult) -> None:
        record = asdict(result)
        record["prediction"] = result.prediction.value if result.prediction else None
        self.records[result.case_id] = record
        write_json(self.path, self.records)

    def get(self, case_id: str) -> PredictionResult | None:
        record = self.records.get(case_id)
        if record is None:
            return None
        return PredictionResult(
            case_id=record["case_id"],
            prediction=(
                OutcomeLabel(record["prediction"]) if record["prediction"] else None
            ),
            case_evidence=[CaseEvidence(**item) for item in record["case_evidence"]],
            law_evidence=[LawEvidence(**item) for item in record["law_evidence"]],
            reasoning=record.get("reasoning"),
            raw_output=record.get("raw_output"),
            api_calls=int(record.get("api_calls", 0)),
            latency_seconds=float(record.get("latency_seconds", 0.0)),
            status=record.get("status", "completed"),
            error=record.get("error"),
        )


class PreparedCaseStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: dict[str, dict] = {}
        if self.path.exists():
            import json

            self.records = json.loads(self.path.read_text(encoding="utf-8"))

    def put(self, prepared: PreparedCase) -> None:
        self.records[prepared.case.case_id] = asdict(prepared)
        write_json(self.path, self.records)

    def get(self, case: InferenceCase) -> PreparedCase | None:
        record = self.records.get(case.case_id)
        if record is None:
            return None
        stored_case = InferenceCase(**record["case"])
        if stored_case != case:
            raise ValueError(f"Prepared checkpoint input changed: {case.case_id}")
        return PreparedCase(
            case=stored_case,
            case_evidence=[CaseEvidence(**item) for item in record["case_evidence"]],
            law_evidence=[LawEvidence(**item) for item in record["law_evidence"]],
            api_calls=int(record.get("api_calls", 0)),
        )


class ALQACPipeline:
    def __init__(self, case_retriever, law_retriever, predictor):
        self.case_retriever = case_retriever
        self.law_retriever = law_retriever
        self.predictor = predictor

    def prepare_case(self, case: InferenceCase, law_top_k: int = 5) -> PreparedCase:
        case_evidence, api_calls = self.case_retriever.retrieve(case)
        law_evidence = self._retrieve_law(case, case_evidence, law_top_k)
        return PreparedCase(case, case_evidence, law_evidence, api_calls)

    def _retrieve_law(
        self,
        case: InferenceCase,
        case_evidence: list[CaseEvidence],
        law_top_k: int,
    ) -> list[LawEvidence]:
        articles = getattr(self.law_retriever, "articles", [])
        # Always start from the ubiquitous civil-procedure articles, then add the
        # provisions the court literally cited in the retrieved evidence.
        priors = procedural_prior_evidence(articles) if articles else []
        citations = (
            extract_law_citations((item.text for item in case_evidence), articles)
            if case_evidence
            else []
        )
        if priors or citations:
            merged: list[LawEvidence] = []
            seen: set[tuple[str, int]] = set()
            for item in citations + priors:
                key = (item.law_id, item.aid)
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
            return merged
        # No corpus/evidence context (e.g. mock run): fall back to BM25 over the query.
        enriched_query = _law_query(case.case_query)
        if case_evidence:
            enriched_query += "\n" + "\n".join(
                item.text[:500] for item in case_evidence[:8]
            )
        return self.law_retriever.search(enriched_query, top_k=law_top_k)

    def predict_prepared(self, prepared: PreparedCase) -> PredictionResult:
        started = time.perf_counter()
        try:
            label, reasoning, raw = self.predictor.predict(
                prepared.case, prepared.case_evidence, prepared.law_evidence
            )
            return PredictionResult(
                case_id=prepared.case.case_id,
                prediction=label,
                case_evidence=prepared.case_evidence,
                law_evidence=prepared.law_evidence,
                reasoning=reasoning,
                raw_output=raw,
                api_calls=prepared.api_calls,
                latency_seconds=time.perf_counter() - started,
            )
        except Exception as error:  # preserve progress and make failures explicit
            return PredictionResult(
                case_id=prepared.case.case_id,
                prediction=None,
                case_evidence=prepared.case_evidence,
                law_evidence=prepared.law_evidence,
                api_calls=prepared.api_calls,
                latency_seconds=time.perf_counter() - started,
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )

    def predict_case(self, case: InferenceCase, law_top_k: int = 5) -> PredictionResult:
        return self.predict_prepared(self.prepare_case(case, law_top_k=law_top_k))
