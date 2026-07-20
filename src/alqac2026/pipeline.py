from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import write_json
from .schemas import (
    CaseEvidence,
    InferenceCase,
    LawEvidence,
    OutcomeLabel,
    PredictionResult,
)


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
        enriched_query = build_law_query(case, case_evidence)
        law_evidence = self.law_retriever.search(enriched_query, top_k=law_top_k)
        return PreparedCase(case, case_evidence, law_evidence, api_calls)

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


def build_law_query(
    case: InferenceCase,
    case_evidence: list[CaseEvidence],
    *,
    max_evidence_sentences: int = 6,
) -> str:
    """Build a compact legal-retrieval query without arbitrary prefix slicing."""
    sections = [re.sub(r"\s+", " ", case.case_query).strip()]
    dispute = re.search(
        r"tranh chấp[^.,;?\n]*", case.case_query, flags=re.IGNORECASE
    )
    if dispute:
        sections.append(dispute.group(0).strip())

    keywords = (
        "điều ",
        "bộ luật",
        "luật ",
        "nghị định",
        "nghị quyết",
        "yêu cầu",
        "chấp nhận",
        "không chấp nhận",
        "tuyên xử",
        "nghĩa vụ",
    )
    selected = []
    ordered = sorted(
        case_evidence,
        key=lambda item: (
            0 if item.query_type == "court_decision" else 1,
            -item.score,
        ),
    )
    for evidence in ordered:
        sentences = re.split(r"(?<=[.!?;])\s+|\n+", evidence.text)
        for sentence in sentences:
            normalized = re.sub(r"\s+", " ", sentence).strip()
            lowered = normalized.lower()
            if normalized and any(keyword in lowered for keyword in keywords):
                selected.append(normalized[:800])
                if len(selected) >= max_evidence_sentences:
                    break
        if len(selected) >= max_evidence_sentences:
            break
    sections.extend(selected)
    return "\n".join(dict.fromkeys(section for section in sections if section))
