from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

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
            prediction_attempts=max(1, int(record.get("prediction_attempts", 1))),
            prediction_failure_types=[
                str(value)
                for value in record.get("prediction_failure_types", [])
            ],
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
    def __init__(
        self,
        case_retriever,
        law_retriever,
        predictor,
        *,
        allow_prediction_fallback: bool = False,
        prediction_fallback_label: OutcomeLabel = OutcomeLabel.B_WIN,
        max_prediction_retries: int = 0,
        prediction_retry_callback: (
            Callable[[dict[str, object]], None] | None
        ) = None,
    ):
        if (
            isinstance(max_prediction_retries, bool)
            or not isinstance(max_prediction_retries, int)
            or not 0 <= max_prediction_retries <= 3
        ):
            raise ValueError("max_prediction_retries must be an integer from 0 to 3")
        self.case_retriever = case_retriever
        self.law_retriever = law_retriever
        self.predictor = predictor
        self.allow_prediction_fallback = allow_prediction_fallback
        self.prediction_fallback_label = prediction_fallback_label
        self.max_prediction_retries = max_prediction_retries
        self.prediction_retry_callback = prediction_retry_callback

    def prepare_case(self, case: InferenceCase, law_top_k: int = 5) -> PreparedCase:
        case_evidence, api_calls = self.case_retriever.retrieve(case)
        enriched_query = build_law_query(case, case_evidence)
        law_evidence = self.law_retriever.search(enriched_query, top_k=law_top_k)
        return PreparedCase(case, case_evidence, law_evidence, api_calls)

    def predict_prepared(self, prepared: PreparedCase) -> PredictionResult:
        started = time.perf_counter()
        failure_types: list[str] = []
        max_attempts = 1 + self.max_prediction_retries
        for attempt_index in range(max_attempts):
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
                    prediction_attempts=attempt_index + 1,
                    prediction_failure_types=failure_types,
                )
            except Exception as error:
                failure_types.append(type(error).__name__)
                _clear_cuda_cache_after_case_failure()
                if attempt_index + 1 < max_attempts:
                    self._emit_prediction_retry(
                        case_id=prepared.case.case_id,
                        failed_attempt=attempt_index + 1,
                        next_attempt=attempt_index + 2,
                        max_attempts=max_attempts,
                        error_type=type(error).__name__,
                    )
                    continue
                if not self.allow_prediction_fallback:
                    return PredictionResult(
                        case_id=prepared.case.case_id,
                        prediction=None,
                        case_evidence=prepared.case_evidence,
                        law_evidence=prepared.law_evidence,
                        api_calls=prepared.api_calls,
                        latency_seconds=time.perf_counter() - started,
                        status="failed",
                        error=f"{type(error).__name__}: {error}",
                        prediction_attempts=attempt_index + 1,
                        prediction_failure_types=failure_types,
                    )
                fallback_label, fallback_reason = _fallback_outcome(
                    prepared.case_evidence,
                    default=self.prediction_fallback_label,
                )
                return PredictionResult(
                    case_id=prepared.case.case_id,
                    prediction=fallback_label,
                    case_evidence=prepared.case_evidence,
                    law_evidence=prepared.law_evidence,
                    reasoning=fallback_reason,
                    api_calls=prepared.api_calls,
                    latency_seconds=time.perf_counter() - started,
                    status="completed",
                    error=f"PredictionFallback:{type(error).__name__}",
                    prediction_attempts=attempt_index + 1,
                    prediction_failure_types=failure_types,
                )
        raise RuntimeError("Prediction retry loop terminated unexpectedly")

    def _emit_prediction_retry(self, **event: object) -> None:
        if self.prediction_retry_callback is None:
            return
        try:
            self.prediction_retry_callback(event)
        except Exception:
            # Observability must never interrupt bounded case recovery.
            return

    def predict_case(self, case: InferenceCase, law_top_k: int = 5) -> PredictionResult:
        return self.predict_prepared(self.prepare_case(case, law_top_k=law_top_k))


def _fallback_outcome(
    case_evidence: list[CaseEvidence],
    *,
    default: OutcomeLabel,
) -> tuple[OutcomeLabel, str]:
    """Choose a submission-safe label from operative evidence after case failure."""
    priority = {
        "operative_verdict": 0,
        "adaptive_missing_scope": 1,
        "remedy_scope": 2,
    }
    ordered = sorted(
        case_evidence,
        key=lambda item: (priority.get(item.query_type, 99), -item.score),
    )
    texts = [re.sub(r"\s+", " ", item.text).casefold() for item in ordered]
    trustworthy = [
        text
        for text in texts
        if any(
            marker in text
            for marker in ("hội đồng xét xử", "tuyên xử", "xử:", "quyết định")
        )
        and not any(
            marker in text
            for marker in (
                "nguyên đơn trình bày",
                "bị đơn cho rằng",
                "đại diện viện kiểm sát đề nghị",
            )
        )
    ]
    combined = "\n".join(trustworthy)
    if any(
        marker in combined
        for marker in (
            "không chấp nhận toàn bộ",
            "bác toàn bộ",
            "không chấp nhận yêu cầu khởi kiện",
            "bác yêu cầu khởi kiện",
        )
    ):
        return OutcomeLabel.B_WIN, "Deterministic fallback from rejection language."
    if "chấp nhận toàn bộ yêu cầu khởi kiện" in combined:
        return OutcomeLabel.A_WIN, "Deterministic fallback from full-acceptance language."
    if "chấp nhận một phần" in combined:
        return (
            OutcomeLabel.PARTIAL_B_WIN,
            "Deterministic fallback for unquantified partial acceptance.",
        )
    if "chấp nhận yêu cầu khởi kiện" in combined:
        return OutcomeLabel.A_WIN, "Deterministic fallback from acceptance language."
    return default, "Configured fallback because no trustworthy operative scope was available."


def _clear_cuda_cache_after_case_failure() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


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
            {
                "operative_verdict": 0,
                "adaptive_missing_scope": 1,
                "remedy_scope": 2,
            }.get(item.query_type, 3),
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
