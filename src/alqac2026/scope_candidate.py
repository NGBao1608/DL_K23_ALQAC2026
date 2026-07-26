from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import runner
from .config import load_config, sha256_file, write_json
from .prediction import PredictionDiagnostics, TransformersQwenBackend
from .query_planning import (
    COURT_ROLE_MARKERS,
    NEGATIVE_ROLE_MARKERS,
    POSITIVE_OPERATIVE_MARKERS,
    PROCEDURAL_ONLY_MARKERS,
    SCOPE_MARKERS,
    normalized_lookup,
)
from .rescore import rescore_prepared_cases
from .schemas import CaseEvidence, InferenceCase, LawEvidence, OutcomeLabel


CANDIDATE_NAME = "qwen3_cached_rescore_scope_v1"
SCHEMA_VERSION = "scope-enum-v1"
MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

SOURCE_ROLES = {
    "COURT_DECISION",
    "COURT_REASONING",
    "PARTY_STATEMENT",
    "PROSECUTOR_PROPOSAL",
    "UNKNOWN",
}
SCOPES = {
    "ALL",
    "MORE_THAN_HALF",
    "AT_MOST_HALF",
    "NONE",
    "UNCLEAR",
}
NEGATIVE_SOURCE_ROLES = {
    "PARTY_STATEMENT",
    "PROSECUTOR_PROPOSAL",
    "UNKNOWN",
}

SCOPE_SYSTEM_PROMPT = """Bạn là hệ thống phân loại kết quả vụ án dân sự Việt Nam.
Chỉ sử dụng case_query và evidence được cung cấp. Không bổ sung tình tiết bên ngoài.

Ưu tiên phần Tuyên xử, Quyết định hoặc kết luận của Hội đồng xét xử. Không coi lời
trình bày của nguyên đơn, bị đơn hay đề nghị của Viện kiểm sát là phán quyết.
Chỉ đánh giá yêu cầu chính của nguyên đơn; loại án phí, thủ tục và yêu cầu độc lập
của người khác.

Trả về ngay đúng một JSON object trên một dòng, không markdown và không văn bản
ngoài JSON:
{"source_role":"COURT_DECISION","claim":"tối đa 20 từ",
"accepted":"tối đa 20 từ","scope":"AT_MOST_HALF","ratio":0.4}

source_role chỉ được là COURT_DECISION, COURT_REASONING, PARTY_STATEMENT,
PROSECUTOR_PROPOSAL hoặc UNKNOWN.
scope chỉ được là ALL, MORE_THAN_HALF, AT_MOST_HALF, NONE hoặc UNCLEAR.
ratio nằm trong [0,1], hoặc null khi không thể định lượng.
ALL tương ứng toàn bộ; MORE_THAN_HALF là một phần lớn hơn 50%; AT_MOST_HALF là
một phần không quá 50%; NONE là bác toàn bộ; UNCLEAR chỉ dùng khi evidence không
đủ để xác định.
"""

QUERY_TYPE_BONUS = {
    "operative_verdict": 3,
    "adaptive_missing_scope": 2,
    "remedy_scope": 1,
}

FULL_ACCEPT_MARKERS = (
    "chấp nhận toàn bộ yêu cầu khởi kiện",
    "chấp nhận toàn bộ yêu cầu của nguyên đơn",
)
FULL_REJECT_MARKERS = (
    "không chấp nhận toàn bộ",
    "bác toàn bộ",
    "không chấp nhận yêu cầu khởi kiện",
    "bác yêu cầu khởi kiện",
)
PARTIAL_MARKERS = (
    "chấp nhận một phần",
    "không chấp nhận một phần",
)

LEXICAL_STOPWORDS = {
    "của",
    "cho",
    "các",
    "với",
    "trong",
    "được",
    "không",
    "theo",
    "một",
    "những",
    "nguyên",
    "đơn",
    "yêu",
    "cầu",
    "tòa",
    "án",
}


@dataclass(frozen=True, slots=True)
class ScopePrediction:
    source_role: str
    claim: str
    accepted: str
    scope: str
    ratio: float | None
    inconsistent: bool

    @property
    def label(self) -> OutcomeLabel | None:
        return {
            "ALL": OutcomeLabel.A_WIN,
            "MORE_THAN_HALF": OutcomeLabel.PARTIAL_A_WIN,
            "AT_MOST_HALF": OutcomeLabel.PARTIAL_B_WIN,
            "NONE": OutcomeLabel.B_WIN,
        }.get(self.scope)


@dataclass(frozen=True, slots=True)
class PromptMetadata:
    ordered_chunk_ids: tuple[str, ...]
    case_evidence_count: int
    law_evidence_count: int
    case_tokens: int
    law_tokens: int


class InstrumentedQwenBackend(TransformersQwenBackend):
    """Candidate-local Qwen backend with safe token and memory telemetry."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.generation_records: list[dict[str, Any]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import torch

        prompt = self._render_prompt(system_prompt, user_prompt)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
        ).to(self.model.device)
        input_tokens = int(inputs["input_ids"].shape[1])
        if input_tokens > self.max_input_tokens:
            raise ValueError(
                f"Prompt exceeds token budget: {input_tokens} > "
                f"{self.max_input_tokens}"
            )
        cuda_enabled = bool(torch.cuda.is_available())
        if cuda_enabled:
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            generation_kwargs = {
                "do_sample": False,
                "max_new_tokens": self.max_new_tokens,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if self.cache_implementation and cuda_enabled:
                generation_kwargs["cache_implementation"] = (
                    self.cache_implementation
                )
            output = self.model.generate(
                **inputs,
                **generation_kwargs,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        generated_tokens = int(generated.shape[0])
        ended_with_eos = bool(
            generated_tokens
            and self.tokenizer.eos_token_id is not None
            and int(generated[-1]) == int(self.tokenizer.eos_token_id)
        )
        record = {
            "input_tokens": input_tokens,
            "generated_tokens": generated_tokens,
            "ended_with_eos": ended_with_eos,
            "hit_max_new_tokens": generated_tokens >= self.max_new_tokens,
            "max_new_tokens": self.max_new_tokens,
            "max_input_tokens": self.max_input_tokens,
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if cuda_enabled else None
            ),
            "peak_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved()) if cuda_enabled else None
            ),
        }
        self.generation_records.append(record)
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def _json_object(text: str) -> tuple[dict[str, Any], bool]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model output does not contain a JSON object")
    fragment = text[start : end + 1]
    payload = json.loads(fragment)
    if not isinstance(payload, dict):
        raise ValueError("Model output JSON must be an object")
    return payload, text.strip() == fragment.strip()


def _ratio_inconsistent(scope: str, ratio: float | None) -> bool:
    if scope == "UNCLEAR":
        return ratio is not None
    if ratio is None:
        return False
    if scope == "ALL":
        return ratio != 1.0
    if scope == "NONE":
        return ratio != 0.0
    if scope == "MORE_THAN_HALF":
        return not 0.5 < ratio < 1.0
    if scope == "AT_MOST_HALF":
        return not 0.0 < ratio <= 0.5
    return True


def parse_scope_prediction(text: str) -> tuple[ScopePrediction, bool]:
    payload, strict_json = _json_object(text)
    required = {"source_role", "claim", "accepted", "scope", "ratio"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Model output is missing fields: {sorted(missing)}")
    extra = set(payload) - required
    if extra:
        raise ValueError(f"Model output has extra fields: {sorted(extra)}")
    source_role = str(payload["source_role"]).strip().upper()
    scope = str(payload["scope"]).strip().upper()
    if source_role not in SOURCE_ROLES:
        raise ValueError(f"Invalid source_role: {source_role}")
    if scope not in SCOPES:
        raise ValueError(f"Invalid scope: {scope}")
    claim = str(payload["claim"]).strip()
    accepted = str(payload["accepted"]).strip()
    if not claim:
        raise ValueError("claim must be non-empty")
    if scope not in {"NONE", "UNCLEAR"} and not accepted:
        raise ValueError("accepted must be non-empty for a non-empty scope")
    ratio_value = payload["ratio"]
    if ratio_value is None:
        ratio = None
    else:
        if isinstance(ratio_value, bool) or not isinstance(
            ratio_value, (int, float)
        ):
            raise ValueError("ratio must be a number or null")
        ratio = float(ratio_value)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio must be within [0, 1]")
    return (
        ScopePrediction(
            source_role=source_role,
            claim=claim,
            accepted=accepted,
            scope=scope,
            ratio=ratio,
            inconsistent=_ratio_inconsistent(scope, ratio),
        ),
        strict_json,
    )


def _lexical_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", normalized_lookup(text))
        if len(token) >= 3 and token not in LEXICAL_STOPWORDS
    }


def _evidence_rank(
    case_query: str,
    evidence: CaseEvidence,
) -> tuple[int, float, str]:
    text = normalized_lookup(evidence.text)
    score = QUERY_TYPE_BONUS.get(evidence.query_type, 0)
    score += 6 * any(marker in text for marker in POSITIVE_OPERATIVE_MARKERS)
    score += 4 * any(marker in text for marker in COURT_ROLE_MARKERS)
    score += 3 * any(marker in text for marker in SCOPE_MARKERS)
    score -= 6 * any(marker in text for marker in NEGATIVE_ROLE_MARKERS)
    if any(marker in text for marker in PROCEDURAL_ONLY_MARKERS) and not any(
        marker in text for marker in SCOPE_MARKERS
    ):
        score -= 3
    query_terms = _lexical_terms(case_query)
    evidence_terms = _lexical_terms(evidence.text)
    if query_terms and query_terms & evidence_terms:
        score += 2
    query_numbers = set(re.findall(r"\d+(?:[.,]\d+)*", case_query))
    evidence_numbers = set(re.findall(r"\d+(?:[.,]\d+)*", evidence.text))
    if query_numbers & evidence_numbers:
        score += 2
    return score, float(evidence.score), evidence.chunk_id


def order_case_evidence(
    case: InferenceCase,
    evidence: list[CaseEvidence],
) -> list[CaseEvidence]:
    unique: dict[str, CaseEvidence] = {}
    for item in evidence:
        existing = unique.get(item.chunk_id)
        if existing is None or float(item.score) > float(existing.score):
            unique[item.chunk_id] = item
    return sorted(
        unique.values(),
        key=lambda item: _evidence_rank(case.case_query, item),
        reverse=True,
    )


def _token_codec(tokenizer):
    if tokenizer is None:
        return (
            lambda text: text.split(),
            lambda tokens: " ".join(str(token) for token in tokens),
        )

    def encode(text: str):
        return tokenizer.encode(text, add_special_tokens=False)

    def decode(tokens) -> str:
        return tokenizer.decode(tokens, skip_special_tokens=True)

    return encode, decode


def _fit_blocks(
    blocks: list[tuple[str, str]],
    budget: int,
    *,
    encode,
    decode,
) -> tuple[str, int]:
    if budget <= 0:
        return "", 0
    rendered: list[str] = []
    used = 0
    for header, body in blocks:
        header_tokens = encode(header)
        separator_tokens = encode(" ")
        minimum = len(header_tokens) + len(separator_tokens)
        if used + minimum >= budget:
            break
        body_tokens = encode(body)
        available = budget - used - minimum
        kept = body_tokens[:available]
        if not kept:
            break
        rendered.append(f"{header} {decode(kept).strip()}")
        used += minimum + len(kept)
    return "\n\n".join(rendered), used


def _trustworthy_scope_hints(evidence: list[CaseEvidence]) -> set[str]:
    hints: set[str] = set()
    for item in evidence:
        text = normalized_lookup(item.text)
        operative = any(marker in text for marker in POSITIVE_OPERATIVE_MARKERS)
        court_role = any(marker in text for marker in COURT_ROLE_MARKERS)
        negative_role = any(marker in text for marker in NEGATIVE_ROLE_MARKERS)
        procedural_only = (
            any(marker in text for marker in PROCEDURAL_ONLY_MARKERS)
            and not any(marker in text for marker in SCOPE_MARKERS)
        )
        if not operative or not court_role or negative_role or procedural_only:
            continue
        if any(marker in text for marker in FULL_ACCEPT_MARKERS):
            hints.add("ALL")
        if any(marker in text for marker in FULL_REJECT_MARKERS):
            hints.add("NONE")
        if any(marker in text for marker in PARTIAL_MARKERS):
            hints.add("PARTIAL")
    return hints


class ScopeCandidatePredictor:
    """Isolated outcome predictor for qwen3_cached_rescore_scope_v1."""

    def __init__(
        self,
        backend,
        *,
        max_input_tokens: int,
        context_law_top_k: int,
        oom_max_input_tokens: int | None,
        oom_context_law_top_k: int | None,
        verification_reserve_tokens: int,
        case_token_share: float,
        law_token_cap: int,
        case_evidence_chars: int,
        law_evidence_chars: int,
        boundary_low: float,
        boundary_high: float,
    ):
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        if context_law_top_k < 0:
            raise ValueError("context_law_top_k cannot be negative")
        if oom_context_law_top_k is not None and not (
            0 <= oom_context_law_top_k <= context_law_top_k
        ):
            raise ValueError(
                "oom_context_law_top_k must be between zero and context_law_top_k"
            )
        if not 0.5 < case_token_share < 1.0:
            raise ValueError("case_token_share must be between 0.5 and 1.0")
        if not 0.0 <= boundary_low <= 0.5 <= boundary_high <= 1.0:
            raise ValueError("Invalid verification boundary interval")
        self.backend = backend
        self.system_prompt = SCOPE_SYSTEM_PROMPT
        self.max_input_tokens = max_input_tokens
        self.context_law_top_k = context_law_top_k
        self.oom_max_input_tokens = oom_max_input_tokens
        self.oom_context_law_top_k = oom_context_law_top_k
        self.verification_reserve_tokens = verification_reserve_tokens
        self.case_token_share = case_token_share
        self.law_token_cap = law_token_cap
        self.case_evidence_chars = case_evidence_chars
        self.law_evidence_chars = law_evidence_chars
        self.boundary_low = boundary_low
        self.boundary_high = boundary_high
        self.last_diagnostics = PredictionDiagnostics()
        self.case_diagnostics: list[dict[str, Any]] = []

    def activate_oom_recovery(self) -> bool:
        changed = False
        if (
            self.oom_max_input_tokens is not None
            and self.max_input_tokens != self.oom_max_input_tokens
        ):
            self.max_input_tokens = self.oom_max_input_tokens
            changed = True
        if (
            self.oom_context_law_top_k is not None
            and self.context_law_top_k != self.oom_context_law_top_k
        ):
            self.context_law_top_k = self.oom_context_law_top_k
            changed = True
        activate_backend = getattr(self.backend, "activate_oom_recovery", None)
        if callable(activate_backend):
            changed = bool(activate_backend()) or changed
        return changed

    def _build_prompt(
        self,
        case: InferenceCase,
        case_evidence: list[CaseEvidence],
        law_evidence: list[LawEvidence],
    ) -> tuple[str, PromptMetadata]:
        ordered = order_case_evidence(case, case_evidence)
        selected_laws = law_evidence[: self.context_law_top_k]
        tokenizer = getattr(self.backend, "tokenizer", None)
        encode, decode = _token_codec(tokenizer)
        protected = f"""## Vụ án
case_query: {case.case_query}

## Bằng chứng vụ án
Không có.

## Điều luật hỗ trợ
Không sử dụng.

Hãy xác định nguồn, yêu cầu chính và phạm vi được Tòa chấp nhận."""
        count_tokens = getattr(self.backend, "count_input_tokens", None)
        if callable(count_tokens):
            protected_tokens = int(count_tokens(self.system_prompt, protected))
        else:
            protected_tokens = len(encode(self.system_prompt + protected))
        evidence_budget = (
            self.max_input_tokens
            - protected_tokens
            - self.verification_reserve_tokens
        )
        if evidence_budget < 0:
            raise ValueError(
                "System prompt and case query exceed the protected token budget"
            )
        case_budget = int(evidence_budget * self.case_token_share)
        law_budget = min(
            evidence_budget - case_budget,
            self.law_token_cap,
        )
        case_blocks = [
            (
                f"[chunk_id={item.chunk_id} | query_type={item.query_type} | "
                f"rank={_evidence_rank(case.case_query, item)[0]}]",
                item.text[: self.case_evidence_chars],
            )
            for item in ordered
        ]
        law_blocks = [
            (
                f"[law_id={item.law_id} | aid={item.aid} | "
                f"article_number={item.article_number!r}]",
                item.text[: self.law_evidence_chars],
            )
            for item in selected_laws
        ]
        cases_text, case_used = _fit_blocks(
            case_blocks,
            case_budget,
            encode=encode,
            decode=decode,
        )
        laws_text, law_used = _fit_blocks(
            law_blocks,
            law_budget,
            encode=encode,
            decode=decode,
        )
        if law_used < law_budget and case_blocks:
            cases_text, case_used = _fit_blocks(
                case_blocks,
                case_budget + law_budget - law_used,
                encode=encode,
                decode=decode,
            )
        cases_text = cases_text or "Không truy hồi được bằng chứng vụ án."
        laws_text = laws_text or "Không sử dụng law evidence trong outcome prompt."
        prompt = f"""## Vụ án
case_query: {case.case_query}

## Bằng chứng vụ án, đã xếp theo độ tin cậy
{cases_text}

## Điều luật hỗ trợ, ưu tiên thấp hơn phán quyết
{laws_text}

Xác định source_role, yêu cầu chính và phạm vi thực tế được Tòa chấp nhận.
Chỉ trả về JSON theo schema đã yêu cầu."""
        return (
            prompt,
            PromptMetadata(
                ordered_chunk_ids=tuple(item.chunk_id for item in ordered),
                case_evidence_count=len(ordered),
                law_evidence_count=len(selected_laws),
                case_tokens=case_used,
                law_tokens=law_used,
            ),
        )

    def _parse_or_repair(
        self,
        raw: str,
        original_prompt: str,
    ) -> tuple[ScopePrediction, str, bool, bool]:
        try:
            parsed, strict_json = parse_scope_prediction(raw)
            return parsed, raw, False, strict_json
        except (KeyError, ValueError, json.JSONDecodeError) as first_error:
            repair = f"""{original_prompt}

## Sửa định dạng
Đầu ra trước không hợp lệ. Đọc lại evidence ở trên và trả về đúng một JSON object
có source_role, claim, accepted, scope và ratio. Không lặp lại đầu ra lỗi."""
            repaired = self.backend.generate(self.system_prompt, repair)
            try:
                parsed, strict_json = parse_scope_prediction(repaired)
                return (
                    parsed,
                    f"{raw}\n---SCOPE_REPAIR---\n{repaired}",
                    True,
                    strict_json,
                )
            except (KeyError, ValueError, json.JSONDecodeError) as second_error:
                raise ValueError(
                    "Scope candidate output failed validation twice: "
                    f"{first_error}; {second_error}"
                ) from second_error

    def _verification_triggers(
        self,
        parsed: ScopePrediction,
        evidence: list[CaseEvidence],
    ) -> list[str]:
        triggers: list[str] = []
        if parsed.scope == "UNCLEAR":
            triggers.append("unclear_scope")
        if parsed.source_role in NEGATIVE_SOURCE_ROLES:
            triggers.append("untrusted_source_role")
        if parsed.inconsistent:
            triggers.append("ratio_scope_inconsistent")
        if (
            parsed.ratio is not None
            and parsed.scope in {"MORE_THAN_HALF", "AT_MOST_HALF"}
            and self.boundary_low <= parsed.ratio <= self.boundary_high
        ):
            triggers.append("near_half_boundary")
        hints = _trustworthy_scope_hints(evidence)
        if len(hints) > 1:
            triggers.append("conflicting_operative_markers")
        if "ALL" in hints and parsed.scope != "ALL":
            triggers.append("full_acceptance_conflict")
        if "NONE" in hints and parsed.scope != "NONE":
            triggers.append("full_rejection_conflict")
        if (
            "PARTIAL" in hints
            and parsed.scope in {"ALL", "NONE"}
            and len(hints) == 1
        ):
            triggers.append("partial_scope_conflict")
        return list(dict.fromkeys(triggers))

    def _verify(
        self,
        prompt: str,
        parsed: ScopePrediction,
        triggers: list[str],
    ) -> tuple[ScopePrediction, str, bool]:
        initial = {
            "source_role": parsed.source_role,
            "claim": parsed.claim,
            "accepted": parsed.accepted,
            "scope": parsed.scope,
            "ratio": parsed.ratio,
        }
        verification = f"""{prompt}

## Kết quả bước đầu
{json.dumps(initial, ensure_ascii=False, sort_keys=True)}

## Kiểm tra có mục tiêu
Các cờ cần kiểm tra: {", ".join(triggers)}.
Chỉ kiểm tra nguồn phán quyết và ranh giới phạm vi chấp nhận. Trả về lại đúng
scope JSON schema; không giải thích ngoài JSON."""
        raw = self.backend.generate(self.system_prompt, verification)
        verified, strict_json = parse_scope_prediction(raw)
        if (
            verified.scope == "UNCLEAR"
            or verified.source_role in NEGATIVE_SOURCE_ROLES
            or verified.inconsistent
        ):
            raise ValueError("Targeted verifier did not resolve the ambiguity")
        return verified, raw, strict_json

    def predict(
        self,
        case: InferenceCase,
        case_evidence: list[CaseEvidence],
        law_evidence: list[LawEvidence],
    ) -> tuple[OutcomeLabel, str, str]:
        self.last_diagnostics = PredictionDiagnostics()
        generation_start = len(
            getattr(self.backend, "generation_records", [])
        )
        prompt, prompt_metadata = self._build_prompt(
            case,
            case_evidence,
            law_evidence,
        )
        record: dict[str, Any] = {
            "case_id": case.case_id,
            **asdict(prompt_metadata),
        }
        try:
            raw = self.backend.generate(self.system_prompt, prompt)
            parsed, combined_raw, repair_used, strict_json = self._parse_or_repair(
                raw,
                prompt,
            )
            triggers = self._verification_triggers(parsed, case_evidence)
            verification_status = "not_required"
            verifier_strict_json: bool | None = None
            if triggers:
                try:
                    parsed, verified_raw, verifier_strict_json = self._verify(
                        prompt,
                        parsed,
                        triggers,
                    )
                    combined_raw = (
                        f"{combined_raw}\n---SCOPE_VERIFICATION---\n{verified_raw}"
                    )
                    verification_status = "passed"
                except (KeyError, ValueError, json.JSONDecodeError):
                    verification_status = "failed"
                    combined_raw = (
                        f"{combined_raw}\n---SCOPE_VERIFICATION_FAILED---"
                    )
                    if parsed.label is None:
                        raise
            if parsed.label is None:
                raise ValueError("Scope candidate did not resolve an outcome label")
            self.last_diagnostics = PredictionDiagnostics(
                output_repair_used=repair_used,
                output_verification=verification_status,
            )
            record.update(
                {
                    "source_role": parsed.source_role,
                    "scope": parsed.scope,
                    "ratio": parsed.ratio,
                    "strict_initial_or_repair_json": strict_json,
                    "repair_used": repair_used,
                    "verification": verification_status,
                    "verification_triggers": triggers,
                    "verifier_strict_json": verifier_strict_json,
                    "status": "completed",
                }
            )
            reasoning = (
                f"scope={parsed.scope}; source_role={parsed.source_role}; "
                f"ratio={parsed.ratio!r}"
            )
            return parsed.label, reasoning, combined_raw
        except Exception as error:
            record.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                }
            )
            raise
        finally:
            generation_records = getattr(
                self.backend,
                "generation_records",
                [],
            )
            record["generations"] = generation_records[generation_start:]
            self.case_diagnostics.append(record)


def create_scope_candidate_predictor(config: dict) -> ScopeCandidatePredictor:
    if config.get("model_name") != MODEL_NAME:
        raise ValueError(f"{CANDIDATE_NAME} requires {MODEL_NAME}")
    if config.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"{CANDIDATE_NAME} requires revision={MODEL_REVISION}"
        )
    if config.get("prompt_variant") != SCHEMA_VERSION:
        raise ValueError(
            f"{CANDIDATE_NAME} requires prompt_variant={SCHEMA_VERSION}"
        )
    if not bool(config.get("load_in_4bit", True)):
        raise ValueError(f"{CANDIDATE_NAME} requires NF4 4-bit loading")
    if bool(config.get("thinking", False)):
        raise ValueError(f"{CANDIDATE_NAME} requires thinking=false")
    if bool(config.get("do_sample", False)):
        raise ValueError(f"{CANDIDATE_NAME} requires do_sample=false")
    if config.get("adapter_path") is not None:
        raise ValueError(f"{CANDIDATE_NAME} does not permit an adapter")
    backend = InstrumentedQwenBackend(
        model_name=config["model_name"],
        revision=config.get("revision"),
        load_in_4bit=bool(config.get("load_in_4bit", True)),
        max_input_tokens=int(config.get("max_input_tokens", 4096)),
        max_new_tokens=int(config.get("max_new_tokens", 320)),
        oom_max_input_tokens=(
            int(config["oom_max_input_tokens"])
            if config.get("oom_max_input_tokens") is not None
            else None
        ),
        oom_max_new_tokens=(
            int(config["oom_max_new_tokens"])
            if config.get("oom_max_new_tokens") is not None
            else None
        ),
        attn_implementation=config.get("attn_implementation"),
        cache_implementation=config.get("cache_implementation"),
        thinking=bool(config.get("thinking", False)),
        adapter_path=config.get("adapter_path"),
    )
    return ScopeCandidatePredictor(
        backend,
        max_input_tokens=int(config.get("max_input_tokens", 4096)),
        context_law_top_k=int(config.get("context_law_top_k", 1)),
        oom_max_input_tokens=(
            int(config["oom_max_input_tokens"])
            if config.get("oom_max_input_tokens") is not None
            else None
        ),
        oom_context_law_top_k=(
            int(config["oom_context_law_top_k"])
            if config.get("oom_context_law_top_k") is not None
            else None
        ),
        verification_reserve_tokens=int(
            config.get("verification_reserve_tokens", 512)
        ),
        case_token_share=float(config.get("case_token_share", 0.85)),
        law_token_cap=int(config.get("law_token_cap", 512)),
        case_evidence_chars=int(config.get("case_evidence_chars", 2600)),
        law_evidence_chars=int(config.get("law_evidence_chars", 1200)),
        boundary_low=float(config.get("verification_boundary_low", 0.45)),
        boundary_high=float(config.get("verification_boundary_high", 0.55)),
    )


def _diagnostic_summary(
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    completed = [
        item for item in diagnostics if item.get("status") == "completed"
    ]
    generations = [
        generation
        for item in diagnostics
        for generation in item.get("generations", [])
    ]
    return {
        "schema_version": "scope-candidate-diagnostics-v1",
        "candidate": CANDIDATE_NAME,
        "case_attempts": len(diagnostics),
        "completed_case_attempts": len(completed),
        "scope_distribution": dict(
            sorted(Counter(item.get("scope") for item in completed).items())
        ),
        "source_role_distribution": dict(
            sorted(
                Counter(item.get("source_role") for item in completed).items()
            )
        ),
        "repair_attempts": sum(
            bool(item.get("repair_used")) for item in completed
        ),
        "strict_json_case_attempts": sum(
            bool(item.get("strict_initial_or_repair_json"))
            for item in completed
        ),
        "verification_passed": sum(
            item.get("verification") == "passed" for item in completed
        ),
        "verification_failed": sum(
            item.get("verification") == "failed" for item in completed
        ),
        "generation_calls": len(generations),
        "generation_hit_max_new_tokens": sum(
            bool(item.get("hit_max_new_tokens")) for item in generations
        ),
        "generation_ended_with_eos": sum(
            bool(item.get("ended_with_eos")) for item in generations
        ),
        "max_input_tokens_observed": max(
            (int(item.get("input_tokens", 0)) for item in generations),
            default=0,
        ),
        "max_generated_tokens_observed": max(
            (int(item.get("generated_tokens", 0)) for item in generations),
            default=0,
        ),
        "peak_cuda_allocated_bytes": max(
            (
                int(item["peak_cuda_allocated_bytes"])
                for item in generations
                if item.get("peak_cuda_allocated_bytes") is not None
            ),
            default=None,
        ),
        "peak_cuda_reserved_bytes": max(
            (
                int(item["peak_cuda_reserved_bytes"])
                for item in generations
                if item.get("peak_cuda_reserved_bytes") is not None
            ),
            default=None,
        ),
        "cases": diagnostics,
    }


def _write_candidate_contract(
    run_dir: Path,
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    diagnostic_artifact = _diagnostic_summary(diagnostics)
    write_json(run_dir / "scope_diagnostics.json", diagnostic_artifact)
    contract = {
        "name": CANDIDATE_NAME,
        "schema_version": SCHEMA_VERSION,
        "predictor_factory": "candidate_local_temporary_override",
        "case_api_network_calls": 0,
        "planner_executed": False,
        "source_artifacts_immutable": True,
        "diagnostics": "scope_diagnostics.json",
    }
    for filename in ("manifest.json", "validation.json"):
        path = run_dir / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["scope_candidate"] = contract
        write_json(path, payload)
    return {
        "contract": contract,
        "diagnostics": diagnostic_artifact,
    }


def run_scope_candidate(
    *,
    config_path: str | Path,
    input_path: str | Path,
    prepared_contexts_path: str | Path,
    run_dir: str | Path,
    public_gold_path: str | Path | None = None,
    selection_profile: str | Path | None = None,
    corpus_path: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the isolated scope candidate over immutable prepared contexts."""
    config = load_config(config_path)
    if config["run"].get("name") != CANDIDATE_NAME:
        raise ValueError(
            f"Scope candidate config must use run.name={CANDIDATE_NAME}"
        )
    prediction_config = config["prediction"]
    if prediction_config.get("model_name") != MODEL_NAME:
        raise ValueError(f"{CANDIDATE_NAME} must not change the outcome model")
    if prediction_config.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"{CANDIDATE_NAME} must keep revision={MODEL_REVISION}"
        )
    if prediction_config.get("adapter_path") is not None:
        raise ValueError(f"{CANDIDATE_NAME} must not use an adapter")
    source_contexts = Path(prepared_contexts_path).resolve()
    target_run = Path(run_dir).resolve()
    if not source_contexts.is_file():
        raise FileNotFoundError(
            f"Prepared contexts do not exist: {source_contexts}"
        )
    if (
        target_run == source_contexts.parent
        or target_run in source_contexts.parents
        or source_contexts.parent in target_run.parents
    ):
        raise ValueError(
            "Scope candidate run directory must be outside the source "
            "PreparedCase artifact tree"
        )
    source_sha256_before = sha256_file(source_contexts)
    source_bytes_before = source_contexts.stat().st_size

    created: list[ScopeCandidatePredictor] = []

    def candidate_factory(prediction_config: dict):
        predictor = create_scope_candidate_predictor(prediction_config)
        created.append(predictor)
        return predictor

    original_factory = runner.create_predictor
    runner.create_predictor = candidate_factory
    try:
        result = rescore_prepared_cases(
            config_path=config_path,
            input_path=input_path,
            prepared_contexts_path=prepared_contexts_path,
            run_dir=run_dir,
            public_gold_path=public_gold_path,
            selection_profile=selection_profile,
            adapter_path=None,
            corpus_path=corpus_path,
            limit=limit,
        )
    finally:
        runner.create_predictor = original_factory

    if (
        sha256_file(source_contexts) != source_sha256_before
        or source_contexts.stat().st_size != source_bytes_before
    ):
        raise RuntimeError(
            "Source PreparedCase artifact changed during candidate execution"
        )
    if len(created) != 1:
        raise RuntimeError(
            "Scope candidate expected exactly one predictor instance"
        )
    candidate_artifacts = _write_candidate_contract(
        Path(run_dir),
        created[0].case_diagnostics,
    )
    result["scope_candidate"] = candidate_artifacts["contract"]
    return result
