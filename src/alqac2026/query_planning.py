from __future__ import annotations

import gc
import hashlib
import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import write_json
from .schemas import CaseEvidence, InferenceCase


PLANNER_PROMPT_VERSION = "structured-case-query-v1"
COMPOSER_VERSION = "deterministic-query-composer-v1"


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    return re.sub(r"\s+", " ", value).strip()


def normalized_lookup(value: str) -> str:
    return normalize_text(value).casefold()


@dataclass(frozen=True, slots=True)
class StructuredQueryPlan:
    case_type: str
    main_claim: str
    requested_remedies: tuple[str, ...]
    legal_objects: tuple[str, ...]
    amounts_or_areas: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for field in ("requested_remedies", "legal_objects", "amounts_or_areas"):
            value[field] = list(value[field])
        return value


@dataclass(frozen=True, slots=True)
class PlannerResult:
    plan: StructuredQueryPlan
    strategy: str
    failure_type: str | None = None


@dataclass(frozen=True, slots=True)
class PlannedQuery:
    text: str
    query_type: str


@dataclass(frozen=True, slots=True)
class EvidenceSufficiency:
    sufficient: bool
    has_operative_decision: bool
    has_court_role: bool
    mentions_claim_or_object: bool
    has_scope_resolution: bool
    has_duplicate_chunk_id: bool
    reasons: tuple[str, ...]


class PlannerBackend(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...

    def release(self) -> None: ...


class QueryPlanner(Protocol):
    def plan(self, case_query: str) -> PlannerResult: ...

    def release(self) -> None: ...


CASE_TYPE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("inheritance", ("thừa kế", "di sản")),
    ("land", ("quyền sử dụng đất", "tranh chấp đất", "thửa đất")),
    ("loan", ("hợp đồng tín dụng", "hợp đồng vay", "vay tài sản", "trả nợ")),
    ("contract", ("hợp đồng", "giao dịch dân sự")),
    ("compensation", ("bồi thường", "thiệt hại")),
    ("marriage_family", ("ly hôn", "hôn nhân", "nuôi con", "tài sản chung")),
    ("labor", ("lao động", "tiền lương", "sa thải")),
    ("administrative", ("quyết định hành chính", "hành vi hành chính")),
)

REMEDY_PATTERNS: tuple[str, ...] = (
    r"\btrả nợ\b",
    r"\bthanh toán(?: khoản)?(?: tiền)?\b",
    r"\bbồi thường(?: thiệt hại)?\b",
    r"\bchia di sản(?: thừa kế)?\b",
    r"\bchia (?:quyền sử dụng đất|tài sản chung|tài sản)\b",
    r"\btrả lại (?:đất|quyền sử dụng đất|tài sản)\b",
    r"\bhủy hợp đồng\b",
    r"\btuyên hợp đồng vô hiệu\b",
    r"\btuyên bố hợp đồng vô hiệu\b",
    r"\bcông nhận quyền sử dụng đất\b",
    r"\bcông nhận hợp đồng\b",
    r"\bbuộc [^.,;:\n]{2,80}",
)

OBJECT_PATTERNS: tuple[str, ...] = (
    r"\bthửa đất(?: số)?\s*[A-Za-zÀ-ỹ0-9./-]+",
    r"\btờ bản đồ(?: số)?\s*[A-Za-zÀ-ỹ0-9./-]+",
    r"\bhợp đồng(?: số)?\s*[A-Za-zÀ-ỹ0-9./-]+",
    r"\bquyền sử dụng đất\b",
    r"\btiền bồi thường\b",
    r"\bdi sản thừa kế\b",
    r"\btài sản chung\b",
    r"\btiền vay\b",
    r"\bkhoản nợ\b",
    r"\bnhà(?: ở)?\b",
)

AMOUNT_AREA_PATTERNS: tuple[str, ...] = (
    r"\b\d[\d.\s]*(?:,\d+)?\s*(?:đồng|VNĐ)\b",
    r"\b\d[\d.\s]*(?:,\d+)?\s*(?:m2|m²|ha)\b",
)


def _ordered_matches(patterns: tuple[str, ...], text: str) -> tuple[str, ...]:
    matches: list[tuple[int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = normalize_text(match.group(0)).strip(" ,.;:")
            if value:
                matches.append((match.start(), value))
    result: list[str] = []
    seen: set[str] = set()
    for _, value in sorted(matches, key=lambda item: (item[0], len(item[1]))):
        key = normalized_lookup(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def detect_case_type(case_query: str) -> str:
    lowered = normalized_lookup(case_query)
    for case_type, keywords in CASE_TYPE_KEYWORDS:
        if any(keyword.casefold() in lowered for keyword in keywords):
            return case_type
    return "other"


def _bounded_exact_span(value: str, max_chars: int = 180) -> str:
    value = normalize_text(value).strip(" ,.;:")
    if len(value) <= max_chars:
        return value
    bounded = value[:max_chars].rsplit(" ", 1)[0].strip(" ,.;:")
    return bounded or value[:max_chars]


class DeterministicQueryPlanner:
    def plan(self, case_query: str) -> PlannerResult:
        query = normalize_text(case_query)
        remedies = _ordered_matches(REMEDY_PATTERNS, query)
        objects = _ordered_matches(OBJECT_PATTERNS, query)
        amounts_or_areas = _ordered_matches(AMOUNT_AREA_PATTERNS, query)
        main_claim = self._extract_main_claim(query, remedies)
        plan = StructuredQueryPlan(
            case_type=detect_case_type(query),
            main_claim=main_claim,
            requested_remedies=remedies,
            legal_objects=objects,
            amounts_or_areas=amounts_or_areas,
        )
        validate_query_plan(plan, query)
        return PlannerResult(plan=plan, strategy="deterministic")

    @staticmethod
    def _extract_main_claim(query: str, remedies: tuple[str, ...]) -> str:
        patterns = (
            r"(?:yêu cầu|đề nghị)(?:\s+Tòa án)?\s+([^,.;?\n]+)",
            r"(tranh chấp\s+[^,.;?\n]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, query, flags=re.IGNORECASE)
            if match:
                return _bounded_exact_span(match.group(1))
        if remedies:
            return remedies[0]
        words = query.split()
        return _bounded_exact_span(" ".join(words[:24]))

    def release(self) -> None:
        return


def _json_object(value: str) -> dict[str, Any]:
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Planner output does not contain a JSON object")
    parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Planner output must be a JSON object")
    return parsed


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Planner field {field} must be a list of strings")
    return tuple(
        dict.fromkeys(normalize_text(item) for item in value if normalize_text(item))
    )


def parse_query_plan(value: str) -> StructuredQueryPlan:
    parsed = _json_object(value)
    required = {
        "case_type",
        "main_claim",
        "requested_remedies",
        "legal_objects",
        "amounts_or_areas",
    }
    if set(parsed) != required:
        raise ValueError("Planner JSON fields do not match the structured schema")
    if not isinstance(parsed["case_type"], str) or not isinstance(
        parsed["main_claim"], str
    ):
        raise ValueError("Planner case_type and main_claim must be strings")
    return StructuredQueryPlan(
        case_type=normalize_text(parsed["case_type"]),
        main_claim=normalize_text(parsed["main_claim"]),
        requested_remedies=_string_tuple(
            parsed["requested_remedies"], "requested_remedies"
        ),
        legal_objects=_string_tuple(parsed["legal_objects"], "legal_objects"),
        amounts_or_areas=_string_tuple(
            parsed["amounts_or_areas"], "amounts_or_areas"
        ),
    )


def validate_query_plan(
    plan: StructuredQueryPlan,
    case_query: str,
    *,
    deterministic_reference: StructuredQueryPlan | None = None,
) -> None:
    source = normalized_lookup(case_query)
    if not plan.main_claim:
        raise ValueError("Planner output is missing main_claim")
    if plan.case_type not in {item[0] for item in CASE_TYPE_KEYWORDS} | {"other"}:
        raise ValueError("Planner case_type is not an allowed canonical value")
    detected = detect_case_type(case_query)
    if detected != "other" and plan.case_type != detected:
        raise ValueError("Planner case_type is not supported by case_query")
    for value in (
        plan.main_claim,
        *plan.requested_remedies,
        *plan.legal_objects,
        *plan.amounts_or_areas,
    ):
        if normalized_lookup(value) not in source:
            raise ValueError("Planner output contains text absent from case_query")
    if deterministic_reference is not None:
        if (
            deterministic_reference.requested_remedies
            and not plan.requested_remedies
        ):
            raise ValueError("Planner output omitted required requested_remedies")
        if deterministic_reference.legal_objects and not plan.legal_objects:
            raise ValueError("Planner output omitted required legal_objects")


SYSTEM_PROMPT = """You are a Vietnamese legal query planner.
Return one JSON object with exactly: case_type, main_claim, requested_remedies,
legal_objects, amounts_or_areas. Use one canonical case_type from inheritance,
land, loan, contract, compensation, marriage_family, labor, administrative,
other. Copy exact spans or lexical terms from case_query for every other field.
Do not infer the judgment, acceptance, rejection, winner, or outcome.
Use arrays for the last three fields and no markdown."""


class LLMAssistedQueryPlanner:
    def __init__(
        self,
        backend: PlannerBackend,
        fallback: DeterministicQueryPlanner | None = None,
    ):
        self.backend = backend
        self.fallback = fallback or DeterministicQueryPlanner()

    def plan(self, case_query: str) -> PlannerResult:
        deterministic = self.fallback.plan(case_query).plan
        try:
            raw = self.backend.generate(
                SYSTEM_PROMPT,
                "case_query:\n" + normalize_text(case_query),
            )
            plan = parse_query_plan(raw)
            validate_query_plan(
                plan,
                case_query,
                deterministic_reference=deterministic,
            )
            return PlannerResult(plan=plan, strategy="llm")
        except Exception as error:
            return PlannerResult(
                plan=deterministic,
                strategy="deterministic_fallback",
                failure_type=type(error).__name__,
            )

    def release(self) -> None:
        try:
            self.backend.release()
        finally:
            self.fallback.release()


class PlannerDeadlineExceeded(TimeoutError):
    pass


class TransformersPlannerBackend:
    def __init__(
        self,
        *,
        model_name: str,
        revision: str,
        max_new_tokens: int = 256,
        timeout_seconds: float = 12.0,
    ):
        self.model_name = model_name
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds
        self.tokenizer = None
        self.model = None

    def _load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.revision,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.revision,
            torch_dtype="auto",
            device_map="auto",
        ).eval()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self._load()
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        deadline = time.monotonic() + self.timeout_seconds

        class DeadlineStoppingCriteria(StoppingCriteria):
            def __init__(self) -> None:
                self.expired = False

            def __call__(self, input_ids, scores, **kwargs) -> bool:
                self.expired = time.monotonic() >= deadline
                return self.expired

        stopper = DeadlineStoppingCriteria()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([stopper]),
            )
        if stopper.expired or time.monotonic() >= deadline:
            raise PlannerDeadlineExceeded(
                f"Planner generation exceeded {self.timeout_seconds} seconds"
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def release(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def create_query_planner(config: dict[str, Any]) -> QueryPlanner:
    strategy = str(config.get("strategy", "llm_assisted"))
    fallback = DeterministicQueryPlanner()
    if strategy == "deterministic":
        return fallback
    if strategy != "llm_assisted":
        raise ValueError(f"Unsupported query planner strategy: {strategy}")
    if config.get("thinking") is not False or config.get("do_sample") is not False:
        raise ValueError("LLM query planner requires thinking=false and do_sample=false")
    backend = TransformersPlannerBackend(
        model_name=str(config["model_name"]),
        revision=str(config["revision"]),
        max_new_tokens=int(config.get("max_new_tokens", 256)),
        timeout_seconds=float(config.get("timeout_seconds", 12.0)),
    )
    return LLMAssistedQueryPlanner(backend, fallback=fallback)


def _query_tokens(parts: list[str], *, max_tokens: int = 25) -> str:
    text = normalize_text(" ".join(part for part in parts if part))
    tokens = text.split()
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return " ".join(tokens)


class DeterministicQueryComposer:
    def compose(self, plan: StructuredQueryPlan) -> list[PlannedQuery]:
        object_terms = list(plan.legal_objects[:2])
        amount_terms = list(plan.amounts_or_areas[:1])
        remedy_terms = list(plan.requested_remedies[:2])
        candidates = (
            PlannedQuery(
                _query_tokens(
                    [
                        "tuyên xử quyết định của hội đồng xét xử về",
                        *object_terms,
                        plan.main_claim,
                    ]
                ),
                "operative_verdict",
            ),
            PlannedQuery(
                _query_tokens(
                    [
                        "hội đồng xét xử quyết định phạm vi",
                        *object_terms,
                        *amount_terms,
                        *remedy_terms,
                    ]
                ),
                "remedy_scope",
            ),
            PlannedQuery(
                _query_tokens(
                    [
                        "tòa án chấp nhận không chấp nhận buộc công nhận",
                        *object_terms,
                        plan.main_claim,
                    ]
                ),
                "adaptive_missing_scope",
            ),
        )
        result: list[PlannedQuery] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = normalized_lookup(candidate.text)
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(candidate)
        return result[:3]


POSITIVE_OPERATIVE_MARKERS = (
    "tuyên xử",
    "xử:",
    "quyết định",
    "chấp nhận",
    "không chấp nhận",
    "bác",
    "buộc",
    "đình chỉ",
    "công nhận",
    "tuyên bố",
)
SCOPE_MARKERS = (
    "chấp nhận",
    "không chấp nhận",
    "bác",
    "buộc",
    "đình chỉ",
    "công nhận",
    "tuyên bố",
)
COURT_ROLE_MARKERS = (
    "hội đồng xét xử",
    "tòa án",
    "tuyên xử",
    "xử:",
    "quyết định",
)
NEGATIVE_ROLE_MARKERS = (
    "nguyên đơn trình bày",
    "bị đơn cho rằng",
    "đề nghị tòa án",
    "đại diện viện kiểm sát đề nghị",
)
PROCEDURAL_ONLY_MARKERS = (
    "thụ lý vụ án",
    "triệu tập",
    "thành phần hội đồng xét xử",
    "người tham gia tố tụng",
)


def _plan_terms(plan: StructuredQueryPlan) -> tuple[str, ...]:
    terms = (
        plan.main_claim,
        *plan.requested_remedies,
        *plan.legal_objects,
        *plan.amounts_or_areas,
    )
    return tuple(
        dict.fromkeys(
            normalized_lookup(term)
            for term in terms
            if len(normalized_lookup(term)) >= 3
        )
    )


def evaluate_evidence_sufficiency(
    evidence: list[CaseEvidence],
    plan: StructuredQueryPlan,
) -> EvidenceSufficiency:
    chunk_ids = [item.chunk_id for item in evidence]
    duplicate = len(chunk_ids) != len(set(chunk_ids))
    terms = _plan_terms(plan)
    has_operative = False
    has_court_role = False
    mentions_claim = False
    has_scope = False
    trustworthy_scope = False
    for item in evidence:
        text = normalized_lookup(item.text)
        operative = any(marker in text for marker in POSITIVE_OPERATIVE_MARKERS)
        court_role = any(marker in text for marker in COURT_ROLE_MARKERS)
        scope = any(marker in text for marker in SCOPE_MARKERS)
        mention = any(term in text for term in terms)
        negative_role = any(marker in text for marker in NEGATIVE_ROLE_MARKERS)
        procedural_only = (
            any(marker in text for marker in PROCEDURAL_ONLY_MARKERS)
            and not scope
        )
        has_operative = has_operative or operative
        has_court_role = has_court_role or court_role
        mentions_claim = mentions_claim or mention
        has_scope = has_scope or scope
        if (
            operative
            and court_role
            and scope
            and mention
            and not negative_role
            and not procedural_only
        ):
            trustworthy_scope = True
    reasons: list[str] = []
    if not has_operative:
        reasons.append("missing_operative_decision")
    if not has_court_role:
        reasons.append("missing_court_role")
    if not mentions_claim:
        reasons.append("missing_claim_or_object")
    if not has_scope:
        reasons.append("missing_scope_resolution")
    if duplicate:
        reasons.append("duplicate_chunk_id")
    if not trustworthy_scope and not reasons:
        reasons.append("untrustworthy_source_role")
    sufficient = trustworthy_scope and not duplicate
    return EvidenceSufficiency(
        sufficient=sufficient,
        has_operative_decision=has_operative,
        has_court_role=has_court_role,
        mentions_claim_or_object=mentions_claim,
        has_scope_resolution=has_scope,
        has_duplicate_chunk_id=duplicate,
        reasons=tuple(reasons),
    )


def query_plan_fingerprint(
    case: InferenceCase,
    *,
    planner_strategy: str,
    planner_model_revision: str,
    planner_prompt_version: str = PLANNER_PROMPT_VERSION,
    composer_version: str = COMPOSER_VERSION,
) -> str:
    payload = {
        "case_id": case.case_id,
        "case_query_sha256": hashlib.sha256(
            case.case_query.encode("utf-8")
        ).hexdigest(),
        "planner_strategy": planner_strategy,
        "planner_model_revision": planner_model_revision,
        "planner_prompt_version": planner_prompt_version,
        "composer_version": composer_version,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class StoredQueryPlan:
    fingerprint: str
    planner_strategy: str
    planner_failure_type: str | None
    plan: StructuredQueryPlan
    queries: tuple[PlannedQuery, ...]


class QueryPlanStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("Query-plan artifact must be a JSON object")
            self.records = parsed

    def get_or_create(
        self,
        case: InferenceCase,
        *,
        planner: QueryPlanner,
        composer: DeterministicQueryComposer,
        configured_strategy: str,
        model_revision: str,
        prompt_version: str = PLANNER_PROMPT_VERSION,
        composer_version: str = COMPOSER_VERSION,
    ) -> StoredQueryPlan:
        fingerprint = query_plan_fingerprint(
            case,
            planner_strategy=configured_strategy,
            planner_model_revision=model_revision,
            planner_prompt_version=prompt_version,
            composer_version=composer_version,
        )
        existing = self.records.get(case.case_id)
        if existing is not None:
            if existing.get("fingerprint") != fingerprint:
                raise ValueError(f"Stored query plan fingerprint changed: {case.case_id}")
            return self._decode(existing)
        result = planner.plan(case.case_query)
        queries = tuple(composer.compose(result.plan))
        if len(queries) < 2:
            raise ValueError("Query composer must produce two primary queries")
        record = {
            "fingerprint": fingerprint,
            "fingerprint_components": {
                "case_id": case.case_id,
                "case_query_sha256": hashlib.sha256(
                    case.case_query.encode("utf-8")
                ).hexdigest(),
                "planner_strategy": configured_strategy,
                "planner_model_revision": model_revision,
                "planner_prompt_version": prompt_version,
                "composer_version": composer_version,
            },
            "planner_strategy": result.strategy,
            "planner_failure_type": result.failure_type,
            "plan": result.plan.to_dict(),
            "queries": [asdict(query) for query in queries],
        }
        self.records[case.case_id] = record
        write_json(self.path, self.records)
        return self._decode(record)

    @staticmethod
    def _decode(record: dict[str, Any]) -> StoredQueryPlan:
        plan_value = record["plan"]
        plan = StructuredQueryPlan(
            case_type=str(plan_value["case_type"]),
            main_claim=str(plan_value["main_claim"]),
            requested_remedies=tuple(plan_value["requested_remedies"]),
            legal_objects=tuple(plan_value["legal_objects"]),
            amounts_or_areas=tuple(plan_value["amounts_or_areas"]),
        )
        return StoredQueryPlan(
            fingerprint=str(record["fingerprint"]),
            planner_strategy=str(record["planner_strategy"]),
            planner_failure_type=record.get("planner_failure_type"),
            plan=plan,
            queries=tuple(PlannedQuery(**query) for query in record["queries"]),
        )
