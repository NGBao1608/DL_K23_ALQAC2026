from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from typing import Protocol

from .schemas import CaseEvidence, InferenceCase, LawEvidence, OutcomeLabel


BASELINE_SYSTEM_PROMPT = """Bạn là hệ thống dự đoán kết quả vụ án dân sự Việt Nam.
Dựa duy nhất trên mô tả tranh chấp, bằng chứng vụ án và điều luật được cung cấp,
hãy xác định kết quả đối với yêu cầu chính của nguyên đơn.

Nhãn hợp lệ:
- A_WIN: Tòa chấp nhận toàn bộ yêu cầu chính của nguyên đơn.
- PARTIAL_A_WIN: Tòa chấp nhận một phần lớn hơn 50% yêu cầu chính.
- PARTIAL_B_WIN: Tòa chấp nhận một phần không quá 50% yêu cầu chính.
- B_WIN: Tòa bác toàn bộ yêu cầu chính của nguyên đơn.

Không suy diễn tình tiết không có trong evidence. Trả về đúng một JSON object:
{"main_claim":"yêu cầu chính","accepted_scope":"phần được chấp nhận",
"acceptance_ratio":0.6,"reasoning":"lập luận ngắn gọn","label":"NHÃN"}
"""


DECISION_FIRST_SYSTEM_PROMPT = """Bạn là hệ thống phân loại kết quả vụ án dân sự Việt Nam.
Chỉ sử dụng case_query, case evidence và law evidence được cung cấp. Không bổ sung
tình tiết bên ngoài evidence.

Thực hiện lần lượt:
1. Xác định chính xác yêu cầu chính của nguyên đơn trong case_query.
2. Tìm phần Tuyên xử, Quyết định hoặc câu thể hiện chấp nhận/bác yêu cầu trong case
   evidence. Đây là nguồn ưu tiên cao nhất khi evidence có xung đột.
3. Tách yêu cầu chính khỏi án phí, thủ tục tố tụng, yêu cầu độc lập của người khác và
   các vấn đề phụ không quyết định thắng-thua của nguyên đơn.
4. Đối chiếu phần được yêu cầu với phần được Tòa chấp nhận. Khi có số tiền, diện tích
   hoặc nhiều cấu phần, hãy tính hoặc ước lượng tỷ lệ được chấp nhận.

Quy tắc phân loại nội bộ của hệ thống:
- A_WIN: toàn bộ yêu cầu chính của nguyên đơn được chấp nhận.
- PARTIAL_A_WIN: chỉ một phần được chấp nhận và phần đó lớn hơn 50% yêu cầu chính.
- PARTIAL_B_WIN: chỉ một phần được chấp nhận và phần đó không quá 50% yêu cầu chính.
- B_WIN: toàn bộ yêu cầu chính bị bác hoặc nguyên đơn không nhận được phần nào.

Nếu không đủ số liệu định lượng, đánh giá số lượng và tầm quan trọng của các cấu phần
được chấp nhận. Không chọn A_WIN chỉ vì evidence có từ "chấp nhận" nếu quyết định thực
tế chỉ chấp nhận một phần.

Trả về đúng một JSON object, không markdown và không văn bản ngoài JSON:
{"main_claim":"yêu cầu chính","accepted_scope":"phần được chấp nhận",
"acceptance_ratio":0.6,"reasoning":"lập luận ngắn gọn","label":"NHÃN"}

acceptance_ratio phải nằm trong [0, 1] nếu có thể xác định; dùng null khi evidence
không đủ để định lượng. label phải nhất quán với acceptance_ratio và quy tắc trên.
"""

DECISION_FIRST_COMPACT_SYSTEM_PROMPT = """Bạn là hệ thống phân loại kết quả vụ án dân sự Việt Nam.
Chỉ sử dụng case_query, case evidence và law evidence được cung cấp. Không bổ sung
tình tiết bên ngoài evidence.

Thực hiện lần lượt:
1. Xác định yêu cầu chính của nguyên đơn trong case_query.
2. Ưu tiên phần Tuyên xử, Quyết định hoặc câu thể hiện chấp nhận/bác yêu cầu trong
   case evidence khi các nguồn xung đột.
3. Loại án phí, thủ tục, yêu cầu độc lập của người khác và vấn đề phụ khỏi tỷ lệ.
4. So sánh phạm vi được yêu cầu với phạm vi được Tòa chấp nhận.

Quy tắc nhãn:
- A_WIN: chấp nhận toàn bộ yêu cầu chính.
- PARTIAL_A_WIN: chấp nhận một phần lớn hơn 50%.
- PARTIAL_B_WIN: chấp nhận một phần không quá 50%.
- B_WIN: bác toàn bộ yêu cầu chính hoặc nguyên đơn không nhận được phần nào.

Trả về ngay đúng một JSON object trên một dòng, không markdown, không lời dẫn và
không văn bản sau dấu đóng object:
{"main_claim":"yêu cầu chính","accepted_scope":"phần được chấp nhận",
"acceptance_ratio":0.6,"reasoning":"tối đa 20 từ","label":"NHÃN"}

acceptance_ratio nằm trong [0, 1], hoặc null khi không thể định lượng. label chỉ được
là A_WIN, PARTIAL_A_WIN, PARTIAL_B_WIN hoặc B_WIN và phải nhất quán với tỷ lệ.
"""


SYSTEM_PROMPTS = {
    "baseline_v1": BASELINE_SYSTEM_PROMPT,
    "decision_first_v2": DECISION_FIRST_SYSTEM_PROMPT,
    "decision_first_v3": DECISION_FIRST_COMPACT_SYSTEM_PROMPT,
}

# Backward-compatible default used by direct unit/integration construction.
SYSTEM_PROMPT = BASELINE_SYSTEM_PROMPT


PRIORITY = {
    "operative_verdict": 0,
    "adaptive_missing_scope": 1,
    "remedy_scope": 2,
    "court_decision": 3,
    "accepted_claim": 4,
    "rejected_claim": 5,
    "court_reasoning": 6,
    "applied_law": 7,
    "court_fee": 8,
    "dispute_type": 9,
    "original": 10,
    "unknown": 11,
}


class GenerationBackend(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


@dataclass(frozen=True, slots=True)
class PredictionDiagnostics:
    output_repair_used: bool = False
    output_verification: str = "not_required"


def build_user_prompt(
    case: InferenceCase,
    case_evidence: list[CaseEvidence],
    law_evidence: list[LawEvidence],
    case_evidence_chars: int = 1400,
    law_evidence_chars: int = 1200,
    *,
    tokenizer=None,
    evidence_token_budget: int | None = None,
    law_top_k: int = 5,
    case_token_share: float = 0.65,
) -> str:
    selected_cases = sorted(
        case_evidence,
        key=lambda item: (PRIORITY.get(item.query_type, 8), -item.score),
    )[:8]
    selected_laws = law_evidence[:law_top_k]
    if evidence_token_budget is None:
        cases_text = "\n\n".join(
            f"[chunk_id={item.chunk_id} | query_type={item.query_type} | "
            f"score={item.score!r}] "
            f"{item.text[:case_evidence_chars]}"
            for item in selected_cases
        ) or "Không truy hồi được bằng chứng vụ án."
        laws_text = "\n\n".join(
            f"[law_id={item.law_id} | aid={item.aid} | "
            f"article_number={item.article_number!r} | score={item.score!r}] "
            f"{item.text[:law_evidence_chars]}"
            for item in selected_laws
        ) or "Không truy hồi được điều luật."
    else:
        if evidence_token_budget < 0:
            raise ValueError("evidence_token_budget must be non-negative")
        if not 0 < case_token_share < 1:
            raise ValueError("case_token_share must be between zero and one")
        encode, decode = _token_codec(tokenizer)
        case_blocks = [
            (
                f"[chunk_id={item.chunk_id} | query_type={item.query_type} | "
                f"score={item.score!r}]",
                item.text,
            )
            for item in selected_cases
        ]
        law_blocks = [
            (
                f"[law_id={item.law_id} | aid={item.aid} | "
                f"article_number={item.article_number!r} | score={item.score!r}]",
                item.text,
            )
            for item in selected_laws
        ]
        case_budget = int(evidence_token_budget * case_token_share)
        law_budget = evidence_token_budget - case_budget
        cases_text, case_used = _fit_blocks(
            case_blocks, case_budget, encode=encode, decode=decode
        )
        laws_text, law_used = _fit_blocks(
            law_blocks, law_budget, encode=encode, decode=decode
        )
        if case_used < case_budget and law_blocks:
            laws_text, law_used = _fit_blocks(
                law_blocks,
                law_budget + case_budget - case_used,
                encode=encode,
                decode=decode,
            )
        if law_used < law_budget and case_blocks:
            cases_text, _ = _fit_blocks(
                case_blocks,
                case_budget + law_budget - law_used,
                encode=encode,
                decode=decode,
            )
        cases_text = cases_text or "Không truy hồi được bằng chứng vụ án."
        laws_text = laws_text or "Không truy hồi được điều luật."
    return f"""## Vụ án
case_query: {case.case_query}

## Bằng chứng vụ án
{cases_text}

## Điều luật liên quan
{laws_text}

Hãy xác định yêu cầu chính, đối chiếu evidence với luật, rồi chọn đúng một nhãn.
Chỉ trả về JSON theo schema đã yêu cầu."""


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


def _fit_blocks(blocks, budget: int, *, encode, decode) -> tuple[str, int]:
    if budget <= 0:
        return "", 0
    rendered = []
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


@dataclass(frozen=True, slots=True)
class ParsedPrediction:
    label: OutcomeLabel
    claimed_label: OutcomeLabel
    reasoning: str
    main_claim: str
    accepted_scope: str
    acceptance_ratio: float | None
    inconsistent: bool


def _label_from_ratio(ratio: float) -> OutcomeLabel:
    if ratio == 1.0:
        return OutcomeLabel.A_WIN
    if ratio == 0.0:
        return OutcomeLabel.B_WIN
    if ratio > 0.5:
        return OutcomeLabel.PARTIAL_A_WIN
    return OutcomeLabel.PARTIAL_B_WIN


def parse_structured_prediction(text: str) -> ParsedPrediction:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model output does not contain a JSON object")
    payload = json.loads(text[start : end + 1])
    required = {
        "main_claim",
        "accepted_scope",
        "acceptance_ratio",
        "reasoning",
        "label",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Model output is missing fields: {sorted(missing)}")
    claimed_label = OutcomeLabel(payload["label"])
    reasoning = str(payload.get("reasoning", "")).strip()
    if not reasoning:
        raise ValueError("Model output is missing reasoning")
    main_claim = str(payload.get("main_claim", "")).strip()
    accepted_scope = str(payload.get("accepted_scope", "")).strip()
    if not main_claim or not accepted_scope:
        raise ValueError("main_claim and accepted_scope must be non-empty strings")
    ratio_value = payload.get("acceptance_ratio")
    if ratio_value is None:
        ratio = None
        final_label = claimed_label
    else:
        if isinstance(ratio_value, bool) or not isinstance(ratio_value, (int, float)):
            raise ValueError("acceptance_ratio must be a number or null")
        ratio = float(ratio_value)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("acceptance_ratio must be within [0, 1]")
        final_label = _label_from_ratio(ratio)
    return ParsedPrediction(
        label=final_label,
        claimed_label=claimed_label,
        reasoning=reasoning,
        main_claim=main_claim,
        accepted_scope=accepted_scope,
        acceptance_ratio=ratio,
        inconsistent=final_label is not claimed_label,
    )


def parse_prediction(text: str) -> tuple[OutcomeLabel, str]:
    parsed = parse_structured_prediction(text)
    return parsed.label, parsed.reasoning


class TransformersQwenBackend:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        revision: str | None = None,
        load_in_4bit: bool = True,
        max_input_tokens: int = 7000,
        max_new_tokens: int = 384,
        oom_max_input_tokens: int | None = None,
        oom_max_new_tokens: int | None = None,
        attn_implementation: str | None = None,
        cache_implementation: str | None = None,
        thinking: bool = False,
        adapter_path: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if max_input_tokens <= 0 or max_new_tokens <= 0:
            raise ValueError("Generation token limits must be positive")
        if oom_max_input_tokens is not None and not (
            0 < oom_max_input_tokens <= max_input_tokens
        ):
            raise ValueError(
                "oom_max_input_tokens must be positive and no greater than "
                "max_input_tokens"
            )
        if oom_max_new_tokens is not None and not (
            0 < oom_max_new_tokens <= max_new_tokens
        ):
            raise ValueError(
                "oom_max_new_tokens must be positive and no greater than "
                "max_new_tokens"
            )
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "quantization_config": quantization_config,
            "trust_remote_code": True,
            "revision": revision,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        ).eval()
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path).eval()
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.oom_max_input_tokens = oom_max_input_tokens
        self.oom_max_new_tokens = oom_max_new_tokens
        self.cache_implementation = cache_implementation
        self.thinking = thinking

    def _render_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )

    def count_input_tokens(self, system_prompt: str, user_prompt: str) -> int:
        prompt = self._render_prompt(system_prompt, user_prompt)
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))

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
        with torch.inference_mode():
            generation_kwargs = {
                "do_sample": False,
                "max_new_tokens": self.max_new_tokens,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if self.cache_implementation and torch.cuda.is_available():
                generation_kwargs["cache_implementation"] = (
                    self.cache_implementation
                )
            output = self.model.generate(
                **inputs,
                **generation_kwargs,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def activate_oom_recovery(self) -> bool:
        """Switch subsequent generations to the configured compact profile."""
        changed = False
        if (
            self.oom_max_input_tokens is not None
            and self.max_input_tokens != self.oom_max_input_tokens
        ):
            self.max_input_tokens = self.oom_max_input_tokens
            changed = True
        if (
            self.oom_max_new_tokens is not None
            and self.max_new_tokens != self.oom_max_new_tokens
        ):
            self.max_new_tokens = self.oom_max_new_tokens
            changed = True
        if self.cache_implementation != "offloaded":
            self.cache_implementation = "offloaded"
            changed = True
        return changed

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


class OutcomePredictor:
    def __init__(
        self,
        backend: GenerationBackend,
        system_prompt: str = SYSTEM_PROMPT,
        case_evidence_chars: int = 1400,
        law_evidence_chars: int = 1200,
        max_input_tokens: int = 6144,
        context_law_top_k: int = 5,
        verification_reserve_tokens: int = 512,
        oom_max_input_tokens: int | None = None,
        oom_context_law_top_k: int | None = None,
    ):
        if case_evidence_chars <= 0 or law_evidence_chars <= 0:
            raise ValueError("Evidence character budgets must be positive")
        if max_input_tokens <= 0 or context_law_top_k <= 0:
            raise ValueError("Prediction context limits must be positive")
        if oom_max_input_tokens is not None and not (
            0 < oom_max_input_tokens <= max_input_tokens
        ):
            raise ValueError(
                "oom_max_input_tokens must be positive and no greater than "
                "max_input_tokens"
            )
        if oom_context_law_top_k is not None and not (
            0 < oom_context_law_top_k <= context_law_top_k
        ):
            raise ValueError(
                "oom_context_law_top_k must be positive and no greater than "
                "context_law_top_k"
            )
        self.backend = backend
        self.system_prompt = system_prompt
        self.case_evidence_chars = case_evidence_chars
        self.law_evidence_chars = law_evidence_chars
        self.max_input_tokens = max_input_tokens
        self.context_law_top_k = context_law_top_k
        self.verification_reserve_tokens = verification_reserve_tokens
        self.oom_max_input_tokens = oom_max_input_tokens
        self.oom_context_law_top_k = oom_context_law_top_k
        self.last_diagnostics = PredictionDiagnostics()

    def activate_oom_recovery(self) -> bool:
        """Use a smaller prompt while retaining the exact prepared evidence."""
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
    ) -> str:
        tokenizer = getattr(self.backend, "tokenizer", None)
        count_tokens = getattr(self.backend, "count_input_tokens", None)
        if tokenizer is None or count_tokens is None:
            return build_user_prompt(
                case,
                case_evidence,
                law_evidence,
                case_evidence_chars=self.case_evidence_chars,
                law_evidence_chars=self.law_evidence_chars,
                law_top_k=self.context_law_top_k,
            )
        protected = build_user_prompt(
            case,
            [],
            [],
            tokenizer=tokenizer,
            evidence_token_budget=0,
            law_top_k=self.context_law_top_k,
        )
        protected_tokens = int(count_tokens(self.system_prompt, protected))
        evidence_budget = (
            self.max_input_tokens
            - protected_tokens
            - self.verification_reserve_tokens
        )
        if evidence_budget < 0:
            raise ValueError(
                "System prompt and case query exceed the protected token budget"
            )
        return build_user_prompt(
            case,
            case_evidence,
            law_evidence,
            tokenizer=tokenizer,
            evidence_token_budget=evidence_budget,
            law_top_k=self.context_law_top_k,
        )

    def _parse_or_repair(
        self, raw: str, original_prompt: str
    ) -> tuple[ParsedPrediction, str, bool]:
        try:
            return parse_structured_prediction(raw), raw, False
        except (KeyError, ValueError, json.JSONDecodeError) as first_error:
            repair = f"""{original_prompt}

## Yêu cầu sửa định dạng
Đầu ra trước không hợp lệ. Đọc lại chính evidence ở trên và trả lời lại ngay bằng
đúng một JSON object có main_claim, accepted_scope, acceptance_ratio, reasoning và
label. reasoning tối đa 20 từ. Không markdown, không lời dẫn và không văn bản sau
dấu đóng object."""
            repaired = self.backend.generate(self.system_prompt, repair)
            try:
                parsed = parse_structured_prediction(repaired)
                return parsed, f"{raw}\n---REPAIR---\n{repaired}", True
            except (KeyError, ValueError, json.JSONDecodeError) as second_error:
                raise ValueError(
                    "Prediction output failed validation twice: "
                    f"{first_error}; {second_error}"
                ) from second_error

    def predict(
        self,
        case: InferenceCase,
        case_evidence: list[CaseEvidence],
        law_evidence: list[LawEvidence],
    ) -> tuple[OutcomeLabel, str, str]:
        self.last_diagnostics = PredictionDiagnostics()
        prompt = self._build_prompt(case, case_evidence, law_evidence)
        raw = self.backend.generate(self.system_prompt, prompt)
        parsed, combined_raw, repair_used = self._parse_or_repair(raw, prompt)
        needs_verification = parsed.inconsistent or parsed.label in {
            OutcomeLabel.PARTIAL_A_WIN,
            OutcomeLabel.PARTIAL_B_WIN,
        }
        verification_status = "not_required"
        if needs_verification:
            verification = self._build_verification_prompt(prompt, combined_raw)
            try:
                verified_raw = self.backend.generate(self.system_prompt, verification)
                parsed = parse_structured_prediction(verified_raw)
                combined_raw = (
                    f"{combined_raw}\n---VERIFICATION---\n{verified_raw}"
                )
                verification_status = "passed"
            except (KeyError, ValueError, json.JSONDecodeError):
                combined_raw = f"{combined_raw}\n---VERIFICATION_FAILED---"
                verification_status = "failed"
        self.last_diagnostics = PredictionDiagnostics(
            output_repair_used=repair_used,
            output_verification=verification_status,
        )
        return parsed.label, parsed.reasoning, combined_raw

    def _build_verification_prompt(self, prompt: str, raw: str) -> str:
        prefix = f"""{prompt}

## Kết quả bước đầu cần kiểm tra
"""
        suffix = """

Kiểm tra lại duy nhất tỷ lệ phần yêu cầu chính được chấp nhận và ranh giới >50%.
Trả về lại đúng JSON schema đã yêu cầu."""
        tokenizer = getattr(self.backend, "tokenizer", None)
        count_tokens = getattr(self.backend, "count_input_tokens", None)
        if tokenizer is None or count_tokens is None:
            return f"{prefix}{raw[-1800:]}{suffix}"
        encode, decode = _token_codec(tokenizer)
        base_tokens = int(count_tokens(self.system_prompt, f"{prefix}{suffix}"))
        available = max(0, self.max_input_tokens - base_tokens)
        raw_tail = decode(encode(raw)[-available:]).strip() if available else ""
        verification = f"{prefix}{raw_tail}{suffix}"
        while (
            raw_tail
            and int(count_tokens(self.system_prompt, verification))
            > self.max_input_tokens
        ):
            tail_tokens = encode(raw_tail)
            raw_tail = decode(tail_tokens[1:]).strip()
            verification = f"{prefix}{raw_tail}{suffix}"
        if int(count_tokens(self.system_prompt, verification)) > self.max_input_tokens:
            raise ValueError("Verifier instructions exceed the input token budget")
        return verification


def create_predictor(config: dict) -> OutcomePredictor:
    backend = TransformersQwenBackend(
        model_name=config["model_name"],
        revision=config.get("revision"),
        load_in_4bit=bool(config.get("load_in_4bit", True)),
        max_input_tokens=int(config.get("max_input_tokens", 7000)),
        max_new_tokens=int(config.get("max_new_tokens", 384)),
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
    prompt_variant = config.get("prompt_variant", "baseline_v1")
    if prompt_variant not in SYSTEM_PROMPTS:
        raise ValueError(f"Unsupported prediction prompt_variant: {prompt_variant}")
    return OutcomePredictor(
        backend,
        system_prompt=SYSTEM_PROMPTS[prompt_variant],
        case_evidence_chars=int(config.get("case_evidence_chars", 1400)),
        law_evidence_chars=int(config.get("law_evidence_chars", 1200)),
        max_input_tokens=int(config.get("max_input_tokens", 6144)),
        context_law_top_k=int(config.get("context_law_top_k", 5)),
        verification_reserve_tokens=int(
            config.get("verification_reserve_tokens", 512)
        ),
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
    )
