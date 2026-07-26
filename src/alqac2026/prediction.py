from __future__ import annotations

import gc
import json
from typing import Protocol

from .schemas import CaseEvidence, InferenceCase, LawEvidence, OutcomeLabel
from .vks import vks_label


SYSTEM_PROMPT = """Bạn là hệ thống dự đoán kết quả vụ án dân sự Việt Nam.
Dựa duy nhất trên mô tả tranh chấp, bằng chứng vụ án và điều luật được cung cấp,
hãy xác định kết quả đối với yêu cầu chính của nguyên đơn.

Nhãn hợp lệ:
- A_WIN: Tòa chấp nhận toàn bộ yêu cầu chính của nguyên đơn.
- PARTIAL_A_WIN: Tòa chấp nhận một phần lớn hơn 50% yêu cầu chính.
- PARTIAL_B_WIN: Tòa chấp nhận một phần không quá 50% yêu cầu chính.
- B_WIN: Tòa bác toàn bộ yêu cầu chính của nguyên đơn.

Không suy diễn tình tiết không có trong evidence. Trả về đúng một JSON object:
{"reasoning":"lập luận ngắn gọn theo Issue-Rule-Application-Conclusion","label":"NHÃN"}
"""


PRIORITY = {
    "vks_opinion": 0,
    "vks_accept": 0,
    "vks_partial": 0,
    "rescue_decision": 1,
    "rescue_reject": 1,
    "rescue_order": 1,
    "court_decision": 2,
    "accepted_claim": 3,
    "rejected_claim": 3,
    "court_reasoning": 4,
    "rescue_reasoning": 4,
    "applied_law": 5,
    "court_fee": 6,
    "dispute_type": 7,
    "original": 8,
    "unknown": 9,
}


class GenerationBackend(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


def build_user_prompt(
    case: InferenceCase,
    case_evidence: list[CaseEvidence],
    law_evidence: list[LawEvidence],
) -> str:
    selected_cases = sorted(
        case_evidence,
        key=lambda item: (PRIORITY.get(item.query_type, 9), -item.score),
    )[:8]
    cases_text = "\n\n".join(
        f"[{item.chunk_id} | {item.query_type}] {item.text[:1400]}"
        for item in selected_cases
    ) or "Không truy hồi được bằng chứng vụ án."
    laws_text = "\n\n".join(
        f"[{item.law_id} | aid={item.aid}] {item.text[:1200]}"
        for item in law_evidence[:5]
    ) or "Không truy hồi được điều luật."
    # The raw procuracy recommendation text is noisy, so it is deliberately kept
    # out of the prompt. The normalised VKS stance is used only as a fallback prior
    # for an unparseable model output (see OutcomePredictor).
    return f"""## Vụ án
case_id: {case.case_id}
case_query: {case.case_query}

## Bằng chứng vụ án
{cases_text}

## Điều luật liên quan
{laws_text}

Hãy xác định yêu cầu chính, đối chiếu evidence với luật, rồi chọn đúng một nhãn.
Chỉ trả về JSON theo schema đã yêu cầu."""


def parse_prediction(text: str) -> tuple[OutcomeLabel, str]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model output does not contain a JSON object")
    payload = json.loads(text[start : end + 1])
    label = OutcomeLabel(payload["label"])
    reasoning = str(payload.get("reasoning", "")).strip()
    if not reasoning:
        raise ValueError("Model output is missing reasoning")
    return label, reasoning


class TransformersQwenBackend:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        revision: str | None = None,
        load_in_4bit: bool = True,
        max_input_tokens: int = 7000,
        max_new_tokens: int = 384,
        thinking: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            trust_remote_code=True,
            revision=revision,
        ).eval()
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.thinking = thinking

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        thinking: bool | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        import torch

        use_thinking = self.thinking if thinking is None else thinking
        new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=use_thinking,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.model.device)
        # temperature > 0 enables sampling (for self-consistency); default is greedy.
        sampling = (
            {"do_sample": True, "temperature": temperature, "top_p": 0.9}
            if temperature and temperature > 0
            else {"do_sample": False}
        )
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                **sampling,
                max_new_tokens=new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_samples(
        self,
        system_prompt: str,
        user_prompt: str,
        n: int,
        temperature: float = 0.7,
        thinking: bool | None = None,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """Draw ``n`` sampled completions in ONE batched call (self-consistency).

        ``num_return_sequences`` runs the samples together on the GPU instead of
        n sequential ``generate`` calls, which is the practical replacement for a
        vLLM continuous batch when only transformers is available.
        """
        import torch

        use_thinking = self.thinking if thinking is None else thinking
        new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=use_thinking
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                num_return_sequences=n,
                max_new_tokens=new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        start = inputs["input_ids"].shape[1]
        return [
            self.tokenizer.decode(row[start:], skip_special_tokens=True) for row in output
        ]

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
    def __init__(self, backend: GenerationBackend):
        self.backend = backend

    def predict(
        self,
        case: InferenceCase,
        case_evidence: list[CaseEvidence],
        law_evidence: list[LawEvidence],
    ) -> tuple[OutcomeLabel, str, str]:
        prompt = build_user_prompt(case, case_evidence, law_evidence)
        raw = self.backend.generate(SYSTEM_PROMPT, prompt)
        try:
            label, reasoning = parse_prediction(raw)
            return label, reasoning, raw
        except (KeyError, ValueError, json.JSONDecodeError) as first_error:
            repair = (
                "Đầu ra trước không hợp lệ. Hãy sửa thành đúng một JSON object có hai "
                "trường reasoning và label; label phải thuộc A_WIN, PARTIAL_A_WIN, "
                f"PARTIAL_B_WIN, B_WIN. Đầu ra lỗi:\n{raw[:1500]}"
            )
            repaired = self.backend.generate(SYSTEM_PROMPT, repair)
            try:
                label, reasoning = parse_prediction(repaired)
                return label, reasoning, f"{raw}\n---REPAIR---\n{repaired}"
            except (KeyError, ValueError, json.JSONDecodeError) as second_error:
                # Rather than fail the case, defer to the procuracy stance when it
                # is unambiguous. Only used as a fallback for an unparseable model
                # output, never an override.
                fallback = vks_label(" ".join(item.text for item in case_evidence))
                if fallback is not None:
                    return (
                        fallback,
                        "Fallback theo đề nghị Viện Kiểm Sát (model không trả JSON hợp lệ).",
                        f"{raw}\n---REPAIR---\n{repaired}\n---VKS_FALLBACK---",
                    )
                raise ValueError(
                    f"Prediction output failed validation twice: {first_error}; {second_error}"
                ) from second_error


def create_predictor(config: dict) -> OutcomePredictor:
    backend = TransformersQwenBackend(
        model_name=config["model_name"],
        revision=config.get("revision"),
        load_in_4bit=bool(config.get("load_in_4bit", True)),
        max_input_tokens=int(config.get("max_input_tokens", 7000)),
        max_new_tokens=int(config.get("max_new_tokens", 384)),
        thinking=bool(config.get("thinking", False)),
    )
    return OutcomePredictor(backend)
