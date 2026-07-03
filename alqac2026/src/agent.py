"""
Agentic RAG verdict predictor for ALQAC 2026.

Given ONLY a case_query (private-test condition), the agent:
  1. (optional) rewrites the factual query into legal-concept sub-queries,
  2. retrieves candidate law articles from the corpus,
  3. reasons IRAC-style over the retrieved law,
  4. predicts one of 4 labels: A_WIN / PARTIAL_A_WIN / B_WIN / PARTIAL_B_WIN.

MODEL CONSTRAINT (ALQAC 2026 rule): open-weight, < 10B params. NO GPT/Claude/Gemini.
Recommended: Qwen/Qwen2.5-7B-Instruct (best open-weight <14B on Vietnamese legal QA),
alternatives: Qwen/Qwen3-8B, google/gemma-2-9b-it, SeaLLMs/SeaLLMs-v3-7B-Chat.

Few-shot examples come from the PUBLIC test (it is labeled) — this is legitimate:
the public set is the dev set. The private set will not include labels.
"""
from __future__ import annotations
import json
import re

from data_utils import LABELS, Case, Article

SYSTEM_PROMPT = """Bạn là một trợ lý pháp lý chuyên về luật dân sự Việt Nam. \
Nhiệm vụ: dựa trên mô tả tranh chấp và các điều luật liên quan được cung cấp, \
hãy lập luận theo phương pháp IRAC (Issue – Rule – Application – Conclusion) \
và dự đoán kết quả vụ kiện sơ thẩm.

Quy ước:
- A = Nguyên đơn (bên khởi kiện). B = Bị đơn (bên bị kiện).
- Chỉ chọn MỘT trong 4 nhãn:
  - A_WIN: yêu cầu của nguyên đơn được chấp nhận toàn bộ.
  - PARTIAL_A_WIN: yêu cầu của nguyên đơn được chấp nhận MỘT PHẦN.
  - B_WIN: yêu cầu của nguyên đơn bị bác toàn bộ (bị đơn thắng).
  - PARTIAL_B_WIN: bị đơn thắng phần lớn / nguyên đơn chỉ được rất ít.

Trả lời DUY NHẤT bằng một đối tượng JSON, không thêm chữ nào khác:
{"reasoning": "<lập luận ngắn gọn IRAC bằng tiếng Việt>", "label": "<một trong 4 nhãn>"}"""


def build_user_prompt(query: str, retrieved: list[tuple[str, str]],
                      few_shots: list[dict] | None = None) -> str:
    blocks = []
    if few_shots:
        for ex in few_shots:
            blocks.append(
                f"### Ví dụ\nTranh chấp: {ex['query']}\n"
                f"Đáp án: {{\"label\": \"{ex['label']}\"}}\n"
            )
    law_text = "\n\n".join(f"[{doc_id}] {text[:1200]}" for doc_id, text in retrieved)
    blocks.append(
        "### Vụ cần dự đoán\n"
        f"Tranh chấp: {query}\n\n"
        f"Các điều luật liên quan (đã truy hồi):\n{law_text}\n\n"
        "Hãy lập luận IRAC và đưa ra nhãn. Chỉ trả về JSON."
    )
    return "\n".join(blocks)


def build_few_shots(public_cases: list[Case], n_per_class: int = 1) -> list[dict]:
    """Balanced few-shot pool drawn from labeled public cases."""
    pool: dict[str, list[dict]] = {l: [] for l in LABELS}
    for c in public_cases:
        if c.verdict_label in pool and len(pool[c.verdict_label]) < n_per_class:
            pool[c.verdict_label].append({"query": c.case_query, "label": c.verdict_label})
    shots = []
    for l in LABELS:
        shots.extend(pool[l])
    return shots


_LABEL_RE = re.compile("|".join(map(re.escape, LABELS)))


def parse_label(text: str) -> str:
    """Robustly extract the label from model output (handles stray text)."""
    try:
        obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
        if obj.get("label") in LABELS:
            return obj["label"]
    except Exception:
        pass
    m = _LABEL_RE.search(text or "")
    return m.group(0) if m else "PARTIAL_A_WIN"  # majority-class fallback


class VerdictAgent:
    """Wraps a local open-weight LLM via vLLM. Swap to transformers if no vLLM."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 max_tokens: int = 512, temperature: float = 0.0,
                 dtype: str = "bfloat16", gpu_mem: float = 0.9):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model_name, dtype=dtype, gpu_memory_utilization=gpu_mem,
                       max_model_len=8192, trust_remote_code=True)
        self.sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        self.tok = self.llm.get_tokenizer()

    def _chat(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = self.llm.generate([prompt], self.sp)
        return out[0].outputs[0].text

    def predict(self, query: str, retrieved: list[tuple[str, str]],
                few_shots: list[dict] | None = None) -> dict:
        user = build_user_prompt(query, retrieved, few_shots)
        raw = self._chat(user)
        return {"label": parse_label(raw), "raw": raw}


# Optional self-consistency: sample k times at T>0 and majority-vote the label.
def self_consistency(agent: "VerdictAgent", query, retrieved, few_shots, k: int = 5) -> str:
    from collections import Counter
    votes = []
    old_t = agent.sp.temperature
    agent.sp.temperature = 0.7
    for _ in range(k):
        votes.append(agent.predict(query, retrieved, few_shots)["label"])
    agent.sp.temperature = old_t
    return Counter(votes).most_common(1)[0][0]
