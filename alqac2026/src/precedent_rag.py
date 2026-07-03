"""
Precedent-RAG for ALQAC 2026 outcome prediction (the 70% component).

Idea (Athena / precedent-enhanced LJP): the 50 labeled public cases form a
"precedent base". For each test case we retrieve the most similar precedents by
case_query embedding and feed their outcomes to the LLM as *dynamic* few-shot,
then prompt it to reason by legal syllogism and pick one of 4 labels.

Validated offline: leave-one-out k-NN on case_query alone already beats the
majority baseline (Macro-F1 0.30 vs 0.138), so the similar-case signal is real.

Open-weight only: embedding via sentence-transformers (Vietnamese_Embedding),
reasoning via the same <10B LLM already loaded. No proprietary APIs.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass

LABELS = ["A_WIN", "PARTIAL_A_WIN", "B_WIN", "PARTIAL_B_WIN"]

SYSTEM_PROMPT = """Bạn là trợ lý pháp lý về luật dân sự Việt Nam. Hãy dự đoán kết quả vụ kiện sơ thẩm \
bằng phương pháp tam đoạn luận pháp lý:
- Đại tiền đề: quy định pháp luật áp dụng.
- Tiểu tiền đề: tình tiết của vụ án (từ bằng chứng).
- Kết luận: đối chiếu để xác định bên thắng.

A = Nguyên đơn, B = Bị đơn. Chọn ĐÚNG MỘT nhãn:
- A_WIN: nguyên đơn được chấp nhận TOÀN BỘ yêu cầu.
- PARTIAL_A_WIN: nguyên đơn được chấp nhận MỘT PHẦN.
- B_WIN: yêu cầu của nguyên đơn bị bác TOÀN BỘ.
- PARTIAL_B_WIN: nguyên đơn chỉ được rất ít / bị đơn thắng phần lớn.
Phân biệt kỹ TOÀN BỘ và MỘT PHẦN.

Tham khảo các vụ tương tự để cảm nhận xu hướng, nhưng quyết định dựa trên bằng chứng và luật của vụ hiện tại.
Trả về DUY NHẤT JSON: {"reasoning":"<tam đoạn luận ngắn>","label":"<nhãn>"}"""


@dataclass
class Precedent:
    case_id: str
    query: str
    label: str
    sim: float = 0.0


class PrecedentRetriever:
    """Embeds the labeled precedent base once; retrieves similar cases per query."""

    def __init__(self, base: list[tuple[str, str, str]],
                 model_name: str = "AITeamVN/Vietnamese_Embedding"):
        # base: list of (case_id, case_query, label)
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self.ids = [b[0] for b in base]
        self.queries = [b[1] for b in base]
        self.labels = [b[2] for b in base]
        self.model = SentenceTransformer(model_name)
        self.emb = self.model.encode(self.queries, normalize_embeddings=True,
                                     show_progress_bar=True).astype("float32")
        self._np = np

    def search(self, query: str, k: int = 4, exclude_id: str | None = None) -> list[Precedent]:
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        sims = (self.emb @ q[0])
        order = self._np.argsort(-sims)
        out = []
        for j in order:
            if self.ids[j] == exclude_id:        # leave-one-out for public-set eval
                continue
            out.append(Precedent(self.ids[j], self.queries[j], self.labels[j], float(sims[j])))
            if len(out) >= k:
                break
        return out


def build_precedent_fewshot(precedents: list[Precedent]) -> str:
    lines = ["Một số vụ án tương tự đã có phán quyết (tham khảo):"]
    for i, p in enumerate(precedents, 1):
        lines.append(f"[Tiền lệ {i}] Tranh chấp: {p.query.strip()}\n→ Kết quả: {p.label}")
    return "\n".join(lines)


def build_prompt(query: str, evidence_ctx: list[tuple[str, str]],
                 law_ctx: list[tuple[str, str]], precedents: list[Precedent]) -> str:
    blocks = [build_precedent_fewshot(precedents), ""]
    blocks.append("### Vụ cần dự đoán")
    blocks.append(f"Tranh chấp: {query}\n")
    if evidence_ctx:
        ev = "\n".join(f"- {t[:600]}" for _, t in evidence_ctx[:8])
        blocks.append(f"Bằng chứng từ hồ sơ vụ án:\n{ev}\n")
    if law_ctx:
        law = "\n".join(f"- [{d}] {t[:400]}" for d, t in law_ctx[:5])
        blocks.append(f"Điều luật liên quan:\n{law}\n")
    blocks.append("Hãy lập luận tam đoạn luận rồi chọn 1 nhãn. Chỉ trả JSON.")
    return "\n".join(blocks)


_LABEL_RE = re.compile("|".join(map(re.escape, LABELS)))


def parse_label(text: str) -> str:
    try:
        obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
        if obj.get("label") in LABELS:
            return obj["label"]
    except Exception:
        pass
    m = _LABEL_RE.search(text or "")
    return m.group(0) if m else "PARTIAL_A_WIN"
