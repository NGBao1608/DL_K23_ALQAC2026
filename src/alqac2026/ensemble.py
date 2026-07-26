"""Multi-signal outcome ensemble (config-toggleable).

Combines several independent outcome signals: precedent case-based reasoning,
an LLM-extracted procuracy (VKS) stance, dual-advocate debate, self-consistency
voting and a reasoning ("thinking") adjudicator. Each signal is a separate
toggle, so any subset can be enabled from configuration.

The module is backend-agnostic: callers pass a ``gen`` callable
``gen(system, user, *, thinking=False, temperature=None, max_new=512) -> str``.
No gold field is read at inference; precedents surface other cases' labels and
always exclude the case being predicted.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .prediction import SYSTEM_PROMPT, parse_prediction
from .schemas import OutcomeLabel


LABELS = ("A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN")
LABEL_GLOSS = (
    "A_WIN = chấp nhận toàn bộ yêu cầu chính; PARTIAL_A_WIN = chấp nhận > 50%; "
    "PARTIAL_B_WIN = chấp nhận <= 50%; B_WIN = bác toàn bộ yêu cầu chính."
)

_LABEL_LINE = re.compile(r"NH[ÃA]N\s*[:\-]?\s*([A-Z_]+)", re.IGNORECASE)
_LABEL_TOKEN = re.compile(r"[A-Z_]+")
_STANCE_TO_LABEL = {
    "CHAP_NHAN": "A_WIN",
    "MOT_PHAN": "PARTIAL_A_WIN",
    "BAC": "B_WIN",
}
_STANCE_TEXT = {
    "A_WIN": "đề nghị chấp nhận toàn bộ yêu cầu",
    "PARTIAL_A_WIN": "đề nghị chấp nhận một phần yêu cầu",
    "B_WIN": "đề nghị bác yêu cầu khởi kiện",
}


class Generate(Protocol):
    def __call__(
        self,
        system: str,
        user: str,
        *,
        thinking: bool = False,
        temperature: float | None = None,
        max_new: int = 512,
    ) -> str: ...


class Embedder(Protocol):
    def encode(self, texts, normalize_embeddings: bool = True, convert_to_numpy: bool = True): ...


@dataclass
class OutcomeConfig:
    use_precedents: bool = False
    num_precedents: int = 3
    use_vks_stance: bool = False
    self_consistency: int = 1
    thinking: bool = False
    use_debate: bool = False


def parse_label(text: str) -> str | None:
    """Extract a whole-token label; 'B_WIN' must not match inside 'PARTIAL_B_WIN'."""
    upper = text.upper()
    line = _LABEL_LINE.search(upper)
    if line and line.group(1) in LABELS:
        return line.group(1)
    tokens = [tok for tok in _LABEL_TOKEN.findall(upper) if tok in LABELS]
    return tokens[-1] if tokens else None


class PrecedentBank:
    """Case-based reasoning memory over the labelled public cases.

    ``cases`` are dicts with ``case_id``, ``query``, ``gold`` and optional
    ``reasoning``. Retrieval embeds an arbitrary query, so it works for private
    cases too (which are absent from the bank); passing ``exclude_case_id`` drops
    the queried case, so predicting a public case never sees its own gold label.
    """

    def __init__(self, embedder: Embedder, cases: list[dict]):
        self.embedder = embedder
        self.cases = cases
        self._vectors = np.asarray(
            embedder.encode(
                [c["query"] for c in cases],
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
            dtype="float32",
        )

    def retrieve(
        self, query: str, k: int = 3, exclude_case_id: str | None = None
    ) -> list[dict]:
        query_vector = np.asarray(
            self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True),
            dtype="float32",
        )[0]
        order = np.argsort(-(self._vectors @ query_vector))
        out: list[dict] = []
        for i in order:
            if exclude_case_id is not None and self.cases[i]["case_id"] == exclude_case_id:
                continue
            out.append(self.cases[i])
            if len(out) >= k:
                break
        return out

    def block(self, query: str, k: int = 3, exclude_case_id: str | None = None) -> str:
        lines = []
        for p in self.retrieve(query, k, exclude_case_id):
            reason = f"; Lý do: {p['reasoning'][:160]}" if p.get("reasoning") else ""
            lines.append(f"- Vụ tương tự: {p['query'][:160]} => Kết quả: {p['gold']}{reason}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Prompt builders
# --------------------------------------------------------------------------- #
_VKS_SYSTEM = (
    "Bạn phân tích hồ sơ vụ án dân sự. Tìm phần ĐẠI DIỆN VIỆN KIỂM SÁT phát biểu hoặc "
    "đề nghị Hội đồng xét xử, và xác định lập trường của họ."
)
# The adjudicator reuses the single-pass predictor's system prompt; the ensemble
# only enriches its user message with extra context, so each enabled signal adds
# to that base prompt rather than replacing it.
_ADJ_SYSTEM = SYSTEM_PROMPT


def _vks_user(evidence: str) -> str:
    return (
        f"BẰNG CHỨNG:\n{evidence[:5000]}\n\nTrả về đúng một JSON object: "
        '{"trich":"câu VKS đề nghị nếu có, rỗng nếu không","lap_truong":'
        '"CHAP_NHAN|MOT_PHAN|BAC|KHONG_RO"}'
    )


def _advocate_system(side: str) -> str:
    who = "nguyên đơn" if side == "A" else "bị đơn"
    goal = "thắng kiện" if side == "A" else "bác yêu cầu khởi kiện"
    return (
        f"Bạn là luật sư của {who}. Nêu 3-4 lập luận mạnh nhất, dựa trên tình tiết và "
        f"pháp luật Việt Nam, để {goal}."
    )


EV_CHARS = 6000


def baseline_user(case: dict) -> str:
    """The single-pass predictor's user prompt.

    With no ensemble signals enabled the adjudicator receives this exact prompt
    and, under greedy decoding, the same prediction as the single-pass predictor.
    Each enabled signal then adds context on top of this base.
    """
    ev = case["ev"][:EV_CHARS] if case["ev"] else "(không truy hồi được đoạn nào)"
    return (
        "## Vụ án\ncase_id: %s\ncase_query: %s\n\n"
        "## Bằng chứng truy hồi (đoạn quyết định ưu tiên lên đầu)\n%s\n\n"
        "Xác định yêu cầu chính rồi chọn đúng một nhãn. Chỉ trả JSON."
    ) % (case["case_id"], case["query"], ev)


def _adjudicator_user(
    case: dict, precedent_block: str, vks_hint: str, arg_a: str, arg_b: str
) -> str:
    # No signals -> identical to baseline_user. Each enabled signal inserts one
    # extra block between the case header and the evidence.
    extra = ""
    if precedent_block:
        extra += f"## Các vụ tương tự đã xử (tham khảo)\n{precedent_block}\n\n"
    if vks_hint:
        extra += f"## Quan điểm Viện Kiểm Sát\nViện Kiểm Sát {vks_hint}.\n\n"
    if arg_a or arg_b:
        extra += f"## Lập luận nguyên đơn\n{arg_a}\n\n## Lập luận bị đơn\n{arg_b}\n\n"
    if not extra:
        return baseline_user(case)
    ev = case["ev"][:EV_CHARS] if case["ev"] else "(không truy hồi được đoạn nào)"
    return (
        f"## Vụ án\ncase_id: {case['case_id']}\ncase_query: {case['query']}\n\n"
        f"{extra}## Bằng chứng truy hồi (đoạn quyết định ưu tiên lên đầu)\n{ev}\n\n"
        "Xác định yêu cầu chính rồi chọn đúng một nhãn. Chỉ trả JSON."
    )


def extract_vks_stance(gen: Generate, evidence: str) -> str:
    """Return an outcome label implied by the procuracy stance, or '' if unclear."""
    try:
        raw = gen(_VKS_SYSTEM, _vks_user(evidence), max_new=200)
        payload = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        return _STANCE_TO_LABEL.get(payload.get("lap_truong", ""), "")
    except (ValueError, KeyError, json.JSONDecodeError):
        return ""


def predict_outcome(
    case: dict,
    gen: Generate,
    config: OutcomeConfig,
    precedent_bank: PrecedentBank | None = None,
    gen_n: Callable[..., list[str]] | None = None,
) -> str:
    """Predict one case's 4-way label using the enabled signals.

    When ``gen_n`` is given (a batched sampler returning N completions in one call)
    and self-consistency > 1, all samples are drawn in a single GPU batch.
    """
    precedent_block = (
        precedent_bank.block(
            case["query"], config.num_precedents, exclude_case_id=case.get("case_id")
        )
        if config.use_precedents and precedent_bank is not None
        else ""
    )
    vks_hint = ""
    if config.use_vks_stance:
        label = extract_vks_stance(gen, case["ev"])
        vks_hint = _STANCE_TEXT.get(label, "")

    arg_a = arg_b = ""
    if config.use_debate:
        brief = f"HỒ SƠ:\n{case['query']}\n{case['ev'][:3000]}"
        arg_a = gen(_advocate_system("A"), brief, max_new=300)
        arg_b = gen(_advocate_system("B"), brief, max_new=300)

    samples = max(1, config.self_consistency)
    max_new = 1024 if config.thinking else 512
    user = _adjudicator_user(case, precedent_block, vks_hint, arg_a, arg_b)
    if samples > 1 and gen_n is not None:
        raws = gen_n(
            _ADJ_SYSTEM, user, n=samples, temperature=0.7,
            thinking=config.thinking, max_new=max_new,
        )
    elif samples > 1:
        raws = [
            gen(_ADJ_SYSTEM, user, thinking=config.thinking, temperature=0.7, max_new=max_new)
            for _ in range(samples)
        ]
    else:
        raws = [gen(_ADJ_SYSTEM, user, thinking=config.thinking, temperature=None, max_new=max_new)]
    votes: Counter[str] = Counter()
    for raw in raws:
        label = _parse(raw)
        if label:
            votes[label] += 1
    if votes:
        return votes.most_common(1)[0][0]
    # Fallback 1: a single greedy single-pass prediction (no precedent/VKS/thinking
    # context) parsed the same way.
    try:
        raw = gen(_ADJ_SYSTEM, baseline_user(case), thinking=False, temperature=None, max_new=512)
        label = _parse(raw)
        if label:
            return label
    except Exception:  # never let a fallback crash the run
        pass
    # Fallback 2: majority class.
    return "PARTIAL_A_WIN"


def _parse(raw: str) -> str | None:
    """Parse the prediction JSON schema; fall back to a bare label token."""
    try:
        return parse_prediction(raw)[0].value
    except (ValueError, KeyError):
        return parse_label(raw)


# Decision-first evidence ordering: chunks containing a ruling lead come first.
_DECISION_KW = re.compile(
    r"tuyên xử|chấp nhận|không chấp nhận|\bbác\b|buộc.{0,40}(trả|bồi thường)|quyết định",
    re.IGNORECASE,
)
_QT_PRIORITY = {
    "vks_opinion": 0, "vks_accept": 0, "vks_partial": 0, "rescue_decision": 1,
    "court_decision": 2, "accepted_claim": 3, "rejected_claim": 3,
    "court_reasoning": 4, "rescue_reasoning": 4, "applied_law": 5,
}


def evidence_text(case_evidence, max_chars: int = EV_CHARS) -> str:
    """Chunks whose text contains a ruling lead, then by query-type, then score."""
    def key(item):
        has_ruling = 0 if _DECISION_KW.search(getattr(item, "text", "")) else 1
        return (has_ruling, _QT_PRIORITY.get(getattr(item, "query_type", ""), 9),
                -getattr(item, "score", 0.0))
    ordered = sorted(case_evidence, key=key)
    joined = "\n\n".join(
        f"[{getattr(e, 'query_type', '')}] {getattr(e, 'text', '')}" for e in ordered
    )
    return joined[:max_chars]


class EnsembleOutcomePredictor:
    """The self-consistency + precedent + thinking ensemble, with a safe fallback.

    ``.predict`` matches ``OutcomePredictor.predict`` so it drops into the pipeline
    unchanged. Any failure degrades to the single-pass fallback predictor.
    """

    def __init__(self, backend, config: OutcomeConfig, precedent_bank=None, fallback=None):
        self.backend = backend
        self.config = config
        self.precedent_bank = precedent_bank
        self.fallback = fallback

    def predict(self, case, case_evidence, law_evidence):
        ecase = {
            "case_id": case.case_id,
            "query": case.case_query,
            "ev": evidence_text(case_evidence),
        }

        def gen(system, user, *, thinking=False, temperature=None, max_new=512):
            return self.backend.generate(
                system, user, temperature=temperature, thinking=thinking, max_new_tokens=max_new
            )

        def gen_n(system, user, *, n, temperature=0.7, thinking=False, max_new=512):
            return self.backend.generate_samples(
                system, user, n=n, temperature=temperature, thinking=thinking, max_new_tokens=max_new
            )

        try:
            label = predict_outcome(ecase, gen, self.config, self.precedent_bank, gen_n)
            return OutcomeLabel(label), "ensemble (SC+precedent+thinking)", ""
        except Exception:
            if self.fallback is not None:
                return self.fallback.predict(case, case_evidence, law_evidence)
            raise


def evaluate(
    cases: list[dict],
    gen: Generate,
    config: OutcomeConfig,
    precedent_bank: PrecedentBank | None = None,
    gen_n: Callable[..., list[str]] | None = None,
) -> dict:
    """Return accuracy and a confusion matrix over ``cases`` (each has a ``gold``)."""
    confusion: Counter[tuple[str, str]] = Counter()
    correct = 0
    for case in cases:
        prediction = predict_outcome(case, gen, config, precedent_bank, gen_n)
        confusion[(case["gold"], prediction)] += 1
        correct += prediction == case["gold"]
    return {
        "accuracy": correct / len(cases) if cases else 0.0,
        "confusion": dict(confusion),
    }
