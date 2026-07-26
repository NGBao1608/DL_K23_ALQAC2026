"""Citation-aware law evidence.

Vietnamese first-instance judgments cite the applied provisions literally, e.g.
"áp dụng các Điều 584, 558, 590 và 603 của Bộ luật Dân sự năm 2015". Those spans
are recoverable from the case evidence returned by the Case Content API (the
"court_decision" / "applied_law" segments), so they are a valid inference input —
not a gold field. Extracting cited provisions this way is substantially more
precise than BM25 over the raw query.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from .data import LAW_NAME_TO_ID
from .schemas import LawArticle, LawEvidence


# Longest names first so "bộ luật tố tụng dân sự" wins over "bộ luật dân sự".
_LAW_NAMES = sorted(LAW_NAME_TO_ID, key=len, reverse=True)
# Match an article, plus any comma/"và"-separated numbers that follow it in the
# same enumeration, e.g. "Điều 584, 558, 590 và 603". Trailing tokens such as a
# year ("năm 2015") are excluded because they are not separator+number runs.
_ARTICLE_PATTERN = re.compile(r"dieu\s+(\d+(?:\s*(?:,|va)\s*\d+)*)")
_NUMBER_PATTERN = re.compile(r"\d+")
_CITATION_SCORE = 1.0


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d")
    return re.sub(r"\s+", " ", text)


# Civil-procedure articles the court applies in almost every first-instance civil
# case (jurisdiction, court fees, trial-in-absentia, appeal window). They are
# rarely present in the retrieved evidence text, so citation extraction misses
# them; adding them as a uniform prior recovers these near-universal provisions.
# Listed by (law_id, article_number); article_number is the list position,
# resolved to the official aid against the corpus.
PROCEDURAL_PRIORS: tuple[tuple[str, int], ...] = (
    ("92/2015/QH13", 26),   # thẩm quyền theo loại tranh chấp
    ("92/2015/QH13", 147),  # án phí sơ thẩm
    ("92/2015/QH13", 35),   # thẩm quyền Tòa án cấp huyện
    ("92/2015/QH13", 228),  # xét xử vắng mặt
    ("92/2015/QH13", 39),   # thẩm quyền theo lãnh thổ
    ("92/2015/QH13", 273),  # thời hạn kháng cáo
)
_PROCEDURAL_SCORE = 0.5


def procedural_prior_evidence(articles: list[LawArticle]) -> list[LawEvidence]:
    """Return LawEvidence for the ubiquitous civil-procedure articles present in the corpus."""
    article_by_number = {
        (article.law_id, article.article_number): article for article in articles
    }
    evidence: list[LawEvidence] = []
    for law_id, number in PROCEDURAL_PRIORS:
        article = article_by_number.get((law_id, number))
        if article is not None:
            evidence.append(
                LawEvidence(
                    law_id=article.law_id,
                    aid=article.aid,
                    text=article.text,
                    score=_PROCEDURAL_SCORE,
                    article_number=article.article_number,
                )
            )
    return evidence


def extract_law_citations(
    texts: Iterable[str], articles: list[LawArticle]
) -> list[LawEvidence]:
    """Return deduplicated ``LawEvidence`` for every ``Điều N of <law>`` citation.

    Each ``Điều N`` is bound to the nearest law name that follows it (the usual
    "Điều X, Y ... của <Luật>" order), falling back to the closest law name if
    none follows. ``N`` maps to the article at list position ``N`` of that law,
    whose ``aid`` is the official corpus identifier.
    """
    article_by_number = {
        (article.law_id, article.article_number): article for article in articles
    }
    evidence: list[LawEvidence] = []
    seen: set[tuple[str, int]] = set()
    for text in texts:
        normalized = _normalize(text or "")
        law_spans = sorted(
            (match.start(), LAW_NAME_TO_ID[name])
            for name in _LAW_NAMES
            for match in re.finditer(re.escape(name), normalized)
        )
        if not law_spans:
            continue
        for match in _ARTICLE_PATTERN.finditer(normalized):
            position = match.start()
            following = [span for span in law_spans if span[0] >= position]
            if following:
                law_id = following[0][1]
            else:
                law_id = min(law_spans, key=lambda span: abs(span[0] - position))[1]
            for token in _NUMBER_PATTERN.findall(match.group(1)):
                number = int(token)
                article = article_by_number.get((law_id, number))
                if article is None:
                    continue
                key = (article.law_id, article.aid)
                if key in seen:
                    continue
                seen.add(key)
                evidence.append(
                    LawEvidence(
                        law_id=article.law_id,
                        aid=article.aid,
                        text=article.text,
                        score=_CITATION_SCORE,
                        article_number=article.article_number,
                    )
                )
    return evidence
