from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from .schemas import InferenceCase, LawArticle, OutcomeLabel, PublicGold


LAW_NAME_TO_ID = {
    "bo luat dan su": "91/2015/QH13",
    "bo luat dan su 2015": "91/2015/QH13",
    "bo luat dan su nam 2015": "91/2015/QH13",
    "bo luat to tung dan su": "92/2015/QH13",
    "bo luat to tung dan su 2015": "92/2015/QH13",
    "bo luat to tung dan su nam 2015": "92/2015/QH13",
    "luat dat dai": "45/2013/QH13",
    "luat dat dai 2013": "45/2013/QH13",
    "luat dat dai nam 2013": "45/2013/QH13",
    "luat hon nhan va gia dinh": "52/2014/QH13",
    "luat hon nhan va gia dinh 2014": "52/2014/QH13",
    "luat hon nhan va gia dinh nam 2014": "52/2014/QH13",
    "luat cac to chuc tin dung": "47/2010/QH12",
    "luat cac to chuc tin dung 2010": "47/2010/QH12",
    "luat thi hanh an dan su": "26/2008/QH12",
    "luat xay dung nam 2014": "50/2014/QH13",
    "luat ho tich": "60/2014/QH13",
    "luat to tung hanh chinh": "93/2015/QH13",
    "nghi quyet 326/2016/ubtvqh14": "326/2016/UBTVQH14",
}


def _read_json(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array: {path}")
    return value


def _validate_unique_cases(cases: list[InferenceCase]) -> None:
    identifiers = [case.case_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate case_id found in input data")
    if any(not case.case_id.strip() or not case.case_query.strip() for case in cases):
        raise ValueError("case_id and case_query must be non-empty strings")


def load_inference_cases(path: str | Path) -> list[InferenceCase]:
    cases = []
    for item in _read_json(path):
        if "case_id" not in item or "case_query" not in item:
            raise ValueError("Every input item must contain case_id and case_query")
        cases.append(
            InferenceCase(
                case_id=str(item["case_id"]),
                case_query=str(item["case_query"]),
            )
        )
    _validate_unique_cases(cases)
    return cases


def load_law_corpus(path: str | Path) -> list[LawArticle]:
    articles: list[LawArticle] = []
    seen: set[tuple[str, int]] = set()
    for law in _read_json(path):
        law_id = str(law["law_id"])
        for article_number, item in enumerate(law["content"], start=1):
            key = (law_id, int(item["aid"]))
            if key in seen:
                raise ValueError(f"Duplicate law identifier: {key}")
            seen.add(key)
            articles.append(
                LawArticle(
                    law_id=law_id,
                    aid=int(item["aid"]),
                    article_number=article_number,
                    text=str(item["content_Article"]),
                )
            )
    return articles


def _normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFD", value.lower())
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d")
    value = re.sub(r"\bso\b", "", value)
    value = re.sub(r"\s+", " ", value).strip(" .,:")
    return value


def _resolve_law_id(name: str) -> str | None:
    normalized = _normalize_name(name)
    for marker, law_id in LAW_NAME_TO_ID.items():
        if normalized == marker or normalized.startswith(marker + " ngay"):
            return law_id
    return None


def _parse_public_law_gold(
    provisions: str, article_lookup: dict[tuple[str, int], LawArticle]
) -> tuple[tuple[str, int], ...]:
    result: list[tuple[str, int]] = []
    for line in (provisions or "").splitlines():
        if "|" not in line:
            continue
        name, article_text = line.rsplit("|", 1)
        match = re.search(r"Điều\s+(\d+)", article_text, flags=re.IGNORECASE)
        law_id = _resolve_law_id(name)
        if not match or not law_id:
            continue
        article = article_lookup.get((law_id, int(match.group(1))))
        if article:
            result.append((article.law_id, article.aid))
    return tuple(dict.fromkeys(result))


def load_public_gold(
    path: str | Path, articles: list[LawArticle]
) -> dict[str, PublicGold]:
    article_lookup = {
        (article.law_id, article.article_number): article for article in articles
    }
    gold: dict[str, PublicGold] = {}
    for item in _read_json(path):
        label = item.get("verdict_label")
        if label not in {member.value for member in OutcomeLabel}:
            raise ValueError(f"Missing or invalid public verdict_label: {item.get('case_id')}")
        case_ids = tuple(item.get("gold_case_evidence", ()))
        record = PublicGold(
            case_id=str(item["case_id"]),
            verdict_label=OutcomeLabel(label),
            law_evidence=_parse_public_law_gold(
                item.get("related_law_provisions", ""), article_lookup
            ),
            case_evidence=case_ids,
        )
        if record.case_id in gold:
            raise ValueError(f"Duplicate public gold case_id: {record.case_id}")
        gold[record.case_id] = record
    return gold

