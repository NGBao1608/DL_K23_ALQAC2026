"""
Data utilities for ALQAC 2026.

Key facts about the data (verified on the public test + corpus):

corpus_law_pub.json : list of 18 law documents.
    each item = {"id": int, "law_id": "91/2015/QH13", "content": [ {"aid": int, "content_Article": str}, ... ]}
    -> The article NUMBER ("Điều N") is NOT stored. Articles are in order,
       so the k-th article in a law's `content` list == "Điều k" of that law.
       (Verified: 584th article of 91/2015/QH13 == grounds for tort liability;
        603rd == liability for damage caused by animals.)

ALQAC2026_public_test.json : list of 50 cases with FULL annotations (dev set).
    Important fields for us:
      case_id, case_query            -> the ONLY two fields present in the PRIVATE test
      verdict_label                  -> gold answer, 4 classes (see LABELS)
      related_law_provisions         -> gold law links, e.g. "Bộ luật Dân sự năm 2015 | Điều 584"
      case_fact / judgment_text / court_reasoning / court_verdict
                                     -> rich text, ONLY in public test. DO NOT depend on these
                                        at inference time; the private test won't have them.

The 4 official outcome labels:
    A_WIN, PARTIAL_A_WIN, B_WIN, PARTIAL_B_WIN
    (A = Nguyên đơn / plaintiff, B = Bị đơn / defendant)
"""
from __future__ import annotations
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field

LABELS = ["A_WIN", "PARTIAL_A_WIN", "B_WIN", "PARTIAL_B_WIN"]

# law_id -> canonical Vietnamese law name.
# NOTE: a few entries are marked TODO — confirm against the corpus / official
# law list before locking. Used only to build gold retrieval links for eval.
LAW_ID_TO_NAME = {
    "91/2015/QH13": "Bộ luật Dân sự 2015",
    "92/2015/QH13": "Bộ luật Tố tụng Dân sự 2015",
    "100/2015/QH13": "Bộ luật Hình sự 2015",
    "45/2013/QH13": "Luật Đất đai 2013",
    "52/2014/QH13": "Luật Hôn nhân và Gia đình 2014",
    "47/2010/QH12": "Luật Các tổ chức tín dụng 2010",
    "66/2014/QH13": "Luật Kinh doanh bất động sản 2014",
    "60/2014/QH13": "Luật Hộ tịch 2014",
    "52/2010/QH12": "Luật Nuôi con nuôi 2010",
    "26/2008/QH12": "Luật Thi hành án dân sự 2008",
    "326/2016/UBTVQH14": "Nghị quyết 326/2016/UBTVQH14 (án phí, lệ phí Tòa án)",
    "93/2015/QH13": "Luật Tố tụng hành chính 2015",
    "50/2014/QH13": "Luật Xây dựng 2014",
    "24/2012/NĐ-CP": "Nghị định 24/2012/NĐ-CP (kinh doanh vàng)",
    "39/2009/QH12": "TODO_confirm_39/2009/QH12",
    "19/2011/NĐ-CP": "TODO_confirm_19/2011/NĐ-CP",
    "37/2015/NĐ-CP": "TODO_confirm_37/2015/NĐ-CP",
    "02/2011/QH13": "TODO_confirm_02/2011/QH13",
}


def _norm(s: str) -> str:
    """Lower + strip accents-insensitive-ish normalization for fuzzy law matching."""
    s = unicodedata.normalize("NFC", s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass
class Article:
    law_id: str          # e.g. "91/2015/QH13"
    law_name: str        # canonical name from LAW_ID_TO_NAME
    art_no: int          # 1-based article number ("Điều N")
    aid: int             # organizers' internal article id
    text: str

    @property
    def doc_id(self) -> str:
        # stable retrieval key
        return f"{self.law_id}|Điều {self.art_no}"


@dataclass
class Case:
    case_id: str
    case_query: str
    verdict_label: str | None = None          # None on private test
    gold_law_refs: list[str] = field(default_factory=list)  # canonical doc_ids if resolvable
    raw: dict = field(default_factory=dict)


def load_corpus(path: str) -> list[Article]:
    with open(path, encoding="utf-8") as f:
        docs = json.load(f)
    arts: list[Article] = []
    for d in docs:
        law_id = d["law_id"]
        name = LAW_ID_TO_NAME.get(law_id, law_id)
        for i, a in enumerate(d["content"], start=1):
            arts.append(Article(law_id, name, i, a["aid"], a["content_Article"]))
    return arts


# Build a fuzzy index name->law_id once for gold-link resolution.
_NAME_INDEX = {_norm(v): k for k, v in LAW_ID_TO_NAME.items()}


def _resolve_law_id(name_str: str) -> str | None:
    """Map a free-text law name from `related_law_provisions` to a law_id."""
    n = _norm(name_str)
    if n in _NAME_INDEX:
        return _NAME_INDEX[n]
    # loose contains-match (handles 'Bộ luật dân sự năm 2015' vs 'Bộ luật Dân sự 2015')
    n2 = n.replace("năm ", "")
    for key, lid in _NAME_INDEX.items():
        k2 = key.replace("năm ", "")
        if k2 and (k2 in n2 or n2 in k2):
            return lid
    return None


def parse_gold_links(related_law_provisions: str) -> list[str]:
    """Turn the gold provision string into canonical doc_ids: 'law_id|Điều N'."""
    out = []
    for line in (related_law_provisions or "").split("\n"):
        line = line.strip()
        if not line or "|" not in line:
            continue
        name_part, art_part = line.rsplit("|", 1)
        m = re.search(r"Điều\s+(\d+)", art_part)
        if not m:
            continue
        art_no = int(m.group(1))
        lid = _resolve_law_id(name_part)
        if lid:
            out.append(f"{lid}|Điều {art_no}")
    return out


def load_public_test(path: str) -> list[Case]:
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    cases = []
    for it in items:
        cases.append(
            Case(
                case_id=it["case_id"],
                case_query=it["case_query"],
                verdict_label=it.get("verdict_label"),
                gold_law_refs=parse_gold_links(it.get("related_law_provisions", "")),
                raw=it,
            )
        )
    return cases


def load_private_test(path: str) -> list[Case]:
    """Private test only has case_id + case_query."""
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return [Case(case_id=it["case_id"], case_query=it["case_query"], raw=it) for it in items]


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    arts = load_corpus(os.getenv("CORPUS_PATH", "data/raw/corpus_law_pub.json"))
    cases = load_public_test(os.getenv("PUBLIC_TEST_PATH", "data/raw/ALQAC2026_public_test.json"))
    print(f"Articles: {len(arts)} | Cases: {len(cases)}")
    from collections import Counter
    print("Labels:", Counter(c.verdict_label for c in cases))
    resolved = sum(len(c.gold_law_refs) for c in cases)
    print(f"Resolved gold law links: {resolved}")
    print("Example case:", cases[0].case_id, "->", cases[0].verdict_label)
    print("  gold:", cases[0].gold_law_refs[:5])
