from alqac2026.citations import extract_law_citations
from alqac2026.data import load_law_corpus


def _articles():
    return load_law_corpus("data/raw/corpus_law_pub.json")


def test_extracts_multi_article_citation_bound_to_following_law():
    articles = _articles()
    text = "Áp dụng các Điều 584, 558, 590 và 603 của Bộ luật Dân sự năm 2015."
    result = extract_law_citations([text], articles)
    pairs = {(item.law_id, item.aid) for item in result}
    lookup = {(a.law_id, a.article_number): a.aid for a in articles}
    expected = {("91/2015/QH13", lookup[("91/2015/QH13", n)]) for n in (584, 558, 590, 603)}
    assert expected <= pairs
    assert all(item.law_id == "91/2015/QH13" for item in result)


def test_disambiguates_civil_procedure_from_civil_code():
    articles = _articles()
    text = "Điều 157 của Bộ luật Tố tụng dân sự và Điều 584 của Bộ luật Dân sự."
    result = extract_law_citations([text], articles)
    by_num = {(a.law_id, a.article_number): a.aid for a in articles}
    pairs = {(item.law_id, item.aid) for item in result}
    assert ("92/2015/QH13", by_num[("92/2015/QH13", 157)]) in pairs
    assert ("91/2015/QH13", by_num[("91/2015/QH13", 584)]) in pairs


def test_no_citation_text_returns_empty():
    assert extract_law_citations(["Không có trích dẫn điều luật nào."], _articles()) == []
    assert extract_law_citations([""], _articles()) == []


def test_deduplicates_repeated_citations():
    articles = _articles()
    text = "Điều 584 Bộ luật Dân sự ... lại viện dẫn Điều 584 của Bộ luật Dân sự."
    result = extract_law_citations([text], articles)
    keys = [(item.law_id, item.aid) for item in result]
    assert len(keys) == len(set(keys))
