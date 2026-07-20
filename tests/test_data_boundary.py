import hashlib
import json
from dataclasses import fields

import pytest

from alqac2026.data import (
    load_inference_cases,
    load_law_corpus,
    load_public_gold,
    validate_private_input,
)
from alqac2026.schemas import InferenceCase


def test_inference_schema_contains_only_private_fields():
    assert {field.name for field in fields(InferenceCase)} == {"case_id", "case_query"}


def test_public_loader_separates_inference_and_gold():
    path = "data/raw/ALQAC2026_public_test.json"
    corpus = load_law_corpus("data/raw/corpus_law_pub.json")
    cases = load_inference_cases(path)
    gold = load_public_gold(path, corpus)
    assert len(cases) == 50
    assert len(gold) == 50
    assert all(not hasattr(case, "verdict_label") for case in cases)
    assert {case.case_id for case in cases} == set(gold)


def test_official_corpus_shape():
    corpus = load_law_corpus("data/raw/corpus_law_pub.json")
    assert len(corpus) == 3352
    assert len({(article.law_id, article.aid) for article in corpus}) == 3352


def test_private_contract_requires_exact_fields_count_and_sha(tmp_path):
    path = tmp_path / "private.json"
    payload = [
        {"case_id": f"case_{index}", "case_query": "query"}
        for index in range(2)
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(
        validate_private_input(path, expected_sha256=digest, expected_cases=2)
    ) == 2

    payload[0]["verdict_label"] = "A_WIN"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly case_id and case_query"):
        validate_private_input(path, expected_sha256=digest, expected_cases=2)
