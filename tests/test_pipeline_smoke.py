from alqac2026.law_retrieval import BM25LawRetriever
from alqac2026.pipeline import (
    ALQACPipeline,
    PreparedCase,
    PreparedCaseStore,
    build_law_query,
)
from alqac2026.prediction import OutcomePredictor
from alqac2026.schemas import CaseEvidence, InferenceCase, OutcomeLabel


class EmptyCaseRetriever:
    def retrieve(self, case):
        return [], 0


class FixedBackend:
    def generate(self, system_prompt, user_prompt):
        return (
            '{"main_claim":"x","accepted_scope":"không chấp nhận",'
            '"acceptance_ratio":0.0,"reasoning":"Smoke test.",'
            '"label":"B_WIN"}'
        )


def test_pipeline_runs_without_api_or_gpu():
    from alqac2026.data import load_law_corpus

    pipeline = ALQACPipeline(
        EmptyCaseRetriever(),
        BM25LawRetriever(load_law_corpus("data/raw/corpus_law_pub.json")),
        OutcomePredictor(FixedBackend()),
    )
    result = pipeline.predict_case(InferenceCase("case_mock", "tranh chấp hợp đồng"))
    assert result.status == "completed"
    assert result.prediction is OutcomeLabel.B_WIN
    assert len(result.law_evidence) == 5


def test_prepared_context_checkpoint_round_trip(tmp_path):
    case = InferenceCase("case_mock", "tranh chấp hợp đồng")
    prepared = PreparedCase(case, [], [], 0)
    store = PreparedCaseStore(tmp_path / "contexts.json")
    store.put(prepared)
    assert PreparedCaseStore(tmp_path / "contexts.json").get(case) == prepared


def test_law_query_selects_legal_sentences_not_arbitrary_prefix():
    query = build_law_query(
        InferenceCase("case_1", "Tranh chấp hợp đồng vay tài sản"),
        [
            CaseEvidence(
                "opaque",
                "Boilerplate without a legal signal. "
                "Tòa chấp nhận yêu cầu và áp dụng Điều 466 Bộ luật Dân sự.",
                1.0,
                "court_decision",
            )
        ],
    )
    assert "Tranh chấp hợp đồng vay tài sản" in query
    assert "Điều 466 Bộ luật Dân sự" in query
    assert "Boilerplate without a legal signal" not in query
