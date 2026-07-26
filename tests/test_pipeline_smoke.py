from alqac2026.law_retrieval import BM25LawRetriever
from alqac2026.pipeline import ALQACPipeline, PreparedCase, PreparedCaseStore
from alqac2026.prediction import OutcomePredictor
from alqac2026.schemas import InferenceCase, OutcomeLabel


class EmptyCaseRetriever:
    def retrieve(self, case):
        return [], 0


class FixedBackend:
    def generate(self, system_prompt, user_prompt):
        return '{"reasoning":"Smoke test.","label":"B_WIN"}'


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
    # No case evidence -> no citations, so law_evidence is exactly the procedural priors.
    from alqac2026.citations import PROCEDURAL_PRIORS

    assert len(result.law_evidence) == len(PROCEDURAL_PRIORS)
    assert all(law.law_id == "92/2015/QH13" for law in result.law_evidence)


def test_prepared_context_checkpoint_round_trip(tmp_path):
    case = InferenceCase("case_mock", "tranh chấp hợp đồng")
    prepared = PreparedCase(case, [], [], 0)
    store = PreparedCaseStore(tmp_path / "contexts.json")
    store.put(prepared)
    assert PreparedCaseStore(tmp_path / "contexts.json").get(case) == prepared
