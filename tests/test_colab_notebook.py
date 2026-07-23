import ast
import json
from pathlib import Path


NOTEBOOKS = {
    "public": Path("notebooks/colab_public.ipynb"),
    "private": Path("notebooks/colab_private.ipynb"),
}


def _notebook_text(track: str) -> str:
    return NOTEBOOKS[track].read_text(encoding="utf-8")


def test_colab_notebooks_are_clean_and_code_cells_compile():
    for path in NOTEBOOKS.values():
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        assert code_cells
        for cell in code_cells:
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse("".join(cell["source"]))


def test_colab_notebooks_expose_only_smoke_and_full_stages():
    for text in map(_notebook_text, NOTEBOOKS):
        assert "STAGE = 'smoke'" in text
        assert "assert STAGE in {'smoke', 'full'}" in text
        assert "RUN_MODE" not in text
        assert "STAGE = 'runtime_check'" not in text
        assert "STAGE = 'resume'" not in text
        assert "source_pin.json" in text
        assert "smoke first" in text.lower()
        assert "['git', 'checkout', '--detach', pinned_commit]" in text
        assert "('gradio', 'gradio-client', 'hf-gradio')" in text
        assert "module_name.startswith('alqac2026.')" in text
        assert "Stale package import outside pinned checkout" in text


def test_public_notebook_runs_live_pipeline_and_evaluator():
    text = _notebook_text("public")
    assert "run_public_stage" in text
    assert "APPROVE_PUBLIC_API_CALLS = False" in text
    assert "ALQAC_TEAM_TOKEN" in text
    assert "metrics.json" in text
    assert "errors.json" in text
    assert "selection_profile.json" in text
    assert "outcome_accuracy" in text
    assert "law_micro_f1" in text
    assert "private_test_60_cases_extracted_corpus.json" not in text


def test_private_notebook_uses_private_inputs_without_evaluator():
    text = _notebook_text("private")
    assert "run_private_stage" in text
    assert "ALQAC_private_test.json" in text
    assert "private_test_60_cases_extracted_corpus.json" in text
    assert "full/selection_profile.json" in text
    assert "ALQAC_TEAM_TOKEN" in text
    assert "metrics.json" not in text
    assert "errors.json" not in text
    assert "outcome_accuracy" not in text
    assert "'upload': 'manual only'" in text


def test_runtime_check_has_no_case_api_or_team_token_dependency():
    text = Path("scripts/check_runtime.py").read_text(encoding="utf-8")
    assert "case_retrieval" not in text
    assert "ALQAC_TEAM_TOKEN" not in text
    assert '"api_network_attempts": 0' in text
    assert 'report["stages"]["embedding"]' in text
    assert 'report["stages"]["reranker"]' in text
    assert text.index('report["stages"]["embedding"]') < text.index(
        'report["stages"]["reranker"]'
    ) < text.index('report["stages"]["generation"]')
