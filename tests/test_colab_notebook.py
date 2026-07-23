import ast
import json
import re
from pathlib import Path


NOTEBOOKS = {
    "public": Path("notebooks/colab_public.ipynb"),
    "private": Path("notebooks/colab_private.ipynb"),
}


def _notebook_text(track: str) -> str:
    return NOTEBOOKS[track].read_text(encoding="utf-8")


def _load_notebook_function(track: str, function_name: str):
    notebook = json.loads(NOTEBOOKS[track].read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
                namespace = {"re": re}
                exec(compile(module, str(NOTEBOOKS[track]), "exec"), namespace)
                return namespace[function_name]
    raise AssertionError(f"Missing notebook function: {function_name}")


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
        assert "restore_hf_model_snapshots" in text
        assert "GIT_REPO_URL = normalize_git_repo_url(GIT_REPO_URL)" in text
        assert "Git clone failed for" in text
        assert "PIP_BASELINE_PATH = Path('/content/alqac-pip-check-baseline.json')" in text
        assert "new_issues = sorted(after_issues - set(pip_baseline['issues']))" in text
        assert "ALQAC dependency installation introduced new conflicts" in text
        assert "subprocess.check_call([sys.executable, '-m', 'pip', 'check'])" not in text


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
    assert "PROJECT_ROOT / 'data/raw/ALQAC_private_test.json'" in text
    assert "DRIVE_ROOT / 'inputs/private/ALQAC_private_test.json'" not in text
    assert text.index("['git', 'clone'") < text.index(
        "for required in (PRIVATE_INPUT, PRIVATE_CORPUS)"
    )
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


def test_colab_git_url_normalizer_accepts_raw_and_markdown_urls():
    raw = "https://github.com/NGBao1608/DL_K23_ALQAC2026.git"
    markdown = f"[{raw}]({raw})"
    for track in NOTEBOOKS:
        normalize = _load_notebook_function(track, "normalize_git_repo_url")
        assert normalize(raw) == raw
        assert normalize(markdown) == raw


def test_colab_git_url_normalizer_rejects_non_github_urls():
    for track in NOTEBOOKS:
        normalize = _load_notebook_function(track, "normalize_git_repo_url")
        try:
            normalize("https://example.com/repository.git")
        except ValueError as error:
            assert "raw https://github.com" in str(error)
        else:
            raise AssertionError("Non-GitHub URL was accepted")
