import ast
import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/colab_rag.ipynb")


def test_colab_notebook_is_clean_and_code_cells_compile():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert code_cells
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        ast.parse("".join(cell["source"]))


def test_colab_notebook_exposes_required_safe_run_controls():
    text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    for name in (
        "TRACK",
        "RUN_MODE",
        "EXPERIMENT",
        "EXECUTION_MODE",
        "RUN_ID",
        "SOURCE_MODE",
        "ADAPTER_PATH",
    ):
        assert name in text
    assert "cache-only" in text
    assert "Check format" in text
    assert "alqac_x" not in text.lower()
    assert "drive/MyDrive/ALQAC2026" in text
    assert "SOURCE_MODE = 'git'" in text
    assert "GIT_REF = 'TuanAnh'" in text
    assert "'git', 'clone', '--branch', GIT_REF, '--single-branch'" in text
    assert "source_pin.json" in text
    assert "['git', 'checkout', '--detach', pinned_commit]" in text
    assert "Run runtime_check first with this RUN_ID" in text
    assert "runtime_check_status') != 'PASS'" in text
    assert "Full/resume requires a completed smoke run" in text
    assert "smoke_manifest.get('git_commit') != pinned_commit" in text
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    secret_preflight = "".join(notebook["cells"][2]["source"])
    assert "required_secrets = ('GITHUB_TOKEN',)" in secret_preflight
    assert "secret_or_none('HF_TOKEN')" not in secret_preflight
    assert "secret_or_none('ALQAC_TEAM_TOKEN')" not in secret_preflight
    assert "if hf_token:" in text
    assert "team_token = secret_or_none('ALQAC_TEAM_TOKEN')" in text
    assert "alqac-pip-check-baseline.json" in text
    assert "('gradio', 'gradio-client')" in text
    assert "'pip', 'uninstall', '-q', '-y', *unused_colab_packages" in text
    assert text.index("'pip', 'uninstall'") < text.index(
        "'pip', 'install', '-q', '-r', 'requirements-colab.txt'"
    )
    assert "introduced new conflicts" in text
    assert "torch_version_after != pip_baseline['torch_version']" in text


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
