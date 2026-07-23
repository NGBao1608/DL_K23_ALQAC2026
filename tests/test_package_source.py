from runpy import run_path


selected_files = run_path("scripts/package_source.py")["selected_files"]


def test_source_bundle_selection_excludes_generated_and_runtime_files(tmp_path):
    (tmp_path / "src/package.egg-info").mkdir(parents=True)
    (tmp_path / "src/package.egg-info/PKG-INFO").write_text("generated")
    (tmp_path / "src/package").mkdir()
    (tmp_path / "src/package/module.py").write_text("value = 1")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache/case_api.sqlite").write_text("runtime")
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/ALQAC_private_test.json").write_text("private")
    (
        tmp_path / "data/raw/private_test_60_cases_extracted_corpus.json"
    ).write_text("private corpus")

    selected = {relative.as_posix() for _, relative in selected_files(tmp_path)}

    assert "src/package/module.py" in selected
    assert "src/package.egg-info/PKG-INFO" not in selected
    assert "cache/case_api.sqlite" not in selected
    assert "data/raw/ALQAC_private_test.json" not in selected
    assert "data/raw/private_test_60_cases_extracted_corpus.json" not in selected
