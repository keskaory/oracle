import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import cache  # noqa: E402
import readme_tables  # noqa: E402


def test_readme_numbers_match_the_result_files():
    # every number in the readme is generated. if this fails, run `python readme_tables.py` from src/
    text = readme_tables.README.read_text(encoding="utf-8")
    assert readme_tables.render(text) == text


def test_render_refuses_a_readme_missing_a_block():
    with pytest.raises(ValueError):
        readme_tables.render("<!-- auto:headline -->\n<!-- /auto -->\n")


def test_offline_mode_refuses_a_cache_miss(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cache, "OFFLINE", True)
    with pytest.raises(RuntimeError, match="offline"):
        cache.get_json("https://example.test/never-cached", {"a": 1})
