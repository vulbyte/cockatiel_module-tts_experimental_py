"""
Tests for model-source resolution (URL / repo-id / local-dir).

Run with the stdlib runner, no third-party deps:

    python3 -m unittest test_model_source -v
"""

import importlib.util
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("tts_service", "tts_service.py")
_M = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_M)


class ResolveModelSourceTest(unittest.TestCase):
    def assert_resolves(self, source, expected):
        self.assertEqual(_M.resolve_model_source(source), expected, source)

    def test_huggingface_url_becomes_repo_id(self):
        self.assert_resolves(
            "https://huggingface.co/facebook/mms-1b-all", "facebook/mms-1b-all"
        )

    def test_trailing_question_mark_is_stripped(self):
        # Operators paste URLs with a trailing '?' (e.g. from a browser).
        self.assert_resolves(
            "https://huggingface.co/facebook/mms-1b-all?", "facebook/mms-1b-all"
        )

    def test_query_string_is_stripped(self):
        self.assert_resolves(
            "https://huggingface.co/facebook/mms-1b-all?foo=bar",
            "facebook/mms-1b-all",
        )

    def test_tree_page_path_is_reduced(self):
        self.assert_resolves(
            "https://huggingface.co/facebook/mms-1b-all/tree/main",
            "facebook/mms-1b-all",
        )

    def test_repo_id_passes_through_unchanged(self):
        self.assert_resolves("facebook/mms-tts-eng", "facebook/mms-tts-eng")

    def test_local_dir_passes_through_unchanged(self):
        self.assert_resolves("./local_model_dir", "./local_model_dir")
        self.assert_resolves("/abs/path/to/model", "/abs/path/to/model")

    def test_empty_source_is_empty(self):
        self.assert_resolves("", "")
        self.assert_resolves(None, "")


if __name__ == "__main__":
    unittest.main()