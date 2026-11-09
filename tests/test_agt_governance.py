"""Tests for the AGT governance integration.

Run with: .venv\\Scripts\\python.exe -m unittest tests.test_agt_governance -v
"""
import os
import tempfile
import unittest
from pathlib import Path

# Keep the audit DB out of the repo before config is imported.
_TEST_AUDIT_DB = Path(tempfile.gettempdir()) / "agt_governance_test.db"
for suffix in ("", "-wal", "-shm"):
    Path(str(_TEST_AUDIT_DB) + suffix).unlink(missing_ok=True)
os.environ["AGT_AUDIT_DB"] = str(_TEST_AUDIT_DB)

from agt.governance import AGTGovernance, verify_audit_integrity  # noqa: E402


class TestAGTGovernance(unittest.TestCase):
    def setUp(self) -> None:
        self.gov = AGTGovernance()

    def test_engine_loaded(self) -> None:
        self.assertIsNotNone(self.gov._evaluator, "policy evaluator failed to load")

    def test_allows_known_tool(self) -> None:
        allowed, reason = self.gov.evaluate_tool_call("chroma_query", {"query": "hello"})
        self.assertTrue(allowed, reason)

    def test_denies_listed_destructive_tool(self) -> None:
        allowed, reason = self.gov.evaluate_tool_call("delete_collection", {})
        self.assertFalse(allowed)
        self.assertTrue(reason)

    def test_denies_unlisted_tool(self) -> None:
        allowed, _ = self.gov.evaluate_tool_call("some_other_tool", {})
        self.assertFalse(allowed)

    def test_denies_dangerous_content(self) -> None:
        allowed, _ = self.gov.evaluate_tool_call(
            "chroma_query", {"query": "'; DROP TABLE users; --"}
        )
        self.assertFalse(allowed)

    def test_audit_chain_is_valid(self) -> None:
        self.gov.evaluate_tool_call("chroma_query", {"query": "x"})
        self.gov.evaluate_tool_call("delete_collection", {})
        result = verify_audit_integrity()
        self.assertTrue(result["valid"], result)
        self.assertGreaterEqual(result["total_entries"], 2)


if __name__ == "__main__":
    unittest.main()
