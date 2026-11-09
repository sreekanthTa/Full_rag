"""Tests for Pillar 1 - the policy engine.

Run with: .venv\\Scripts\\python.exe -m unittest tests.test_agt_policy -v
"""
import unittest

from agt.pillars import policy


class TestPolicyEngine(unittest.TestCase):
    def test_engine_loaded(self) -> None:
        self.assertIsNotNone(policy._build(), "policy evaluator failed to load")

    def test_allows_known_tool(self) -> None:
        allowed, reason = policy.evaluate("chroma_query", {"query": "hello"})
        self.assertTrue(allowed, reason)

    def test_denies_listed_destructive_tool(self) -> None:
        allowed, reason = policy.evaluate("delete_collection", {})
        self.assertFalse(allowed)
        self.assertTrue(reason)

    def test_denies_unlisted_tool(self) -> None:
        allowed, _ = policy.evaluate("some_other_tool", {})
        self.assertFalse(allowed)

    def test_denies_dangerous_content(self) -> None:
        allowed, _ = policy.evaluate(
            "chroma_query", {"query": "'; DROP TABLE users; --"}
        )
        self.assertFalse(allowed)


if __name__ == "__main__":
    unittest.main()
