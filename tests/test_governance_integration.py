"""Tests for the Agent OS governance integration (governance.py + the app).

Everything here is hermetic: the model, vector store, and memory tiers are
stubbed, and the audit DB is written to a temp file. We only exercise the
governance wiring itself.

Run with: .venv\\Scripts\\python.exe -m unittest tests.test_governance_integration -v
"""
import os
import tempfile
import unittest
from pathlib import Path

# Keep the audit DB out of the repo before config is imported.
_TEST_AUDIT_DB = Path(tempfile.gettempdir()) / "gov_integration_test.db"
for suffix in ("", "-wal", "-shm"):
    Path(str(_TEST_AUDIT_DB) + suffix).unlink(missing_ok=True)
os.environ["AGT_AUDIT_DB"] = str(_TEST_AUDIT_DB)

import governance  # noqa: E402
import rag_engine  # noqa: E402

# --- make ask() hermetic: no Chroma, no Postgres, no Mem0, no guardrails ----
rag_engine.search = lambda q, top_k=4: ["chunk-a", "chunk-b"]
rag_engine.load_recent_messages = lambda sid: []
rag_engine.mem0_get = lambda sid: []
rag_engine.save_exchange = lambda *a, **k: None
rag_engine.mem0_add = lambda *a, **k: None
rag_engine.settings.guardrails_enabled = False


class _Resp:
    content = "test answer"


class _LLM:
    def invoke(self, msgs):
        return _Resp()


rag_engine.get_llm = lambda: _LLM()

from agt.pillars import signals  # noqa: E402
from sre import circuit, slo  # noqa: E402
from provenance import delta  # noqa: E402
import runtime.ternot as ternot  # noqa: E402


class TestGovernanceIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        governance.register_termination_hook()

    def setUp(self) -> None:
        governance.revive()
        signals.resume()
        circuit.reset()

    def tearDown(self) -> None:
        governance.revive()
        signals.resume()
        circuit.reset()

    # --- identity / ring ----------------------------------------------------

    def test_context_reports_identity_and_ring(self) -> None:
        ctx = governance.context()
        self.assertIsNotNone(ctx["did"])
        self.assertIsInstance(ctx["trust"], int)
        self.assertTrue(ctx["ring"].startswith("RING_"))
        self.assertIn("ring_limits", ctx)

    # --- admission control --------------------------------------------------

    def test_admit_allows_normally(self) -> None:
        allowed, reason = governance.admit()
        self.assertTrue(allowed, reason)

    def test_admit_refuses_when_paused(self) -> None:
        signals.pause()
        allowed, reason = governance.admit()
        self.assertFalse(allowed)
        self.assertIn("paused", reason)

    def test_kill_switch_lifecycle(self) -> None:
        # A kill is a hard stop, and revive restores service.
        ternot.kill("manual", "test")
        self.assertTrue(governance.terminated())
        self.assertFalse(governance.admit()[0])
        self.assertFalse(rag_engine.ask("again?", "t-session")["admitted"])

        governance.revive()
        self.assertTrue(governance.admit()[0])
        self.assertTrue(rag_engine.ask("back?", "t-session")["admitted"])

    # --- the governed ask() flow --------------------------------------------

    def test_ask_is_governed(self) -> None:
        out = rag_engine.ask("what is x?", "t-session")
        self.assertTrue(out["admitted"])
        self.assertEqual(out["answer"], "test answer")
        self.assertIn("agent", out)
        self.assertEqual(out["agent"]["ring"], governance.context()["ring"])
        self.assertIsNotNone(out["trace_id"])

    def test_ask_grows_provenance_and_seals(self) -> None:
        before = delta.count()
        rag_engine.ask("q1", "t-session")
        self.assertGreater(delta.count(), before)
        governance.seal()
        self.assertTrue(governance.verify_provenance())

    def test_ask_feeds_the_slo(self) -> None:
        rag_engine.ask("qa", "t-session")
        self.assertIn("compliance", slo.status())

    def test_replay_of_last_turn_is_clean(self) -> None:
        rag_engine.ask("replay me", "t-session")
        result = governance.replay_trace()
        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 2)

    # --- sre circuit + runtime sandbox --------------------------------------

    def test_generate_returns_fallback_when_circuit_open(self) -> None:
        for _ in range(5):
            circuit.record(False)
        self.assertEqual(circuit.state(), "OPEN")
        answer = governance.generate([object()])
        self.assertIn("temporarily unavailable", answer)

    def test_sandbox_session_runs(self) -> None:
        with governance.sandbox_session() as sid:
            self.assertIsInstance(sid, str)
            self.assertTrue(sid.startswith("session:"))

    # --- security / correctness fixes ---------------------------------------

    def test_fail_closed_when_engine_unavailable(self) -> None:
        original = rag_engine.get_agt_governance
        rag_engine.get_agt_governance = lambda: None
        previous = rag_engine.settings.agt_fail_closed
        try:
            rag_engine.settings.agt_fail_closed = True
            out = rag_engine.ask("blocked?", "t-session")
            self.assertTrue(out["agt_blocked"])
            self.assertIn("governance unavailable", out["answer"])

            rag_engine.settings.agt_fail_closed = False
            out = rag_engine.ask("allowed?", "t-session")
            self.assertFalse(out["agt_blocked"])
            self.assertEqual(out["answer"], "test answer")
        finally:
            rag_engine.settings.agt_fail_closed = previous
            rag_engine.get_agt_governance = original

    def test_agent_identity_is_consistent(self) -> None:
        import runtime.limit as limit
        import runtime.ternot as ternot

        # Every pillar now falls back to the same configured agent id.
        self.assertEqual(limit._agent(), ternot._agent())
        self.assertNotEqual(limit._agent(), "rag-assitant")


class TestGovernanceEndpoints(unittest.TestCase):
    ADMIN_KEY = "test-admin-key"

    @classmethod
    def setUpClass(cls) -> None:
        from fastapi.testclient import TestClient

        import main
        from config import settings

        settings.admin_api_key = cls.ADMIN_KEY
        settings.allow_insecure_dev = False

        cls.client = TestClient(main.app)

    def _admin(self) -> dict:
        return {"X-Admin-Key": self.ADMIN_KEY}

    def test_health_snapshot(self) -> None:
        body = self.client.get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("ring", body)
        self.assertIn("trust_score", body)
        self.assertIn("circuit", body)

    def test_slo_endpoint(self) -> None:
        body = self.client.get("/slo").json()
        self.assertIn("compliance", body)

    def test_admin_requires_key(self) -> None:
        # Control endpoints are fail-closed: no key -> 401, wrong key -> 401.
        self.assertEqual(self.client.post("/admin/kill").status_code, 401)
        self.assertEqual(
            self.client.post("/admin/kill", headers={"X-Admin-Key": "wrong"}).status_code,
            401,
        )

    def test_provenance_endpoints(self) -> None:
        self.assertEqual(self.client.get("/provenance/verify").status_code, 200)
        resp = self.client.post("/provenance/seal", headers=self._admin())
        self.assertIn("seal", resp.json())

    def test_replay_endpoint(self) -> None:
        self.assertEqual(self.client.get("/replay/last").status_code, 200)

    def test_delivery_endpoints(self) -> None:
        deploy = self.client.post(
            "/admin/deploy", params={"version": "t1"}, headers=self._admin()
        ).json()
        self.assertEqual(deploy["state"], "deploying")
        promote = self.client.post("/admin/promote", headers=self._admin()).json()
        self.assertEqual(promote["state"], "active")


if __name__ == "__main__":
    unittest.main()
