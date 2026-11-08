"""Microsoft Agent Governance Toolkit (AGT) wrapper for the Full-RAG assistant.

Provides tool-level governance using the Agent Control Specification (ACS)
runtime. Each tool call in the RAG pipeline (Chroma, LLM, Mem0, PDF ingest)
is evaluated against policies before execution.

If AGT is unavailable or misconfigured, calls pass through unguarded so the
RAG pipeline keeps working (fail-open).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from config import settings

_POLICY_DIR = Path(settings.agt_policy_dir)
_MANIFEST_PATH = _POLICY_DIR / "manifest.yaml"

_engine = None
_engine_ready = False


def _init_engine() -> None:
    """Lazy-initialize the AGT policy engine.

    Tries the ACS runtime first (newer versions), then falls back to the
    agent_os PolicyEngine (v4.x). Fail-open on any error.
    """
    global _engine, _engine_ready
    if _engine_ready:
        return

    try:
        # Try the newer ACS API first
        from agent_control_specification import AgentControl, HostSession  # type: ignore

        runtime = AgentControl.from_path(str(_MANIFEST_PATH))
        _engine = ("acs", runtime, HostSession)
        _engine_ready = True
        return
    except Exception:
        pass

    try:
        # Fall back to agent_os PolicyEngine (v4.x)
        from agent_os import PolicyEngine  # type: ignore

        engine = PolicyEngine()
        _engine = ("policy_engine", engine, None)
        _engine_ready = True
        return
    except Exception as exc:  # noqa: BLE001 - AGT must never break the app
        print(f"[agt] initialization failed: {exc}")
        _engine = None
        _engine_ready = True


def get_agt_engine() -> Optional[tuple]:
    """Return the initialized AGT engine tuple, or None if unavailable."""
    _init_engine()
    return _engine


class AGTGovernance:
    """Convenience wrapper around AGT for tool-level governance."""

    def __init__(self) -> None:
        self._engine = get_agt_engine()
        self._session = None
        if self._engine is not None:
            kind, engine, HostSessionCls = self._engine
            if kind == "acs" and HostSessionCls is not None:
                try:
                    self._session = HostSessionCls(
                        engine,
                        agent_id=settings.agt_agent_id,
                        session_id=settings.agt_session_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[agt] session creation failed: {exc}")

    # ------------------------------------------------------------------
    # Pre-tool-call evaluation
    # ------------------------------------------------------------------

    def evaluate_tool_call(
        self,
        tool_name: str,
        args: dict,
    ) -> tuple[bool, str]:
        """Evaluate a tool call against AGT policies before execution.

        Returns (is_allowed, reason). If AGT is unavailable, allowed by default.
        """
        if self._engine is None:
            return True, ""

        kind, engine, _ = self._engine

        try:
            if kind == "acs" and self._session is not None:
                evaluation = self._session.pre_tool_call(
                    tool_name=tool_name,
                    args=args,
                )
                verdict = getattr(evaluation, "verdict", evaluation)
                decision = getattr(verdict, "decision", None)
                if decision is not None:
                    permits = getattr(decision, "permits", True)
                else:
                    permits = getattr(verdict, "permits", True)
                reason = getattr(verdict, "reason", "") or ""
                if not permits:
                    return False, reason
                return True, ""

            elif kind == "policy_engine":
                # agent_os.PolicyEngine.validate_request
                result = engine.validate_request(
                    agent_id=settings.agt_agent_id,
                    tool=tool_name,
                    args=args,
                )
                if isinstance(result, bool):
                    if not result:
                        return False, "PolicyEngine denied request"
                    return True, ""
                permitted = getattr(result, "permitted", getattr(result, "allowed", True))
                reason = getattr(result, "reason", "") or ""
                if not permitted:
                    return False, reason
                return True, ""

        except Exception as exc:  # noqa: BLE001
            print(f"[agt] tool call evaluation failed: {exc}")
            return True, ""

        return True, ""

    # ------------------------------------------------------------------
    # Guarded execution helper
    # ------------------------------------------------------------------

    def guarded_call(
        self,
        tool_name: str,
        fn: Callable,
        *args,
        **kwargs,
    ):
        """Execute `fn` only if AGT policy permits the tool call.

        Returns the result of `fn`, or raises PermissionError if denied.
        """
        allowed, reason = self.evaluate_tool_call(tool_name, {"args": str(args), "kwargs": str(kwargs)})
        if not allowed:
            raise PermissionError(f"AGT policy denied tool call '{tool_name}': {reason}")
        return fn(*args, **kwargs)


def get_agt_governance() -> Optional[AGTGovernance]:
    """Return an initialized AGTGovernance instance, or None if disabled."""
    if not settings.agt_enabled:
        return None
    return AGTGovernance()