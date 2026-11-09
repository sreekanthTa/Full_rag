"""Microsoft Agent Governance Toolkit (AGT) integration for the Full-RAG assistant.

Tool-level governance using AGT's declarative policy engine
(`agent_os.policies.PolicyEvaluator`). Policies live as YAML in
`agt/policies/` and are evaluated before every tool call in the RAG pipeline
(Chroma, LLM, Mem0, PDF ingest). Per-tool call budgets come from
`agt/budgets.yaml`, and every decision is written to a tamper-evident audit log.

Posture:
  - A policy DENY always blocks (unless shadow mode is on).
  - If the engine cannot be built, calls follow `settings.agt_fail_closed`.
  - A policy evaluation error follows `settings.agt_fail_closed`.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional

import yaml

from config import settings

_POLICY_DIR = Path(settings.agt_policy_dir)
_BUDGETS_PATH = _POLICY_DIR.parent / "budgets.yaml"

_evaluator = None
_rate_limiter = None
_audit = None
_ready = False
_lock = threading.Lock()


class _RateLimiter:
    """Per-tool token buckets built from budgets.yaml."""

    def __init__(self, budgets: dict) -> None:
        from agent_os.policies.rate_limiting import RateLimitConfig, TokenBucket

        self._buckets: dict = {}
        for tool, cfg in (budgets or {}).items():
            per_minute = float((cfg or {}).get("per_minute", 0) or 0)
            if per_minute > 0:
                config = RateLimitConfig(capacity=per_minute, refill_rate=per_minute / 60.0)
                self._buckets[tool] = TokenBucket.from_config(config)

    def allow(self, tool_name: str) -> bool:
        bucket = self._buckets.get(tool_name)
        if bucket is None:
            return True
        return bucket.consume(1)


def _load_budgets() -> dict:
    if not _BUDGETS_PATH.exists():
        return {}
    data = yaml.safe_load(_BUDGETS_PATH.read_text(encoding="utf-8")) or {}
    return data.get("budgets", {}) or {}


def _init_engine() -> None:
    """Build the policy evaluator, rate limiter, and audit log once."""
    global _evaluator, _rate_limiter, _audit, _ready
    if _ready:
        return
    with _lock:
        if _ready:
            return
        _ready = True
        try:
            from agent_os.policies import PolicyEvaluator

            evaluator = PolicyEvaluator()
            evaluator.load_policies(str(_POLICY_DIR))
            if not evaluator.policies:
                raise RuntimeError(f"no policy documents found in {_POLICY_DIR}")
            _evaluator = evaluator
            _rate_limiter = _RateLimiter(_load_budgets())
            if settings.agt_audit_enabled:
                from .audit import GovernanceAudit

                _audit = GovernanceAudit()
        except Exception as exc:  # noqa: BLE001 - AGT must never crash the app
            print(f"[agt] initialization failed: {exc}")
            _evaluator = None


def get_agt_engine():
    """Return the initialized policy evaluator, or None if unavailable."""
    _init_engine()
    return _evaluator


def _flatten(args: dict) -> str:
    """Flatten tool arguments into a single string for content rules."""
    return " ".join(str(v) for v in args.values() if v is not None)


class AGTGovernance:
    """Evaluates tool calls against the AGT declarative policies."""

    def __init__(self) -> None:
        _init_engine()
        self._evaluator = _evaluator
        self._rate_limiter = _rate_limiter
        self._audit = _audit

    def evaluate_tool_call(self, tool_name: str, args: dict) -> tuple[bool, str]:
        """Evaluate a tool call before execution.

        Returns (is_allowed, reason). If AGT is unavailable or an evaluation
        error occurs, the configured fail posture (`agt_fail_closed`) applies.
        """
        if self._evaluator is None:
            if settings.agt_fail_closed:
                return False, "AGT policy engine unavailable (fail-closed)"
            return True, ""

        allowed = True
        reason = ""

        if self._rate_limiter is not None and not self._rate_limiter.allow(tool_name):
            allowed, reason = False, f"rate limit exceeded for tool '{tool_name}'"
        else:
            context = {
                "tool_name": tool_name,
                "agent_id": settings.agt_agent_id,
                "session_id": settings.agt_session_id,
                "content": _flatten(args),
            }
            for key, value in args.items():
                if isinstance(key, str):
                    context.setdefault(key, value)
            try:
                decision = self._evaluator.evaluate(context)
                allowed = decision.allowed
                reason = "" if allowed else (decision.reason or "denied by policy")
            except Exception as exc:  # noqa: BLE001 - never crash the pipeline
                allowed = settings.agt_fail_closed
                reason = f"policy evaluation error: {exc}"

        if settings.agt_shadow_mode:
            allowed = True

        self._record(tool_name, args, allowed, reason)
        return allowed, reason

    def _record(self, tool_name: str, args: dict, allowed: bool, reason: str) -> None:
        if self._audit is None:
            return
        try:
            trace_id = self._audit.trace(tool_name, args)
            if allowed:
                self._audit.allow(trace_id)
            else:
                self._audit.deny(trace_id, reason)
        except Exception as exc:  # noqa: BLE001 - audit must never break the app
            print(f"[agt] audit write failed: {exc}")

    def guarded_call(self, tool_name: str, fn: Callable, *args, **kwargs):
        """Execute `fn` only if AGT policy permits the tool call."""
        allowed, reason = self.evaluate_tool_call(
            tool_name, {"args": str(args), "kwargs": str(kwargs)}
        )
        if not allowed:
            raise PermissionError(f"AGT policy denied tool call '{tool_name}': {reason}")
        return fn(*args, **kwargs)


def get_agt_governance() -> Optional[AGTGovernance]:
    """Return an initialized AGTGovernance instance, or None if disabled."""
    if not settings.agt_enabled:
        return None
    return AGTGovernance()


def verify_audit_integrity() -> dict:
    """Verify the tamper-evident audit chain."""
    _init_engine()
    if _audit is None:
        return {"valid": True, "total_entries": 0, "message": "audit disabled"}
    return _audit.verify()
