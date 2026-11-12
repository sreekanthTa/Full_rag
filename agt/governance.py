"""AGT governance facade: composes the pillars into one entry point.

The app talks to AGT through a single object:

    gov = get_agt_governance()
    allowed, reason = gov.evaluate_tool_call("chroma_query", {"query": "..."})

Each call is checked in order - RBAC role permission, then the declarative
policy engine, then the per-tool rate budget - and every decision is written
to the tamper-evident audit log.
"""
from __future__ import annotations

import threading
from pathlib import Path

import yaml

from agt.pillars import audit, policy, rbac
from config import settings

_BUDGETS_PATH = Path(__file__).resolve().parent / "budgets.yaml"


def _load_budgets() -> dict:
    """Build one token bucket per tool from budgets.yaml (per_minute)."""
    try:
        from agent_os.policies.rate_limiting import RateLimitConfig, TokenBucket

        data = yaml.safe_load(_BUDGETS_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 - budgets are optional
        print(f"[agt.governance] budgets unavailable: {exc}")
        return {}

    buckets: dict = {}
    for tool, cfg in (data.get("budgets") or {}).items():
        per_minute = float((cfg or {}).get("per_minute", 0) or 0)
        if per_minute <= 0:
            continue
        buckets[tool] = TokenBucket.from_config(
            RateLimitConfig(capacity=per_minute, refill_rate=per_minute / 60.0)
        )
    return buckets


class AGTGovernance:
    """One entry point over the AGT pillars (rbac, policy, budgets, audit)."""

    def __init__(self) -> None:
        self._evaluator = policy._build()
        self._rbac = rbac._build()
        self._buckets = _load_budgets()
        self.policies = list(self._evaluator.policies) if self._evaluator is not None else []

    def evaluate_tool_call(self, tool_name: str, args: dict | None = None) -> tuple[bool, str]:
        """Check one tool call; returns (allowed, reason)."""
        args = args or {}

        allowed, reason = rbac.check(settings.agt_agent_id, tool_name)
        if allowed:
            allowed, reason = policy.evaluate(tool_name, args)
        if allowed and not self._consume(tool_name):
            allowed, reason = False, f"rate budget exhausted for '{tool_name}'"

        audit.record(tool_name, args, allowed, reason)
        return allowed, reason

    def _consume(self, tool_name: str) -> bool:
        bucket = self._buckets.get(tool_name)
        return bucket.consume() if bucket is not None else True


_engine = None
_ready = False
_lock = threading.Lock()


def get_agt_engine() -> AGTGovernance | None:
    """The shared AGT governance engine, or None if disabled/unavailable."""
    global _engine, _ready
    if _ready:
        return _engine
    with _lock:
        if _ready:
            return _engine
        _ready = True
        if not settings.agt_enabled:
            return None
        try:
            _engine = AGTGovernance()
        except Exception as exc:  # noqa: BLE001 - AGT must never break the app
            print(f"[agt.governance] init failed: {exc}")
            _engine = None
    return _engine


def get_agt_governance() -> AGTGovernance | None:
    """Alias used by rag_engine."""
    return get_agt_engine()


def verify_audit_integrity() -> dict:
    """Verify the tamper-evident audit chain."""
    return audit.verify()
