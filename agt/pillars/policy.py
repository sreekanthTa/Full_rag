"""Pillar 1 - Policy Engine: declarative allow/deny for tool calls."""
from __future__ import annotations
import threading
from pathlib import Path
from agent_os.policies import PolicyEvaluator
from config import settings

_POLICY_DIR = Path(settings.agt_policy_dir)
_evaluator = None
_ready = False
_lock = threading.Lock()

def _build():
    global _evaluator, _ready
    if _ready:
        return _evaluator
    with _lock:
        if _ready:
            return _evaluator

        _ready = True

        try:
            ev = PolicyEvaluator()
            ev.load_policies(str(_POLICY_DIR))
            _evaluator = ev if ev.policies else None
        except Exception as exc:
            print(f"[agt.policy] init failed: {exc}")
            _evaluator = None
        return _evaluator


def evaluate(tool_name: str, args: dict) -> tuple[bool, str]:
    """Evaluate a tool call against the policy engine."""
    ev  = _build()

    if ev is None:
        return (not settings.agt_fail_closed), "No policy engine available"

    ctx = {"tool_name": tool_name, "agent_id": settings.agt_agent_id,
           "session_id": settings.agt_session_id,
           "content": " ".join(str(v) for v in args.values() if v is not None)}

    ctx.update({k: v for k, v in args.items() if v is not None})

    try:
        d = ev.evaluate(ctx)
    except Exception as exc:
        return False, f"Policy evaluation failed: {exc}"

    allowed = bool(d.allowed) or settings.agt_shadow_mode
    return allowed, "" if allowed else (d.reason or "denied by policy")