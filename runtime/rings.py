"""Runtime pillar 1 - Execution Rings: what our trust score lets us touch."""
from __future__ import annotations
import logging
import threading

logger = logging.getLogger(__name__)

_enforcer = None
_ready = False
_lock = threading.Lock()

_FLAGS = {"network": "network", "filesystem": "filesystem_writable", "subprocess": "subprocess"}


def _build():
    global _enforcer, _ready
    if _ready:
        return _enforcer
    with _lock:
        if _ready:
            return _enforcer
        _ready = True
        try:
            from agent_runtime import RingEnforcer
            _enforcer = RingEnforcer()
        except Exception as exc:
            logger.warning("rings init failed: %s", exc)
            _enforcer = None
    return _enforcer


def ring_for(score: int) -> str:
    """Which ring a given 0-1000 score earns (pure helper, for tests/insight)."""
    _build()
    if _enforcer is None:
        return "RING_3_SANDBOX"
    return _enforcer.compute_ring(score / 1000).name


def ring() -> str:
    """Our current execution ring, derived from the mesh trust score."""
    _build()
    if _enforcer is None:
        return "RING_3_SANDBOX"
    from mesh import trust
    return _enforcer.compute_ring(trust.score() / 1000).name


def constraints() -> dict:
    """The resource limits enforced for our current ring."""
    _build()
    if _enforcer is None:
        return {}
    from agent_runtime import ExecutionRing
    c = _enforcer.get_constraints(ExecutionRing[ring()])
    return {
        "network": c.network_allowed,
        "filesystem_writable": c.filesystem_writable,
        "filesystem_scope": c.filesystem_scope,
        "subprocess": c.subprocess_allowed,
        "max_concurrent_tools": c.max_concurrent_tools,
    }



def allows(kind: str) -> bool:
    """True if our ring permits 'network' | 'filesystem' | 'subprocess'."""
    return bool(constraints().get(_FLAGS.get(kind, kind), False))
