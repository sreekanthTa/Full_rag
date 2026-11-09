"""Runtime pillar 2 - Resource Limits: per-ring token-bucket rate limiting."""
from __future__ import annotations
import threading
from mesh import identity
from runtime import rings

_session = "main"
_limiter = None
_ready = False
_lock = threading.Lock()

def _build():
    global _limiter, _ready
    if _ready:
        return _limiter
    with _lock:
        if _ready:
            return _limiter
        _ready = True
        try:
            from agent_runtime import AgentRateLimiter
            _limiter = AgentRateLimiter()
        except Exception as exc:
            print(f"[runtime.limits] init failed: {exc}")
            _limiter = None
    return _limiter

def _agent() -> str:
    return identity.did() or "rag-assistant"

def allow(cost: float = 1.0) -> bool:
    """True if our agent may spend `cost` right now under its current ring."""
    lim = _build()
    if lim is None:
        return False  # fail-closed: no limiter, no work
    from agent_runtime import ExecutionRing
    ring = ExecutionRing[rings.ring()]
    agent = _agent()
    lim.update_ring(agent, _session, ring)
    return lim.try_check(agent, _session, ring, cost)

def stats() -> dict:
    """Current token-bucket state for this agent."""
    lim = _build()
    if lim is None:
        return {}
    st = lim.get_stats(_agent(), _session)
    if st is None:
        return {}
    return {
        "ring": st.ring.name,
        "capacity": st.capacity,
        "tokens_available": st.tokens_available,
        "total_requests": st.total_requests,
        "rejected_requests": st.rejected_requests,
    }