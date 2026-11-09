"""SRE pillar 4 - Circuit Breaker: stop hammering a failing dependency."""
from __future__ import annotations
import threading

from mesh import identity

_FAILURE_THRESHOLD = 5

_breaker = None
_ready = False
_lock = threading.Lock()

def _build():
    global _breaker, _ready
    if _ready:
        return _breaker
    with _lock:
        if _ready:
            return _breaker
        _ready = True
        try:
            from agent_sre.cascade import CircuitBreaker, CircuitBreakerConfig
            _breaker = CircuitBreaker(
                identity.did() or "rag-assistant",
                CircuitBreakerConfig(failure_threshold=_FAILURE_THRESHOLD),
            )
        except Exception as exc:
            print(f"[sre.circuit] init failed: {exc}")
            _breaker = None
    return _breaker

def call(func, *args, fallback=None, **kwargs):
    """Run func through the breaker; returns fallback if the circuit is open."""
    breaker = _build()
    if breaker is None:
        return fallback
    return breaker.call(func, *args, fallback=fallback, **kwargs)

def record(success: bool) -> None:
    """Report one outcome to the breaker."""
    breaker = _build()
    if breaker is None:
        return
    if success:
        breaker.record_success()
    else:
        breaker.record_failure()

def state() -> str:
    """'CLOSED' (healthy) or 'OPEN' (tripped)."""
    breaker = _build()
    if breaker is None:
        return "unknown"
    return breaker.get_state().name

def reset() -> None:
    """Force the breaker back to closed."""
    breaker = _build()
    if breaker is not None:
        breaker.reset()
