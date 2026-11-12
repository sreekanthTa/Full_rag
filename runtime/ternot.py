"""Runtime pillar 4 - Termination Control: the kill switch."""
from __future__ import annotations
import logging
import threading

from config import settings
from mesh import identity

logger = logging.getLogger(__name__)

_session = "main"
_switch = None
_ready = False
_lock = threading.Lock()

def _build():
    global _switch, _ready
    if _ready:
        return _switch
    with _lock:
        if _ready:
            return _switch
        _ready = True
        try:
            from agent_runtime import KillSwitch
            _switch = KillSwitch()
        except Exception as exc:
            logger.warning("termination init failed: %s", exc)
            _switch = None
    return _switch

def _agent() -> str:
    return identity.did() or settings.agt_agent_id

def _reason(name: str):
    from agent_runtime import KillResult
    enum_cls = type(KillResult().reason)
    return enum_cls[name.upper()] if name.upper() in enum_cls.__members__ else enum_cls.MANUAL

def register(process_handle) -> bool:
    """Register our agent's shutdown callback (invoked on kill)."""
    switch = _build()
    if switch is None:
        return False
    switch.register_agent(_agent(), process_handle)
    return True

def kill(reason: str = "manual", details: str = "") -> dict:
    """Kill our agent's session; returns a summary of what happened."""
    switch = _build()
    if switch is None:
        return {}
    result = switch.kill(_agent(), _session, _reason(reason), details=details)
    return {
        "terminated": result.terminated,
        "reason": result.reason.value,
        "compensation": result.compensation_triggered,
        "handoffs": result.handoff_success_count,
    }