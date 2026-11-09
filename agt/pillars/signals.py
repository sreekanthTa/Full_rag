"""Pillar 5 - Signals: kernel-level pause/resume/kill control for the agent."""
from __future__ import annotations
import threading

from agent_control_plane.signals import AgentSignal, SignalDispatcher
from config import settings

_dispatcher = None
_ready = False
_lock = threading.Lock()


def _build():
    global _dispatcher, _ready
    if _ready:
        return _dispatcher
    with _lock:
        if _ready:
            return _dispatcher
        _ready = True
        try:
            _dispatcher = SignalDispatcher(agent_id=settings.agt_agent_id)
        except Exception as exc:
            print(f"[agt.signals] init failed: {exc}")
            _dispatcher = None
    return _dispatcher


def pause() -> dict:
    """Pause the agent (SIGSTOP)."""
    disp = _build()
    if disp:
        disp.signal(AgentSignal.SIGSTOP, source="api")
    return status()


def resume() -> dict:
    """Resume the agent (SIGCONT)."""
    disp = _build()
    if disp:
        disp.signal(AgentSignal.SIGCONT, source="api")
    return status()


def kill(reason: str = "killed via API") -> dict:
    """Terminate the agent (SIGKILL). SIGKILL raises AgentKernelPanic by design."""
    disp = _build()
    if disp:
        try:
            disp.signal(AgentSignal.SIGKILL, source="api", reason=reason)
        except Exception:  # noqa: BLE001 - SIGKILL raises AgentKernelPanic
            pass
    return status()


def status() -> dict:
    """Current agent state."""
    disp = _build()
    if disp is None:
        return {"running": True, "stopped": False, "terminated": False}
    return {
        "running": disp.is_running,
        "stopped": disp.is_stopped,
        "terminated": disp.is_terminated,
    }