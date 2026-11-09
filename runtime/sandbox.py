"""Runtime pillar 3 - Sandboxing: isolated sessions with their own execution ring."""
from __future__ import annotations
import asyncio
import concurrent.futures
import threading

from mesh import identity

_hypervisor = None
_ready = False
_lock = threading.Lock()

def _build():
    global _hypervisor, _ready
    if _ready:
        return _hypervisor
    with _lock:
        if _ready:
            return _hypervisor
        _ready = True
        try:
            from agent_runtime import Hypervisor
            _hypervisor = Hypervisor()
        except Exception as exc:
            print(f"[runtime.sandbox] init failed: {exc}")
            _hypervisor = None
    return _hypervisor

def _run(coro):
    """Run an async Hypervisor call from sync code, even inside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()

def create() -> str | None:
    """Create and activate an isolated session for our agent; return its id."""
    h = _build()
    if h is None:
        return None
    from agent_runtime import SessionConfig

    async def _go():
        session = await h.create_session(SessionConfig(), identity.did() or "rag-assistant")
        sid = session.sso.session_id
        await h.activate_session(sid)
        return sid

    try:
        return _run(_go())
    except Exception as exc:
        print(f"[runtime.sandbox] create failed: {exc}")
        return None

def join(session_id: str, sigma: float = 0.9) -> str | None:
    """Join our agent to a session; return the ring it was assigned."""
    h = _build()
    if h is None:
        return None
    try:
        ring = _run(h.join_session(session_id, identity.did() or "rag-assistant", sigma_raw=sigma))
        return ring.name if ring is not None else None
    except Exception as exc:
        print(f"[runtime.sandbox] join failed: {exc}")
        return None

def monitor() -> list:
    """Sessions flagged for behavioral drift."""
    h = _build()
    if h is None:
        return []
    try:
        return _run(h.monitor_sessions())
    except Exception as exc:
        print(f"[runtime.sandbox] monitor failed: {exc}")
        return []

def close(session_id: str) -> str | None:
    """Terminate a session; return the final state."""
    h = _build()
    if h is None:
        return None
    try:
        return _run(h.terminate_session(session_id))
    except Exception as exc:
        print(f"[runtime.sandbox] close failed: {exc}")
        return None