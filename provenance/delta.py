"""Hypervisor pillar 1 - Delta Engine: record state changes as a hash-linked chain."""
from __future__ import annotations
import logging
import threading

from mesh import identity
from provenance import SESSION

logger = logging.getLogger(__name__)

_engine = None
_ready = False
_lock = threading.Lock()

def _build():
    global _engine, _ready
    if _ready:
        return _engine
    with _lock:
        if _ready:
            return _engine
        _ready = True
        try:
            from hypervisor.audit.delta import DeltaEngine
            _engine = DeltaEngine(SESSION)
        except Exception as exc:
            logger.warning("delta init failed: %s", exc)
            _engine = None
    return _engine

def record(path: str, operation: str, content_hash: str | None = None) -> dict | None:
    """Record one state change (e.g. a file write) into the delta chain."""
    engine = _build()
    if engine is None:
        return None
    from hypervisor.audit.delta import VFSChange
    change = VFSChange(path=path, operation=operation, content_hash=content_hash)
    delta = engine.capture(identity.did() or "rag-assistant", [change])
    return {
        "id": delta.delta_id,
        "turn": delta.turn_id,
        "hash": delta.delta_hash,
        "parent": delta.parent_hash,
    }

def count() -> int:
    """How many deltas are in the chain."""
    engine = _build()
    return len(engine._deltas) if engine is not None else 0

def root() -> str | None:
    """The current hash-chain root - one fingerprint for every delta so far."""
    engine = _build()
    return engine.compute_hash_chain_root() if engine is not None else None

def verify() -> tuple[bool, str | None]:
    """Check the delta chain is intact (nothing was tampered with)."""
    engine = _build()
    if engine is None:
        return False, "delta engine unavailable"
    return engine.verify_chain()
