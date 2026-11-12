"""Hypervisor pillar 2 - Commitment Anchoring: seal a delta chain with a verifiable commitment."""
from __future__ import annotations
import logging
import threading

from mesh import identity
from provenance import SESSION, delta

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
            from hypervisor.audit.commitment import CommitmentEngine
            _engine = CommitmentEngine()
        except Exception as exc:
            logger.warning("commitment init failed: %s", exc)
            _engine = None
    return _engine

def seal(participants: list[str] | None = None) -> dict | None:
    """Anchor the current delta-chain root as a commitment."""
    engine = _build()
    root = delta.root()
    if engine is None or root is None:
        return None
    who = participants or [identity.did() or "rag-assistant"]
    record = engine.commit(SESSION, root, who, delta.count())
    return {
        "session": record.session_id,
        "root": record.hash_chain_root,
        "count": record.delta_count,
        "to": record.committed_to,
    }

def verify() -> bool:
    """True if the anchored root still matches the live delta chain."""
    engine = _build()
    root = delta.root()
    if engine is None or root is None:
        return False
    return engine.verify(SESSION, root)
