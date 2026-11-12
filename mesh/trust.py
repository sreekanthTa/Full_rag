"""Mesh pillar 3 - Trust: this agent's 0-1000 reputation score."""

from __future__ import annotations
import logging
import threading

from config import settings
from mesh import identity

logger = logging.getLogger(__name__)

_DEFAULT_SCORE = 500

_tracker = None
_agent = None
_ready = False
_lock = threading.Lock()

def _build():
    global _tracker, _agent, _ready
    if _ready:
        return 

    with  _lock:
        if _ready:
            return

        _ready = True

        try:
            from agentmesh.trust_types import TrustTracker
            _tracker = TrustTracker()
            _agent = identity.did() or settings.agt_agent_id
        except Exception as exc:
                    logger.warning("trust init failed: %s", exc)

def score() -> int:
    """Current trust score on the 0-1000 scale."""

    _build()

    if _tracker is None:
         return _DEFAULT_SCORE

    return round(_tracker.get_score(_agent) * 1000)

def level() -> str:
     """Trust tier name for the current score."""

     _build()

     if _tracker is None:
          return "unknown"

     from agentmesh.trust.levels import trust_level_for_score

     return trust_level_for_score(score())


def record(peer_id: str, action: str, success: bool) -> int:
     """Record one interaction outcome; return the new 0-1000 score."""
     _build()

     if _tracker is None:
          return  _DEFAULT_SCORE

     _tracker.record_interaction(_agent, peer_id, action, success)

     return score()
