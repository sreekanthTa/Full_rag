"""Mesh pillar 5 - Protocol: A2A / MCP agent cards and the peer trust gate."""
from __future__ import annotations
import logging
import threading

from mesh import identity

logger = logging.getLogger(__name__)

_REQUIRED_SCORE = 700  # agentmesh default peer-trust threshold

_card = None
_bridge = None
_ready = False
_lock = threading.Lock()

def _build():
    global _card, _bridge, _ready
    if _ready:
        return
    with _lock:
        if _ready:
            return
        _ready = True
        try:
            from agentmesh.trust import TrustedAgentCard, TrustBridge, ProtocolBridge
            ident = identity.current()
            if ident is not None:
                c = TrustedAgentCard.from_identity(ident)
                c.sign(ident)
                _card = c
                _bridge = ProtocolBridge(
                    agent_did=str(ident.did),
                    trust_bridge=TrustBridge(agent_did=str(ident.did)),
                )
        except Exception as exc:
            logger.warning("protocol init failed: %s", exc)

def card() -> dict | None:
    """Our signed Agent Card - what we advertise to peers over A2A/MCP."""
    _build()
    return _card.model_dump() if _card is not None else None

def verify_card(data: dict) -> bool:
    """A peer is only as real as its signature: False if the card was tampered with."""
    try:
        from agentmesh.trust import TrustedAgentCard
        return bool(TrustedAgentCard(**data).verify_signature())
    except Exception as exc:
        logger.warning("verify_card failed: %s", exc)
        return False

def accepts(data: dict, required_score: int = _REQUIRED_SCORE) -> bool:
    """Zero-trust gate: accept a peer only if its card verifies AND its score clears the bar."""
    if not verify_card(data):
        return False
    claimed = int(round(float(data.get("trust_score", 0.0)) * 1000))
    return claimed >= required_score

def protocol_for(peer_did: str) -> str | None:
    """The wire protocol we'd speak to this peer (a2a/mcp/...), or None if unknown."""
    _build()
    return _bridge.get_protocol_for_peer(peer_did) if _bridge is not None else None