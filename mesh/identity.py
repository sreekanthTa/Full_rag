"""Mesh pillar 1 - Zero-Trust Identity: who the agent is."""
from __future__ import annotations
import threading
from pathlib import Path

import yaml

_CONFIG = Path(__file__).resolve().parent / "mesh.yaml"

_identity = None
_ready = False
_lock = threading.Lock()

def _load_config() -> dict:
    """Load the mesh.yaml configuration file."""
    if not _CONFIG.exists():
        return {}
    return yaml.safe_load(_CONFIG.read_text())

def _build():
    global _identity, _ready
    if _ready:
        return _identity

    with _lock:
        if _ready:
            return _identity

        _ready = True

        try:
            from agentmesh.identity import AgentIdentity

            cfg = _load_config().get("agent", {})

            _identity = AgentIdentity.create(
                name=cfg.get("name", "rag-assistant"),
                sponsor=cfg.get("sponsor", "sponsor@example.com"),
                capabilities=cfg.get("capabilities", []),
                organization=cfg.get("organization"),
            )

        except Exception as e:
            _identity = None
            _ready = False
    return _identity

def did() -> str | None:
    """Return the agent's Decentralized Identifier (DID)."""
    identity = _build()
    return str(identity.did) if identity else None

def has_capability(capability: str) -> bool:
    """Check if the agent has a specific capability."""
    ident = _build()
    return bool(ident is not None and ident.has_capability(capability))


def is_active() -> bool:
    """True while the identity is neither suspended nor revoked."""
    ident = _build()
    return bool(ident is not None and ident.is_active())