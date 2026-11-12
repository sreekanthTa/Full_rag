"""Mesh pillar 2 - Credentials: short-lived bearer tokens + SPIFFE workload certs."""


from __future__ import annotations
import logging
import threading
from pathlib import Path
import yaml
from mesh import identity

logger = logging.getLogger(__name__)

_CONFIG = Path(__file__).resolve().parent / "mesh.yaml"

_manager = None
_spiffe = None
_ready = False
_lock = threading.Lock()


def _load_config() -> dict:
    if not _CONFIG.exists():
        return {}

    return yaml.safe_load(_CONFIG.read_text())


def _build():
    global _manaager, _spiffe, _ready

    if _ready:
        return

    with _lock:
        if _ready:
            return
        _ready = True

        try:
            from agentmesh.identity import CredentialManager, SPIFFEIdentity
            cfg = _load_config()
            _manager = CredentialManager(default_ttl=int(cfg.get("credential_ttl_seconds", 900)))

            did = identity.did()
            if did:
                agent = cfg.get("agent", {})
                _spiffe = SPIFFEIdentity.create(
                    agent_did=did,
                    agent_name=agent.get("name", "rag-assistant"),
                    trust_domain=cfg.get("trust_domain", "agentmesh.local"),
                    organization=agent.get("organization"),
                )
        except Exception as exc:
            logger.warning("credentials init failed: %s", exc)

def issue(capabilites = None, resources = None, ttl_seconds=None) -> dict | None:
    """Mint a short-lived credential (bearer token) for this agent."""
    _build()

    if _manager is None:
        return None
    did = identity.did()

    if did is None:
        return None

    cred = _manager.issue(
            agent_did=did, capabilities=capabilites, resources=resources, ttl_seconds=ttl_seconds
        )
    
    return cred.model_dump()

def validate(token: str) -> dict | None:
    """Return the credential behind a raw token if still valid, else None."""
    _build()
    if _manager is None:
        return None
    cred = _manager.validate(token)
    return cred.model_dump() if cred is not None else None

def spiffe_id() -> str | None:
    """This agent's SPIFFE ID (workload identity), or None."""
    _build()
    return _spiffe.spiffe_id if _spiffe is not None else None

def issue_svid(ttl_hours: int = 1) -> dict | None:
    """Issue a SPIFFE SVID (x509 workload certificate) for this agent."""
    _build()
    if _spiffe is None:
        return None
    return _spiffe.issue_svid(ttl_hours=ttl_hours).model_dump()                      

          