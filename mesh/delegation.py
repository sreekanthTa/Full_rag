"""Mesh pillar 4 - Delegation: hand a narrowed subset of capabilities to a child agent."""
from __future__ import annotations
import threading

from mesh import identity

_MAX_DEPTH = 5  # = agentmesh.constants.DEFAULT_DELEGATION_MAX_DEPTH

_chain = None
_ready = False
_lock = threading.Lock()

def _build():
    global _chain, _ready
    if _ready:
        return _chain
    with _lock:
        if _ready:
            return _chain
        _ready = True
        try:
            from agentmesh.identity.delegation import ScopeChain
            parent = identity.current()
            if parent is not None:
                chain, root = ScopeChain.create_root(
                    sponsor_email=parent.sponsor_email,
                    root_agent_did=str(parent.did),
                    capabilities=list(parent.capabilities),
                    sponsor_verified=parent.sponsor_verified,
                )
                chain.add_link(root)
                _chain = chain
        except Exception as exc:
            print(f"[mesh.delegation] init failed: {exc}")
            _chain = None
    return _chain

def delegate(name: str, capabilities: list[str]) -> dict | None:
    """Mint a child agent holding only a NARROWED subset of this agent's capabilities."""
    chain = _build()
    parent = identity.current()
    if chain is None or parent is None:
        return None
    if chain.get_depth() >= _MAX_DEPTH:
        print("[mesh.delegation] refused: max delegation depth reached")
        return None

    granted = [c for c in capabilities if parent.has_capability(c)]  # narrowing
    if not granted:
        return None

    child = parent.delegate(name=name, capabilities=granted)

    from agentmesh.identity.delegation import DelegationLink
    prev = chain.links[-1]
    link = DelegationLink(
        link_id=str(child.did),
        depth=len(chain.links),
        parent_did=str(parent.did),
        child_did=str(child.did),
        parent_capabilities=list(parent.capabilities),
        delegated_capabilities=granted,
        parent_signature="local",
        previous_link_hash=prev.link_hash,
        link_hash="",
    )
    link = link.model_copy(update={"link_hash": link.compute_hash()})
    chain.add_link(link)

    return {
        "name": child.name,
        "did": str(child.did),
        "parent_did": child.parent_did,
        "depth": child.delegation_depth,
        "capabilities": granted,
    }

def verify() -> tuple[bool, str | None]:
    """Validate the whole chain (hash links + structure)."""
    chain = _build()
    if chain is None:
        return False, "delegation chain unavailable"
    return chain.verify()

def trace(capability: str) -> list[dict]:
    """Full provenance: who granted this capability, hop by hop."""
    chain = _build()
    return chain.trace_capability(capability) if chain is not None else []