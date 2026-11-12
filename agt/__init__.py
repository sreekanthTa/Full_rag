"""Microsoft Agent Governance Toolkit (AGT) integration for the RAG assistant.

Five governance pillars live in ``agt.pillars``:

    policy   - declarative allow/deny rules   (Pillar 1)
    rbac     - role-based tool permissions    (Pillar 2)
    audit    - tamper-evident decision log    (Pillar 3)
    vfs      - agent scratch / episodic store  (Pillar 4)
    signals  - pause / resume / kill control   (Pillar 5)
"""
from __future__ import annotations

from .pillars import audit, policy, rbac, signals, vfs
from .governance import (
    AGTGovernance,
    get_agt_engine,
    get_agt_governance,
    verify_audit_integrity,
)

__all__ = [
    "policy",
    "rbac",
    "audit",
    "vfs",
    "signals",
    "governance",
    "AGTGovernance",
    "get_agt_engine",
    "get_agt_governance",
    "verify_audit_integrity",
]
