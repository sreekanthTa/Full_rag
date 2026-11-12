"""Pillar 2 - RBAC: role-based permission checks for tool calls."""
from __future__ import annotations
import logging
import threading
from pathlib import Path

from agent_os.integrations.rbac import RBACManager
from config import settings

logger = logging.getLogger(__name__)

_ROLES_PATH = Path(__file__).resolve().parent.parent / "roles.yaml"
_manager = None
_ready = False
_lock = threading.Lock()


def _build():
    global _manager, _ready
    if _ready:
        return _manager
    with _lock:
        if _ready:
            return _manager
        _ready = True
        try:
            _manager = RBACManager.from_yaml(str(_ROLES_PATH)) if _ROLES_PATH.exists() else None
        except Exception as exc:
            logger.warning("rbac init failed: %s", exc)
            _manager = None
    return _manager

def check(agent_id: str, tool_name: str) -> tuple[bool, str]:
    mgr = _build()
    if mgr is None:
        return (not settings.agt_fail_closed), "rbac unavailable"
    try:
        allowed = mgr.has_permission(agent_id, tool_name)
    except Exception as exc:
        return (not settings.agt_fail_closed), f"rbac error: {exc}"
    role = mgr.get_role(agent_id).value
    return (True, "") if allowed else (False, f"role '{role}' may not use '{tool_name}'")