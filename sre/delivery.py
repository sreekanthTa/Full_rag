"""SRE pillar 3 - Progressive Delivery: blue/green rollouts gated on health."""
from __future__ import annotations
import logging
import threading

logger = logging.getLogger(__name__)

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
            from agent_sre.delivery import BlueGreenManager, BlueGreenConfig
            _manager = BlueGreenManager(BlueGreenConfig(drain_timeout_seconds=0))
        except Exception as exc:
            logger.warning("delivery init failed: %s", exc)
            _manager = None
    return _manager

def _env(env) -> dict:
    return {
        "name": getattr(env.name, "value", env.name),
        "state": getattr(env.state, "value", env.state),
        "version": env.agent_version,
        "health": env.health_status,
    }

def deploy(version: str) -> dict:
    """Deploy a new version to the inactive side."""
    manager = _build()
    if manager is None:
        return {}
    return _env(manager.deploy(version))

def promote(health_check=None) -> dict:
    """Health-gate the new side, then switch traffic to it."""
    manager = _build()
    if manager is None:
        return {}
    if not manager.validate(health_check or (lambda: True)):
        return {"switched": False, "reason": "health check failed"}
    return {"switched": True, **_env(manager.switch())}

def rollback() -> dict:
    """Switch traffic back to the previous side."""
    manager = _build()
    if manager is None:
        return {}
    return _env(manager.rollback())

def status() -> dict:
    """Blue/green status for both sides."""
    manager = _build()
    return manager.get_status() if manager is not None else {}
