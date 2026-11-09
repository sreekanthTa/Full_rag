"""Pillar 3 - Audit: tamper-evident record of every governance decision."""

from __future__ import annotations
import threading
from pathlib import Path

from config import settings

_recorder = None
_ready = False
_lock = threading.Lock()


def _build():
    global _recorder, _ready
    if _ready:
        return _recorder
    with _lock:
        if _ready:
            return _recorder
        _ready = True
        try:
            from agent_os import FlightRecorder
            db = Path(settings.agt_audit_db)
            db.parent.mkdir(parents=True, exist_ok=True)
            _recorder = FlightRecorder(str(db))
        except Exception as exc:
            print(f"[agt.audit] init failed: {exc}")
            _recorder = None

    return _recorder


def record(tool_name: str, args:dict, allowed: bool, reason: str):
    """Record a governance decision in the audit log."""
    rec = _build()
    if rec is None:
        return
    try:
      trace_id = rec.start_trace(
          agent_id=settings.agt_agent_id,
          tool_name=tool_name,
          tool_args=args,
      ) 

      if allowed:
          rec.log_success(trace_id)
      else:
          rec.log_violation(trace_id, reason)

    except Exception as exc:
        print(f"[agt.audit] record failed: {exc}")

def verify() -> dict:
    rec = _build()
    if rec is None:
        return {"valid": True, "total_entries": 0, "message": "audit disabled"}
    return rec.verify_integrity()
