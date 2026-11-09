"""Tamper-evident audit log for AGT tool-call decisions.

Thin wrapper over the toolkit's FlightRecorder (SQLite + Merkle hash chain).
Every allow/deny decision is written as one trace so the log can be verified
after the fact with `verify_integrity()`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from config import settings


class GovernanceAudit:
    """Records governance decisions to a Merkle-chained SQLite log."""

    def __init__(self) -> None:
        from agent_os import FlightRecorder

        db_path = Path(settings.agt_audit_db)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._recorder = FlightRecorder(db_path=str(db_path), enable_batching=True)

    def trace(self, tool_name: str, args: dict, input_prompt: Optional[str] = None) -> str:
        return self._recorder.start_trace(
            agent_id=settings.agt_agent_id,
            tool_name=tool_name,
            tool_args=args,
            input_prompt=input_prompt,
        )

    def allow(self, trace_id: str, result: Any = None, execution_time_ms: Optional[float] = None) -> None:
        self._recorder.log_success(trace_id, result=result, execution_time_ms=execution_time_ms)

    def deny(self, trace_id: str, reason: str) -> None:
        self._recorder.log_violation(trace_id, reason)

    def verify(self) -> dict:
        return self._recorder.verify_integrity()

    def close(self) -> None:
        self._recorder.close()
