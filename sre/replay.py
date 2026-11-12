"""SRE pillar 2 - Replay: capture an execution as a trace and re-run it."""
from __future__ import annotations
import logging
import threading

from mesh import identity

logger = logging.getLogger(__name__)

_engine = None
_ready = False
_lock = threading.Lock()

def _build():
    global _engine, _ready
    if _ready:
        return _engine
    with _lock:
        if _ready:
            return _engine
        _ready = True
        try:
            from agent_sre.replay import ReplayEngine
            _engine = ReplayEngine()
        except Exception as exc:
            logger.warning("replay init failed: %s", exc)
            _engine = None
    return _engine

def capture(task: str, steps: list) -> object:
    """Record a task as a trace of (name, output) steps."""
    from agent_sre.replay import TraceCapture
    tc = TraceCapture(identity.did() or "rag-assistant", task)
    for name, output in steps:
        tc.start_span(name)
        tc.end_span(output=output)
    return tc.trace

def replay(trace, overrides: dict | None = None) -> dict:
    """Re-run a captured trace; report any divergence from the original."""
    engine = _build()
    if engine is None:
        return {}
    result = engine.replay(trace, overrides)
    return {
        "original": result.original_trace_id,
        "success": result.success,
        "steps": result.steps_total,
        "diffs": [{"type": d.diff_type.value, "span": d.span_name} for d in result.diffs],
    }
