"""Agent OS governance: how the RAG app uses the pillars we built.

The app answers one question at a time (``rag_engine.ask``). On the way it
touches each pillar, in this order:

    mesh        -> WHO is asking            identity + trust score
    runtime     -> WHAT they may touch      execution ring + rate budget
    agt         -> is this tool allowed?    policy + rbac + audit
    sre.circuit -> shield the model call    trip on repeated failures
    sre.slo     -> did we keep our promise? record the outcome
    provenance  -> tamper-evident record    hash-linked delta chain

This module is deliberately thin: one small, commented function per pillar,
each delegating to the pillar module. Nothing here re-implements a pillar, and
nothing here takes the app down when a pillar is unavailable - the pillars
themselves degrade safely.
"""
from __future__ import annotations

from contextlib import contextmanager


# ---------------------------------------------------------------------------
# mesh + runtime - who is asking, and what their trust score lets them touch
# ---------------------------------------------------------------------------

def context() -> dict:
    """Describe the caller: identity, trust, and the runtime ring it earns."""
    from mesh import identity, trust
    from runtime import rings

    return {
        "did": identity.did(),
        "trust": trust.score(),
        "trust_level": trust.level(),
        "ring": rings.ring(),
        "ring_limits": rings.constraints(),
    }


# ---------------------------------------------------------------------------
# runtime - admission control: the one gate every request must pass
# ---------------------------------------------------------------------------

# Set when the runtime kill switch fires. Every later request is refused until
# the process restarts or ``revive`` is called - a kill is a hard stop.
_terminated = False


def _on_kill() -> None:
    """Callback the kill switch invokes to stop this agent."""
    global _terminated
    _terminated = True


def register_termination_hook() -> bool:
    """Wire the runtime kill switch to our terminated flag (run once at startup)."""
    from runtime import ternot

    return ternot.register(_on_kill)


def terminated() -> bool:
    """True once the kill switch has fired."""
    return _terminated


def revive() -> None:
    """Clear the terminated flag (the counterpart to a kill)."""
    global _terminated
    _terminated = False


def admit(cost: float = 1.0) -> tuple[bool, str]:
    """Decide whether a request may be served.

    Lifecycle comes first - a kill or an AGT pause is a hard stop - then the
    runtime rate budget is spent. Returns ``(True, "")`` when allowed, else
    ``(False, reason)``. Call it at the very start of a request so nothing is
    done for a request we are going to refuse anyway.
    """
    from agt.pillars import signals
    from runtime import limit

    if _terminated:
        return False, "the agent has been terminated"
    if not signals.status().get("running", True):
        return False, "the agent is paused"
    if not limit.allow(cost):
        return False, "rate budget exhausted for this agent"
    return True, ""


# ---------------------------------------------------------------------------
# sre - shield the model call with the circuit breaker, then record the result
# ---------------------------------------------------------------------------

def generate(messages) -> str:
    """Call the LLM behind the circuit breaker and return the answer text.

    The breaker watches for repeated failures: once it trips, calls return the
    fallback immediately instead of hanging on a broken model.
    """
    from sre import circuit

    def _call() -> str:
        from rag_engine import get_llm

        response = get_llm().invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    return circuit.call(_call, fallback="I'm sorry, the assistant is temporarily unavailable.")


def record_outcome(ok: bool) -> None:
    """Feed one request's success/failure into the SLO / error budget."""
    from sre import slo

    slo.record(ok)


# ---------------------------------------------------------------------------
# provenance - an append-only, tamper-evident record of what changed
# ---------------------------------------------------------------------------

def record_state(path: str, operation: str, content_hash: str | None = None) -> dict | None:
    """Append one state change (a file ingest, a config edit) to the delta chain."""
    from provenance import delta

    return delta.record(path, operation, content_hash)


def seal() -> dict | None:
    """Anchor the current delta-chain root as a verifiable commitment."""
    from provenance import commitment

    return commitment.seal()


def verify_provenance() -> bool:
    """True if the anchored commitment still matches the live delta chain."""
    from provenance import commitment

    return commitment.verify()


# ---------------------------------------------------------------------------
# sre replay - capture a turn as a trace so it can be replayed later
# ---------------------------------------------------------------------------

_last_trace = None


def capture(task: str, steps: list):
    """Record a turn as a replayable trace, and remember it as the latest."""
    global _last_trace
    from sre import replay

    _last_trace = replay.capture(task, steps)
    return _last_trace


def replay_trace(overrides: dict | None = None) -> dict:
    """Re-run the most recent captured trace and report any divergence."""
    if _last_trace is None:
        return {"replayed": False, "reason": "no trace captured yet"}
    from sre import replay

    return replay.replay(_last_trace, overrides)


# ---------------------------------------------------------------------------
# runtime sandbox - run a risky operation inside an isolated session
# ---------------------------------------------------------------------------

@contextmanager
def sandbox_session():
    """Yield an isolated execution session id (or None if unavailable).

    Keeps the async Hypervisor out of the caller's way, so ingest code can read
    as a plain ``with`` block::

        with sandbox_session() as sid:
            ... do the risky work ...
    """
    from runtime import sandbox

    session_id = sandbox.create()
    try:
        yield session_id
    finally:
        if session_id:
            sandbox.close(session_id)


# ---------------------------------------------------------------------------
# health - one snapshot across every pillar, used by the /health endpoint
# ---------------------------------------------------------------------------

def snapshot() -> dict:
    """Collect the current state of each pillar into a single dict."""
    from mesh import identity, trust
    from runtime import limit, rings
    from sre import circuit, slo
    from provenance import delta

    entries = delta.count()
    return {
        "did": identity.did(),
        "trust": trust.score(),
        "ring": rings.ring(),
        "limits": limit.stats(),
        "circuit": circuit.state(),
        "slo": slo.status(),
        # Only meaningful once at least one delta exists.
        "provenance_entries": entries,
        "provenance_ok": delta.verify()[0] if entries else None,
    }
