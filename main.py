"""FastAPI entrypoint for the RAG assistant.

Run with: uvicorn main:app --reload
Endpoints:
  POST /upload   -> upload a PDF, chunk + embed + store in Chroma
  POST /ask      -> ask a question, get a RAG answer using both memory tiers
                    (NemoGuardrails input/output checks + AGT tool governance)
  GET  /memory   -> list long-term memories for a session
  GET  /history  -> fetch a session's recent short-term chat history
  POST /evaluate -> run RAGAS evaluation over the built-in test cases
  DELETE /memory -> clear long-term memories + short-term buffer for a session

Agent OS operations (see governance.py for the pillar wiring):
  GET  /health            -> liveness + a snapshot of every pillar
  GET  /slo               -> SLO compliance + error budget
  GET  /provenance/verify -> is the anchored commitment still valid?
  POST /provenance/seal   -> anchor the delta chain as a commitment
  POST /admin/pause|resume|kill   -> agent control (AGT signals + kill switch)
  POST /admin/deploy|promote|rollback -> blue/green progressive delivery
"""
from __future__ import annotations

import logging
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

import governance
import rag_engine
from config import settings
from logging_config import configure_logging
from memory import clear_short_term, load_recent_messages, mem0_clear, mem0_get
from security import require_admin

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup and shutdown: logging, config checks, warm-up, governance hook."""
    configure_logging()

    problems = settings.validate()
    for problem in problems:
        logger.warning("config: %s", problem)
    if settings.strict and problems:
        raise RuntimeError("invalid configuration: " + "; ".join(problems))

    # Hook the runtime kill switch into the app's terminated flag.
    governance.register_termination_hook()

    # Open the Postgres connection in the background so the first /history
    # call doesn't pay the ~15s cold-start cost.
    if settings.postgres_ready:
        threading.Thread(
            target=load_recent_messages,
            args=("__warmup__",),
            daemon=True,
        ).start()

    yield


app = FastAPI(title="RAG Assistant", version="1.0.0", lifespan=lifespan)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    session_id: str = "default"


class HealthResponse(BaseModel):
    status: str
    groq_ready: bool
    mem0_ready: bool
    vector_db: str
    guardrails_ready: bool
    agt_ready: bool
    agt_policy_count: int = 0
    agt_audit_enabled: bool = False
    # Agent OS pillars (mesh / runtime / sre / provenance)
    agent_did: str | None = None
    trust_score: int = 0
    ring: str = ""
    circuit: str = ""
    slo_status: str | None = None
    provenance_entries: int = 0
    provenance_ok: bool | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness plus a one-shot snapshot of every Agent OS pillar."""
    from agt.governance import get_agt_engine

    gr = rag_engine.get_guardrails()
    engine = get_agt_engine()
    snap = governance.snapshot()
    return HealthResponse(
        status="ok",
        groq_ready=settings.groq_ready,
        mem0_ready=settings.mem0_ready,
        vector_db=settings.vector_db_path,
        guardrails_ready=gr is not None,
        agt_ready=engine is not None,
        agt_policy_count=len(engine.policies) if engine is not None else 0,
        agt_audit_enabled=settings.agt_audit_enabled,
        agent_did=snap["did"],
        trust_score=snap["trust"],
        ring=snap["ring"],
        circuit=snap["circuit"],
        slo_status=(snap["slo"] or {}).get("status"),
        provenance_entries=snap["provenance_entries"],
        provenance_ok=snap["provenance_ok"],
    )


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Avoid trusting the raw filename: write to a uuid-based path.
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}.pdf"
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not save file: {exc}")

    try:
        count = rag_engine.add_pdf(dest)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not process PDF: {exc}")
    finally:
        # Clean up the temp upload after ingestion.
        if dest.exists():
            dest.unlink(missing_ok=True)

    return {
        "message": f"Uploaded and indexed '{file.filename}'",
        "chunks": count,
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = rag_engine.ask(question, req.session_id)
    if result.get("admitted") is False:
        # Runtime rate budget was exhausted - tell the caller to back off.
        raise HTTPException(status_code=429, detail=result["answer"])
    return result


@app.get("/memory")
def list_memory(session_id: str = "default") -> dict:
    items = mem0_get(session_id)
    return {"session_id": session_id, "memories": items}


@app.get("/history")
def get_history(session_id: str = "default") -> dict:
    """Return the session's recent short-term chat history, oldest first."""
    messages = load_recent_messages(session_id)
    history = [
        {
            "role": "user" if m.type == "human" else "assistant",
            "content": m.content,
        }
        for m in messages
    ]
    return {"session_id": session_id, "history": history}


@app.post("/evaluate")
def evaluate() -> dict:
    """Run RAGAS evaluation over the built-in test cases (can take a while)."""
    try:
        from evaluation import run_evaluation
        return run_evaluation()
    except Exception as exc:  # noqa: BLE001 - surface eval errors to caller
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/memory")
def delete_memory(session_id: str = "default") -> dict:
    mem0_clear(session_id)
    clear_short_term(session_id)
    return {"message": f"Memory cleared for session '{session_id}'."}


# ---------------------------------------------------------------------------
# Agent OS operations
#
# These expose the pillars we built as small, direct endpoints. They are
# grouped here so the governance surface is easy to find.
# ---------------------------------------------------------------------------

@app.get("/slo")
def slo_status() -> dict:
    """Current SLO compliance and error budget (sre pillar)."""
    from sre import slo

    return slo.status()


@app.get("/provenance/verify")
def provenance_verify() -> dict:
    """Check the anchored commitment still matches the live delta chain."""
    return {"verified": governance.verify_provenance()}


@app.post("/provenance/seal", dependencies=[Depends(require_admin)])
def provenance_seal() -> dict:
    """Anchor the current delta-chain root as a verifiable commitment."""
    return {"seal": governance.seal()}


@app.get("/replay/last")
def replay_last() -> dict:
    """Deterministically re-run the last captured turn; report divergence."""
    return governance.replay_trace()


@app.post("/admin/pause", dependencies=[Depends(require_admin)])
def admin_pause() -> dict:
    """Pause the agent (AGT kernel signal - SIGSTOP)."""
    from agt.pillars import signals

    return signals.pause()


@app.post("/admin/resume", dependencies=[Depends(require_admin)])
def admin_resume() -> dict:
    """Resume the agent (AGT kernel signal - SIGCONT)."""
    from agt.pillars import signals

    return signals.resume()


@app.post("/admin/kill", dependencies=[Depends(require_admin)])
def admin_kill(reason: str = "manual") -> dict:
    """Terminate the agent's runtime session via the kill switch.

    After a kill, /ask is refused (429) until /admin/revive is called.
    """
    from runtime import ternot

    return ternot.kill(reason=reason)


@app.post("/admin/revive", dependencies=[Depends(require_admin)])
def admin_revive() -> dict:
    """Clear a previous kill so the agent can serve requests again."""
    governance.revive()
    return {"terminated": governance.terminated()}


@app.post("/admin/deploy", dependencies=[Depends(require_admin)])
def admin_deploy(version: str) -> dict:
    """Progressive delivery: deploy a version to the inactive (green) side."""
    from sre import delivery

    return delivery.deploy(version)


@app.post("/admin/promote", dependencies=[Depends(require_admin)])
def admin_promote() -> dict:
    """Progressive delivery: health-gate and switch traffic to the new side."""
    from sre import delivery

    try:
        return delivery.promote()
    except RuntimeError as exc:
        # No inactive side to promote / validation could not run.
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/admin/rollback", dependencies=[Depends(require_admin)])
def admin_rollback() -> dict:
    """Progressive delivery: switch traffic back to the previous side."""
    from sre import delivery

    try:
        return delivery.rollback()
    except RuntimeError as exc:
        # Nothing to roll back to (no previous switch yet).
        raise HTTPException(status_code=409, detail=str(exc))