"""FastAPI entrypoint for the RAG assistant.

Run with: uvicorn main:app --reload
Endpoints:
  POST /upload   -> upload a PDF, chunk + embed + store in Chroma
  POST /ask      -> ask a question, get a RAG answer using both memory tiers
  GET  /memory   -> list long-term memories for a session
  GET  /history  -> fetch a session's recent short-term chat history
  DELETE /memory -> clear long-term memories + short-term buffer for a session
"""
from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

import rag_engine
from config import settings
from memory import clear_short_term, load_recent_messages, mem0_clear, mem0_get

app = FastAPI(title="RAG Assistant", version="1.0.0")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.on_event("startup")
def _warm_database() -> None:
    """Open the Postgres connection in the background so the first
    /history call doesn't pay the ~15s cold-start cost."""
    if settings.postgres_ready:
        threading.Thread(
            target=load_recent_messages,
            args=("__warmup__",),
            daemon=True,
        ).start()


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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        groq_ready=settings.groq_ready,
        mem0_ready=settings.mem0_ready,
        vector_db=settings.vector_db_path,
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
    return rag_engine.ask(question, req.session_id)


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


@app.delete("/memory")
def delete_memory(session_id: str = "default") -> dict:
    mem0_clear(session_id)
    clear_short_term(session_id)
    return {"message": f"Memory cleared for session '{session_id}'."}