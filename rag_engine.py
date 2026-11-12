"""RAG engine: PDF ingest, chunking, chunk embedding, Chroma persistence,
and retrieval-augmented answering via langchain_groq.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Optional

import chromadb
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import governance
from config import settings
from memory import load_recent_messages, save_exchange, mem0_add, mem0_get
from agt import get_agt_governance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared resources
# ---------------------------------------------------------------------------

_GUARDRAILS_DIR = Path(__file__).resolve().parent / "guardrails"

_chroma_client = None
_collection = None
_llm = None
_rails = None
_rails_ready = False
_agt = None
_agt_ready = False


def get_client() -> chromadb.ClientAPI:
    """Lazy Chroma client persisted at settings.vector_db_path."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.vector_db_path)
    return _chroma_client


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        agt = get_agt_governance()
        if agt is not None:
            allowed, reason = agt.evaluate_tool_call("chroma_query", {"operation": "get_or_create"})
            if not allowed:
                raise PermissionError(f"AGT denied chroma_query: {reason}")
        _collection = get_client().get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        agt = get_agt_governance()
        if agt is not None:
            allowed, reason = agt.evaluate_tool_call("llm_generate", {"model": "openai/gpt-oss-20b"})
            if not allowed:
                raise PermissionError(f"AGT denied llm_generate: {reason}")
        _llm = ChatGroq(
            api_key=settings.groq_api_key,
            model="openai/gpt-oss-20b",
            temperature=0.2,
            # gpt-oss-20b spends tokens on reasoning; leave enough room that
            # longer structured outputs (e.g. RAGAS judge calls) aren't cut off.
            max_tokens=8192,
        )
    return _llm


def _init_rails() -> Optional[object]:
    """Lazy-build the NemoGuardrails engine from guardrails/config.yml."""
    global _rails, _rails_ready
    if _rails_ready:
        return _rails
    _rails_ready = True
    try:
        from nemoguardrails import RailsConfig, LLMRails

        config = RailsConfig.from_path(str(_GUARDRAILS_DIR))
        _rails = LLMRails(config)
    except Exception as exc:  # noqa: BLE001 - guardrails must never break the app
        logger.warning("guardrails initialization failed: %s", exc)
        _rails = None
    return _rails


def get_guardrails() -> Optional[object]:
    """Return the guardrails engine, or None if disabled/unavailable."""
    if not settings.guardrails_enabled:
        return None
    return _init_rails()


def get_agt_governance() -> Optional[object]:
    """Return the AGT governance engine, or None if disabled/unavailable."""
    global _agt, _agt_ready
    if _agt_ready:
        return _agt
    _agt_ready = True
    if not settings.agt_enabled:
        return None
    try:
        from agt import AGTGovernance

        _agt = AGTGovernance()
    except Exception as exc:  # noqa: BLE001 - AGT must never break the app
        logger.warning("agt initialization failed: %s", exc)
        _agt = None
    return _agt


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def add_pdf(pdf_path: str | Path) -> int:
    """Load a PDF, split into chunks, embed, and store in Chroma.

    Runs under Agent OS governance: AGT approves the ingest tool call, the
    runtime sandbox gives the work an isolated session, and the result is
    appended to the tamper-evident provenance chain. Returns chunk count.
    """
    path = Path(pdf_path)

    # AGT tool-level check (policy + rbac + audit) before touching the file.
    agt = get_agt_governance()
    if agt is not None:
        allowed, reason = agt.evaluate_tool_call("pdf_ingest", {"path": str(path)})
        if not allowed:
            raise PermissionError(f"AGT denied pdf_ingest: {reason}")

    raw = path.read_bytes()

    # Run the ingest inside an isolated runtime session (the sandbox pillar).
    with governance.sandbox_session():
        docs = PyPDFLoader(str(path)).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = splitter.split_documents(docs)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        for i, chunk in enumerate(chunks):
            content = chunk.page_content
            ids.append(str(uuid.uuid4()))
            documents.append(content)
            metadatas.append({
                "source": path.name,
                "chunk": i,
                # cheap content hash so re-upload of identical text can be deduped
                "hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            })

        get_collection().add(ids=ids, documents=documents, metadatas=metadatas)

    # Append the ingest to the tamper-evident provenance chain.
    governance.record_state(path.name, "ingest", hashlib.sha256(raw).hexdigest())
    return len(chunks)


def search(query: str, top_k: int = 4) -> list[str]:
    """Return the top_k most relevant chunk texts for a query.

    The query text is passed straight to Chroma so it embeds with the same
    model used at ingestion time (Chroma's default embedding function).
    """
    agt = get_agt_governance()
    if agt is not None:
        allowed, reason = agt.evaluate_tool_call(
            "chroma_query",
            {"query": query[:200], "top_k": top_k},
        )
        if not allowed:
            logger.warning("chroma_query denied: %s", reason)
            return []

    collection = get_collection()
    result = collection.query(
        query_texts=[query],
        n_results=top_k,
    )
    docs = result.get("documents") or []
    return docs[0] if docs else []


def format_context(chunks: list[str], max_chars: int = 6000) -> str:
    merged = "\n\n---\n\n".join(chunks)
    return merged[:max_chars]


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------

def _prompt(question: str, context: str, long_term: list, recent_msgs: list) -> list:
    sections = [
        "You are a helpful assistant that answers questions strictly from the",
        "provided document context and memory. If the answer is not in the",
        "context, say you don't know. Cite the source snippets where possible.",
    ]

    if long_term:
        facts = "\n".join((m.get("memory", "") or m.get("text", ""))
                          for m in long_term)
        sections.append(f"Long-term memories about this user:\n{facts}")

    if recent_msgs:
        recent = "\n".join(f"{type(m).__name__}: {m.content}" for m in recent_msgs)
        sections.append(f"Recent conversation:\n{recent}")

    sections.append(f"Relevant document context:\n{context}")
    return [SystemMessage(content="\n\n".join(sections)), HumanMessage(content=question)]


def _record(session_id: str, question: str, answer: str, ok: bool,
            operation: str) -> None:
    """Persist an exchange and log its governance outcome.

    Writes to both memory tiers, feeds the SLO / error budget, and appends a
    provenance delta (the answer's hash) to the tamper-evident chain.
    """
    save_exchange(session_id, question, answer)
    mem0_add(question, answer, session_id)
    governance.record_outcome(ok)
    governance.record_state(session_id, operation,
                            hashlib.sha256(answer.encode("utf-8")).hexdigest())


def ask(question: str, session_id: str = "default") -> dict:
    """Run the full RAG answer flow for one question, under Agent OS governance.

    The flow, in order:

      1. context  - mesh identity + trust decide the runtime ring
      2. admit    - runtime rate budget; refuse before doing any real work
      3. retrieve - Chroma search + both memory tiers build the prompt
      4. generate - AGT checks the tool call, the circuit breaker guards the
                    model call, and (when enabled) NemoGuardrails wrap the turn
      5. record   - SLO outcome + a provenance delta for the exchange

    Short-term context is the session's Postgres history; long-term context is
    Mem0. The Q/A exchange is saved back into both.
    """
    # 1. Who is asking, and which ring their trust score earns.
    agent = governance.context()

    # 2. Admission control - refuse an exhausted budget before any work.
    admitted, reason = governance.admit()
    if not admitted:
        return {
            "answer": f"I'm sorry, {reason}.",
            "sources": [],
            "session_id": session_id,
            "admitted": False,
            "guardrails_blocked": False,
            "agt_blocked": False,
            "agt_reason": "",
            "agent": agent,
        }

    # 3. Retrieval + memory - build the grounded prompt.
    chunks = search(question)
    context = format_context(chunks) if chunks else ""
    long_term = mem0_get(session_id)
    recent = load_recent_messages(session_id)
    messages = _prompt(question, context, long_term, recent)

    agt = get_agt_governance()
    agt_blocked = False
    agt_reason = ""

    # Governance engine unavailable: honor fail-closed before touching the model.
    if agt is None and settings.agt_fail_closed:
        agt_blocked = True
        agt_reason = "governance unavailable"
        refusal = "I'm sorry, I can't respond to that. governance unavailable"
        _record(session_id, question, refusal, ok=False, operation="agt_unavailable")
        return {
            "answer": refusal,
            "sources": [],
            "session_id": session_id,
            "admitted": True,
            "guardrails_blocked": False,
            "agt_blocked": True,
            "agt_reason": agt_reason,
            "agent": agent,
        }

    # 4. Generation, with NemoGuardrails around it when enabled.
    guardrails = get_guardrails()
    if guardrails is not None:
        # Register the RAG step as a Nemo action, then run the turn through Nemo
        # so the flows in guardrails/config.yml apply input/output rails.
        def _rag_answer(**_kwargs) -> str:
            # AGT pre-tool-call check, then the circuit-guarded model call.
            agt_inner = get_agt_governance()
            if agt_inner is not None:
                allowed, why = agt_inner.evaluate_tool_call(
                    "llm_generate",
                    {"prompt_length": sum(len(m.content) for m in messages)},
                )
                if not allowed:
                    raise PermissionError(f"AGT denied llm_generate: {why}")
            return governance.generate(messages)

        guardrails.register_action(_rag_answer, name="GenerateRagAnswer")
        try:
            result = guardrails.generate(
                messages=[{"role": "user", "content": question}],
                # Enable the `self check facts` output rail and ground it on
                # the retrieved chunks.
                state={"check_facts": True, "relevant_chunks": chunks},
            )
            content = (
                result.get("content", "") if isinstance(result, dict) else str(result)
            )
            answer, blocked, reason = content.strip(), False, ""
        except Exception as exc:  # noqa: BLE001 - enable_rails_exceptions -> blocked
            answer, blocked, reason = "I'm sorry, I can't respond to that.", True, str(exc)

        if blocked:
            # Still persist the blocked exchange for auditing, but return the
            # guardrail refusal instead of the model output.
            _record(session_id, question, answer, ok=False, operation="blocked")
            return {
                "answer": answer,
                "sources": [],
                "session_id": session_id,
                "admitted": True,
                "guardrails_blocked": True,
                "guardrails_reason": reason,
                "agt_blocked": agt_blocked,
                "agt_reason": agt_reason,
                "agent": agent,
            }
    else:
        # No guardrails: AGT check directly, then the guarded model call.
        if agt is not None:
            allowed, reason = agt.evaluate_tool_call(
                "llm_generate",
                {"prompt_length": sum(len(m.content) for m in messages)},
            )
            if not allowed:
                agt_blocked = True
                agt_reason = reason
                answer = f"I'm sorry, I can't respond to that. {reason}"
            else:
                answer = governance.generate(messages)
        else:
            answer = governance.generate(messages)

    # 5. Persist the exchange, record the SLO outcome + a provenance delta,
    #    and capture a replayable trace of the turn.
    _record(session_id, question, answer, ok=True, operation="ask")
    trace = governance.capture(question, [
        ("retrieve", {"chunks": len(chunks)}),
        ("generate", {"answer": answer}),
    ])

    return {
        "answer": answer,
        "sources": chunks,
        "session_id": session_id,
        "admitted": True,
        "guardrails_blocked": False,
        "agt_blocked": agt_blocked,
        "agt_reason": agt_reason,
        "agent": agent,
        "trace_id": trace.trace_id,
    }