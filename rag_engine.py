"""RAG engine: PDF ingest, chunking, chunk embedding, Chroma persistence,
and retrieval-augmented answering via langchain_groq.
"""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Optional

import chromadb
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from memory import load_recent_messages, save_exchange, mem0_add, mem0_get

# ---------------------------------------------------------------------------
# Shared resources
# ---------------------------------------------------------------------------

_GUARDRAILS_DIR = Path(__file__).resolve().parent / "guardrails"

_chroma_client = None
_collection = None
_llm = None
_rails = None
_rails_ready = False


def get_client() -> chromadb.ClientAPI:
    """Lazy Chroma client persisted at settings.vector_db_path."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.vector_db_path)
    return _chroma_client


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
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
        print(f"[guardrails] initialization failed: {exc}")
        _rails = None
    return _rails


def get_guardrails() -> Optional[object]:
    """Return the guardrails engine, or None if disabled/unavailable."""
    if not settings.guardrails_enabled:
        return None
    return _init_rails()


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def add_pdf(pdf_path: str | Path) -> int:
    """Load a PDF, split into chunks, embed, and store in Chroma.

    Returns the number of chunks stored.
    """
    path = Path(pdf_path)
    loader = PyPDFLoader(str(path))
    docs = loader.load()

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
        doc_id = str(uuid.uuid4())
        ids.append(doc_id)
        documents.append(content)
        metadatas.append({
            "source": path.name,
            "chunk": i,
            # cheap content hash so re-upload of identical text can be deduped
            "hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        })

    collection = get_collection()
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(chunks)


def search(query: str, top_k: int = 4) -> list[str]:
    """Return the top_k most relevant chunk texts for a query.

    The query text is passed straight to Chroma so it embeds with the same
    model used at ingestion time (Chroma's default embedding function).
    """
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


def ask(question: str, session_id: str = "default") -> dict:
    """Run the full RAG answer flow for one question.

    Short-term context comes from the session's persisted Postgres history
    (PostgresChatMessageHistory); long-term context comes from Mem0. The new
    Q/A exchange is then saved back into both.

    NemoGuardrails input/output checks are applied when enabled.
    """
    chunks = search(question)
    context = format_context(chunks) if chunks else ""

    long_term = mem0_get(session_id)
    recent = load_recent_messages(session_id)

    messages = _prompt(question, context, long_term, recent)

    guardrails = get_guardrails()
    if guardrails is not None:
        # Register the RAG step (retrieval + memory + LLM) as a Nemo action, then
        # run the turn through Nemo so the flows in guardrails/config.yml apply
        # the input and output rails around it.
        def _rag_answer(**_kwargs) -> str:
            msgs = _prompt(question, context, long_term, recent)
            response = get_llm().invoke(msgs)
            return response.content if hasattr(response, "content") else str(response)

        guardrails.register_action(_rag_answer, name="GenerateRagAnswer")
        try:
            result = guardrails.generate(
                messages=[{"role": "user", "content": question}],
                # Enable the `self check facts` output rail and give it the
                # retrieved chunks to ground the answer against.
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
            # guardrail refusal message instead of the model output.
            save_exchange(session_id, question, answer)
            mem0_add(question, answer, session_id)
            return {
                "answer": answer,
                "sources": [],
                "session_id": session_id,
                "guardrails_blocked": True,
                "guardrails_reason": reason,
            }
    else:
        response = get_llm().invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

    # persist into both tiers of memory
    save_exchange(session_id, question, answer)
    mem0_add(question, answer, session_id)

    return {
        "answer": answer,
        "sources": chunks,
        "session_id": session_id,
        "guardrails_blocked": False,
    }