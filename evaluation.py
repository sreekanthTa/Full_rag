"""RAGAS evaluation for the RAG pipeline.

For each test case the real pipeline runs (search -> prompt -> Groq answer),
then RAGAS scores the result:
  - faithfulness      : is the answer grounded in the retrieved context?
  - answer_relevancy  : does the answer address the question?
  - context_precision : are the retrieved chunks ordered by relevance?
  - context_recall     : did retrieval find what the reference answer needs?

Run via POST /evaluate, or directly: python -c "import evaluation, json;
print(json.dumps(evaluation.run_evaluation(), indent=2))"
"""
from __future__ import annotations

import sys
import types

# ragas 0.4.3 imports ChatVertexAI from langchain_community, a symbol removed
# in langchain-community 0.4.x. ragas only uses it in an isinstance() list,
# so a stub module is enough to let the package import.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:
        pass

    _stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from datasets import Dataset
from langchain_core.embeddings import Embeddings as LangchainEmbeddings
from ragas import evaluate
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from rag_engine import _prompt, format_context, get_llm, search

# ---------------------------------------------------------------------------
# Test cases: reference answers are grounded in the ingested sample document
# "Introduction to Retrieval-Augmented Generation (RAG)".
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "question": "What is RAG?",
        "reference": (
            "Retrieval-Augmented Generation (RAG) is a technique that combines "
            "information retrieval with a large language model: the application "
            "first searches a collection of documents for relevant information, "
            "then provides the retrieved text to the model as context for "
            "generating an answer."
        ),
    },
    {
        "question": "What are the two main workflows in a basic RAG application?",
        "reference": (
            "The ingestion workflow prepares documents for search, and the "
            "question-answering workflow retrieves relevant pieces of those "
            "documents and uses them to answer a user's question."
        ),
    },
    {
        "question": "Why is chunk overlap useful?",
        "reference": (
            "Overlap helps preserve context when an important sentence falls "
            "near a chunk boundary."
        ),
    },
    {
        "question": "What does an embedding model produce?",
        "reference": (
            "An embedding model converts text into a list of numbers called a "
            "vector. Texts with similar meanings often have vectors that are "
            "close to one another in the embedding space."
        ),
    },
    {
        "question": "Why should a RAG assistant preserve page numbers?",
        "reference": (
            "Keeping the page number and source filename lets the application "
            "show where an answer came from, so a user can verify the answer "
            "against the original document."
        ),
    },
]

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

# RAGAS metrics judge with `strictness` samples per call (n=3 by default),
# but Groq only allows n=1 completions — force single-sample judgments.
for _metric in METRICS:
    if hasattr(_metric, "strictness"):
        _metric.strictness = 1

# ---------------------------------------------------------------------------
# Embeddings matching the vector store: chunks are stored with Chroma's
# default embedding function, so evaluation must use the same one.
# ---------------------------------------------------------------------------


class ChromaEmbeddings(LangchainEmbeddings):
    """LangChain embeddings adapter over Chroma's default embedding model."""

    def __init__(self) -> None:
        self._ef = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._ef(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(map(float, self._ef([text])[0]))


def _generate(question: str, chunks: list[str]) -> str:
    """Answer with the real prompt path but without persisting to memory,
    so evaluation runs don't pollute Mem0 / Postgres history."""
    context = format_context(chunks) if chunks else ""
    response = get_llm().invoke(_prompt(question, context, [], []))
    return response.content if hasattr(response, "content") else str(response)


def run_evaluation() -> dict:
    """Run the full RAGAS evaluation over TEST_CASES and return scores."""
    rows = []
    for case in TEST_CASES:
        question = case["question"]
        chunks = search(question)
        rows.append({
            "question": question,
            "contexts": chunks,
            "answer": _generate(question, chunks),
            "reference": case["reference"],
        })

    result = evaluate(
        Dataset.from_list(rows),
        metrics=METRICS,
        llm=LangchainLLMWrapper(get_llm()),
        embeddings=LangchainEmbeddingsWrapper(ChromaEmbeddings()),
        # Groq's free tier rate-limits quickly; ragas defaults to 16 parallel
        # workers, which triggers 429s. Keep concurrency low and allow time
        # for exponential backoff instead of timing jobs out.
        run_config=RunConfig(max_workers=2, timeout=300, max_wait=120),
        show_progress=False,
        raise_exceptions=True,
    )

    scores = result.scores
    if hasattr(scores, "mean"):  # pandas DataFrame
        averages = {k: float(v) for k, v in scores.mean(numeric_only=True).items()}
        per_row = scores.to_dict(orient="records")
    else:  # plain dict of lists
        averages = {k: sum(v) / len(v) for k, v in dict(scores).items() if v}
        per_row = dict(scores)

    return {
        "cases": len(rows),
        "averages": averages,
        "per_case": per_row,
        "questions": [r["question"] for r in rows],
        "answers": [r["answer"] for r in rows],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_evaluation(), indent=2, ensure_ascii=False))
