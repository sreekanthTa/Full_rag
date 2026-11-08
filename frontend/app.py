"""Streamlit chat UI for the RAG assistant backend.

Run backend first:  uvicorn main:app --reload
Run this:           streamlit run frontend/app.py
"""
from __future__ import annotations

import uuid

import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Assistant", page_icon="📄", layout="centered")
st.title("RAG Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
session_id = st.session_state.session_id


def load_history() -> None:
    """Fetch previous session messages from the backend once per page load."""
    try:
        resp = requests.get(
            f"{BACKEND_URL}/history",
            params={"session_id": session_id},
            timeout=30,
        )
        resp.raise_for_status()
        for item in resp.json().get("history", []):
            role = "user" if item["role"] == "user" else "assistant"
            with st.chat_message(role):
                st.markdown(item["content"])
            st.session_state.messages.append(
                {"role": role, "content": item["content"]}
            )
        st.session_state.history_loaded = True
    except requests.RequestException as exc:
        st.error(f"Could not load history: {exc}")


if "messages" not in st.session_state:
    st.session_state.messages = []
if not st.session_state.get("history_loaded"):
    load_history()

# -------------------------------------------------------------------------
# Sidebar: PDF upload + memory controls
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded and st.button("Index PDF"):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/upload",
                files={"file": (uploaded.name, uploaded.getvalue(),
                                "application/pdf")},
                timeout=120,
            )
            if resp.ok:
                st.success(resp.json().get("message", "Uploaded"))
            else:
                st.error(resp.json().get("detail", resp.text))
        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")

    st.header("Memory")
    if st.button("Clear memory"):
        try:
            requests.delete(
                f"{BACKEND_URL}/memory",
                params={"session_id": session_id},
                timeout=10,
            )
            st.session_state.messages = []
            st.session_state.history_loaded = True
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Clear failed: {exc}")

    st.caption(f"session: `{session_id[:12]}`")

# -------------------------------------------------------------------------
# Chat
# -------------------------------------------------------------------------
question = st.chat_input("Ask a question about your documents")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/ask",
                json={"question": question, "session_id": session_id},
                timeout=120,
            )
            resp.raise_for_status()
            answer = resp.json().get("answer", "")
        except requests.RequestException as exc:
            answer = f"Request failed: {exc}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
