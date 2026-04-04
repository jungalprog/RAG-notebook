from typing import Dict, Any, List

import streamlit as st
import os

from backend.core import run_llm
from ingestion import run_ingestion


def _format_sources(context_docs: List[any]) -> List[str]:
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]


st.set_page_config(page_title="RAG Notebook", layout="centered")
st.title("RAG Notebook")

# --- Initialize session state ---
if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = True
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about the document"}
    ]

# --- Sidebar ---
with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

# --- Upload Section (only shown if no file uploaded yet) ---
if st.session_state.show_uploader:
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["txt", "csv", "pdf", "png", "jpg", "jpeg"],
        help="Upload a file to send to the vector store"
    )

    if uploaded_file is not None:
        st.write(f"- **Name:** {uploaded_file.name}")
        st.write(f"- **Type:** {uploaded_file.type}")

        if st.button("⬆️ Upload to Vector Store"):
            TEMP_DIR = "temp_uploads"
            os.makedirs(TEMP_DIR, exist_ok=True)
            FILE_PATH = os.path.join(TEMP_DIR, uploaded_file.name)

            with open(FILE_PATH, "wb") as f:
                f.write(uploaded_file.read())

            with st.spinner("Uploading to vector store..."):
                run_ingestion(FILE_PATH)

            # Mark upload as complete
            st.session_state.selected_doc = uploaded_file.name
            st.session_state.show_uploader = False
            st.rerun()
    else:
        st.info("👆 Please upload a file to get started.")

# --- Chat Section (only shown after successful upload) ---
if not st.session_state.show_uploader and st.session_state.selected_doc:
    st.header(f"📄 {st.session_state.selected_doc}")
    st.success("✅ File successfully uploaded to vector store!")

    if st.button("➕ Upload another file"):
        st.session_state.show_uploader = True
        st.session_state.selected_doc = None
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me anything about the document"}
        ]
        st.rerun()

    st.divider()

    # --- Messages ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- {s}")

    # --- Chat Input ---
    prompt = st.chat_input(
        f"Ask a question about {st.session_state.selected_doc}")
    if prompt:
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "sources": []})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving docs and generating answer..."):
                    result: Dict[str, Any] = run_llm(prompt)
                    answer = str(result.get("answer", "")
                                 ).strip() or "(No answer returned.)"
                    sources = _format_sources(result.get("context", []))

                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.markdown(f"- {s}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error("Failed to generate a response")
                st.exception(e)
