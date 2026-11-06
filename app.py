"""Streamlit UI for the multi-document Agentic RAG system."""
import os
from pathlib import Path
from typing import Any, List

import streamlit as st

from dotenv import load_dotenv

from ingestion import PDFIngestion
from vectorstore import VectorStore
from agentic_rag import AgenticRAG

load_dotenv()

# --- Page config ---
st.set_page_config(
    page_title="Multi-Document Agentic RAG",
    page_icon="📚",
    layout="wide",
)

# --- Constants ---
ROOT_DIR = Path(__file__).resolve().parent
DOCS_DIR = ROOT_DIR / "docs"
DATA_DIR = ROOT_DIR / "data"
configured_chroma_dir = Path(
    os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma"))
)
CHROMA_DIR = (
    configured_chroma_dir
    if configured_chroma_dir.is_absolute()
    else ROOT_DIR / configured_chroma_dir
)


def ensure_dirs():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def list_pdf_files() -> List[str]:
    if not DOCS_DIR.is_dir():
        return []
    return sorted(
        path.name
        for path in DOCS_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in PDFIngestion.SUPPORTED_EXTENSIONS
    )


def process_uploads(uploaded_files: List[Any]) -> None:
    """Save supported document uploads using safe file names."""
    ensure_dirs()
    for uploaded_file in uploaded_files:
        filename = Path(uploaded_file.name).name
        if Path(filename).suffix.lower() not in PDFIngestion.SUPPORTED_EXTENSIONS:
            raise ValueError(f"{filename!r} has an unsupported file type.")
        content = uploaded_file.getbuffer()
        if not content:
            raise ValueError(f"{filename!r} is empty.")
        (DOCS_DIR / filename).write_bytes(content)


def build_index(api_key: str) -> VectorStore:
    """Ingest supported files in docs/ and build the Chroma index."""
    ingestion = PDFIngestion()
    chunks = ingestion.load_and_chunk(DOCS_DIR)
    if not chunks:
        raise ValueError(
            "No readable text was found. Scanned PDFs need OCR and are not supported yet."
        )

    vs = VectorStore(persist_dir=CHROMA_DIR, api_key=api_key)
    try:
        vs.add_documents(chunks)
    except Exception:
        vs.close()
        raise
    return vs


def get_agent() -> AgenticRAG:
    if st.session_state.agent is not None and not isinstance(
        st.session_state.agent, AgenticRAG
    ):
        st.session_state.agent = None
    if st.session_state.agent is None:
        st.session_state.agent = AgenticRAG(
            persist_dir=CHROMA_DIR,
            k=5,
            max_retries=2,
            api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
            chroma_client=(
                st.session_state.vectorstore.client
                if st.session_state.vectorstore is not None
                else None
            ),
        )
    return st.session_state.agent


def unique_sources(sources: List[dict]) -> List[dict]:
    """Remove repeated chunks from the same document page in the source list."""
    seen = set()
    unique = []
    for source in sources:
        citation = (source["source"], source["page"])
        if citation not in seen:
            seen.add(citation)
            unique.append(source)
    return unique


def format_citation(source: dict) -> str:
    location = source.get("location", f"Page {source['page']}")
    return f"**{source['source']}**, {location}"


ensure_dirs()
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = VectorStore.has_documents(
        persist_dir=CHROMA_DIR,
    )
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: Document management ---
with st.sidebar:
    st.title("📚 Multi-Document RAG")
    st.markdown("---")

    if not os.getenv("GOOGLE_API_KEY", "").strip():
        st.caption("Set GOOGLE_API_KEY in your .env file to build the index and answer questions.")

    st.markdown("---")
    st.subheader("1. Upload documents")
    uploaded = st.file_uploader(
        "Upload supported documents",
        type=sorted(extension.lstrip(".") for extension in PDFIngestion.SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
    )
    if uploaded and st.button("Save files", use_container_width=True):
        try:
            process_uploads(uploaded_files=uploaded)
            st.session_state.docs_loaded = False
            st.session_state.agent = None
            st.success(f"Saved {len(uploaded)} file(s).")
        except (OSError, ValueError) as error:
            st.error(f"Could not save PDFs: {error}")

    st.markdown("---")
    st.subheader("2. Build / Rebuild Index")
    pdfs = list_pdf_files()
    if pdfs:
        st.write(f"Supported files in docs/: {len(pdfs)}")
        for p in pdfs:
            st.caption(f"• {p}")
    else:
        st.info("No PDFs found. Upload some first.")

    if st.button("Build Index", use_container_width=True, type="primary", disabled=not pdfs):
        with st.spinner("Ingesting and indexing documents..."):
            try:
                st.session_state.vectorstore = build_index(
                    api_key=os.getenv("GOOGLE_API_KEY", "").strip()
                )
                st.session_state.docs_loaded = True
                st.session_state.agent = None  # force re-init
                st.success("Index built successfully!")
            except Exception as e:
                st.error(f"Index build failed: {e}")

    if st.session_state.docs_loaded:
        st.success("✅ Index ready")

    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# --- Main: Q&A interface ---
st.title("Ask Questions About Your Documents")
st.markdown("Answers use your uploaded documents and cite the source location.")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {format_citation(s)}")

# Chat input
question = st.chat_input("Ask a question about your documents...")
if question:
    if not st.session_state.docs_loaded:
        st.warning("Please build the index first (upload PDFs → Build Index).")
    else:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer via agentic RAG
        with st.chat_message("assistant"):
            with st.spinner("Agentic RAG is thinking..."):
                try:
                    agent = get_agent()
                    result = agent.invoke(question)
                    answer = result["answer"]
                    sources = unique_sources(result["sources"])
                    grounded = result["grounded"]

                    st.markdown(answer)

                    if not result["retrieval_used"]:
                        st.info("ℹ️ General response — no document retrieval was needed.")
                    elif grounded:
                        st.success("✅ Answer is grounded in the documents.")
                    else:
                        st.warning("⚠️ Answer may not be fully grounded.")

                    if sources:
                        with st.expander("📎 Sources"):
                            for s in sources:
                                st.markdown(f"- {format_citation(s)}")

                    with st.expander("🔍 Agent trace"):
                        st.json({
                            "rewritten_question": result["rewritten_question"],
                            "grounded": grounded,
                            "retry_count": result["retry_count"],
                            "retrieval_used": result["retrieval_used"],
                            "num_sources": len(sources),
                        })

                    # Save to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                except Exception as e:
                    st.error(f"Error: {e}")