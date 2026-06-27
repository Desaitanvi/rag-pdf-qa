import streamlit as st
import tempfile
import os
from ingest import ingest_pdf
from rag_chain import get_answer_with_sources

# ── Page config ──
st.set_page_config(
    page_title="PDF Q&A Agent",
    page_icon="📄",
    layout="centered"
)

# ── Header ──
st.title("📄 RAG Document Q&A Agent")
st.markdown("Upload a PDF and ask questions about its content.")
st.divider()

# ── Session state for chat history ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_ingested" not in st.session_state:
    st.session_state.pdf_ingested = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

# ── Sidebar — PDF Upload ──
with st.sidebar:
    st.header("📁 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any PDF to start asking questions"
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("Processing PDF..."):
                # Save to temp file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Ingest PDF into ChromaDB
                ingest_pdf(tmp_path)
                os.unlink(tmp_path)  # Delete temp file

                st.session_state.pdf_ingested = True
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.messages = []  # Reset chat

            st.success(f"✅ '{uploaded_file.name}' processed!")

    if st.session_state.pdf_ingested:
        st.info(f"📄 Active: **{st.session_state.pdf_name}**")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("**Stack:**")
    st.markdown("- LangChain + LCEL")
    st.markdown("- ChromaDB (vector store)")
    st.markdown("- Groq Llama 3.3 70B")
    st.markdown("- Streamlit")

# ── Chat display ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("pages"):
            st.caption(f"📍 Sources: Pages {msg['pages']}")

# ── Chat input ──
if not st.session_state.pdf_ingested:
    st.info("👈 Please upload a PDF from the sidebar to start.")
else:
    if question := st.chat_input("Ask a question about your PDF..."):

        # Show user message
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        with st.chat_message("user"):
            st.markdown(question)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, pages, source_docs = get_answer_with_sources(question)
                    st.markdown(answer)
                    if pages:
                        st.caption(f"📍 Sources: Pages {sorted(pages)}")

                    # Show source chunks expander
                    with st.expander("View source chunks"):
                        for i, doc in enumerate(source_docs):
                            pg = doc.metadata.get('page', 0) + 1
                            st.markdown(f"**Chunk {i+1} — Page {pg}:**")
                            st.markdown(doc.page_content)
                            st.divider()

                except Exception as e:
                    answer = f"❌ Error: {str(e)}"
                    pages = []
                    st.error(answer)

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "pages": pages
        })