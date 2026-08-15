import streamlit as st


def render_header():
    """Renders centered portfolio branding, clear chat action, and architectural overview."""
    st.markdown(
        """
    <div class="portfolio-header">
        <div class="header-title-group">
            <h1>Sheharyar Nadeem</h1>
            <p>Software Developer & AI Engineer</p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("chat_history") and len(st.session_state.chat_history) > 0:
        c1, c2 = st.columns([4, 1])
        with c2:
            if st.button("Clear Chat", key="clear_chat_button", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

    with st.expander("How this CV RAG Assistant Works & Technical Models"):
        st.markdown(
            """
        **1. Document Chunking**  
        The candidate's CV document is split into semantic text chunks (`chunk_size=700`, `overlap=140`) to preserve context continuity.

        **2. Embeddings & Vector Storage**  
        Chunks are converted into dense vector embeddings using **NVIDIA Code & Text Embeddings** (`nvidia/nv-embedcode-7b-v1`) and indexed in a local **ChromaDB** vector database.

        **3. Context Retrieval & LLM Generation**  
        Top relevant CV snippets are retrieved via similarity search and passed to **Meta Llama 3.3 70B Instruct** (`meta/llama-3.3-70b-instruct`) to generate factually grounded responses.

        <div class="disclaimer-banner">
            <strong>System & Performance Disclaimer:</strong><br>
            Powered by <strong>NVIDIA AI Foundation Free Endpoints</strong>. Responses and performance may be slow, and users can experience rate limits during peak usage. If an API request times out, please wait a moment and try again.
        </div>
        """,
            unsafe_allow_html=True,
        )

