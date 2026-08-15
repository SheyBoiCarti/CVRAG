import streamlit as st

from rag.config import Config


def render_sidebar():
    """Renders candidate profile, quick links, system diagnostics, and session controls in the sidebar."""
    with st.sidebar:
        st.markdown(
            """
        <div class="sidebar-profile-card">
            <div class="sidebar-profile-name">Sheharyar Nadeem</div>
            <div class="sidebar-profile-title">Software Engineer & AI Developer</div>
            <div style="font-size: 0.825rem; color: var(--text-muted); line-height: 1.5;">
                🎓 <strong>Honours Computer Science</strong> (Cum Laude)<br>
                📍 <strong>Location</strong>: Open to Remote & Relocation
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sidebar-section-title">Quick Links</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
        - 🌐 [Live Portfolio](https://sheharyar-nadeem.vercel.app/)
        - 🐙 [GitHub Profile](https://github.com/SheyBoiCarti)
        - 💼 [LinkedIn Connect](https://linkedin.com/in/sheharyar-nadeem/)
        """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.markdown(
            '<div class="sidebar-section-title">Session Controls</div>',
            unsafe_allow_html=True,
        )
        has_history = bool(st.session_state.get("chat_history"))
        if st.button(
            "Clear Conversation",
            key="sidebar_clear_btn",
            use_container_width=True,
            disabled=not has_history,
        ):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("---")

        st.markdown(
            '<div class="sidebar-section-title">RAG Engine Architecture</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
        - **LLM Backbone**: `{Config.LLM_MODEL}`
        - **Embeddings**: `{Config.EMBEDDING_MODEL}`
        - **Vector Storage**: `ChromaDB` (Persistent)
        """,
            unsafe_allow_html=True,
        )

