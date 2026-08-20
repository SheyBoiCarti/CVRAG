
import streamlit as st


def render_hero() -> str | None:
    """Renders the hero introduction and clickable prompt chips when conversation is empty."""
    st.markdown(
        """
    <div class="hero-container" style="text-align: center; display: flex; flex-direction: column; align-items: center;">
        <h2 class="hero-title" style="text-align: center;">Ask about my experience, skills & background</h2>
        <p class="hero-subtitle" style="text-align: center; margin: 0 auto; max-width: 620px;">Explore my software engineering journey, technical projects, education, and professional background through an AI assistant powered by a Retrieval-Augmented Generation (RAG) pipeline indexing my CV.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="suggestions-label" style="text-align: center; margin-top: 1rem;">Suggested Questions</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    selected_prompt = None

    with col1:
        if st.button("Professional Summary", use_container_width=True):
            selected_prompt = "Give me a summary of Sheharyar's professional experience and key strengths."

    with col2:
        if st.button("Technical Skills", use_container_width=True):
            selected_prompt = "What technical skills, programming languages, and tools does Sheharyar possess?"

    with col3:
        if st.button("Work Experience", use_container_width=True):
            selected_prompt = "What are Sheharyar's key employment roles, engineering experience, and achievements?"

    with col4:
        if st.button("AI & Machine Learning", use_container_width=True):
            selected_prompt = "Tell me about Sheharyar's AI, Machine Learning, and RAG project experience."

    with col5:
        if st.button("Education & Degrees", use_container_width=True):
            selected_prompt = "What education, academic degrees, or certifications does Sheharyar hold?"

    with col6:
        if st.button("Backend & Architecture", use_container_width=True):
            selected_prompt = "What experience does Sheharyar have in backend systems and software development?"

    return selected_prompt

