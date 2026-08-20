import os
from typing import Any

import streamlit as st


def get_human_source_label(sources: list[str]) -> str:
    """Formats raw source filenames into clean, human-readable citation expander labels."""
    if not sources:
        return "Grounded RAG Context (CV Document)"
    labels = []
    for s in sources:
        base = os.path.basename(s).lower()
        if "cv" in base or "resume" in base:
            labels.append("CV Document")
        else:
            labels.append(s)
    return f"Grounded RAG Context ({', '.join(sorted(set(labels)))})"


def render_chat_history(chat_history: list[tuple[Any, ...]]):
    """Renders the prior conversation history and reference citation cards."""
    for item in chat_history:
        user_msg = item[0]
        ai_msg = item[1]
        sources = item[2] if len(item) > 2 else []
        docs = item[3] if len(item) > 3 else []

        with st.chat_message("user"):
            st.write(user_msg)

        with st.chat_message("assistant"):
            st.write(ai_msg)
            if sources:
                label = get_human_source_label(sources)
                with st.expander(label):
                    for i, doc in enumerate(docs, 1):
                        content = getattr(doc, "page_content", getattr(doc, "excerpt", str(doc)))
                        ev_id = getattr(doc, "evidence_id", f"S{i}")
                        status = getattr(doc, "support_status", "")
                        status_badge = f" — [{status}]" if status else ""
                        st.markdown(
                            f"<div class='source-snippet'><div class='source-snippet-header'>Citation [{ev_id}]{status_badge}</div>{content}</div>",
                            unsafe_allow_html=True,
                        )

