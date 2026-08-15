import streamlit as st


def inject_custom_css():
    """Inject custom CSS styles for the modern minimal dark-mode CV RAG application."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-canvas: #090d16;
        --bg-card: #131b2e;
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
        --border-light: #1e293b;
        --accent-blue: #3b82f6;
        --user-bubble: #151d2a;
    }

    /* Global canvas & typography */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background-color: var(--bg-canvas) !important;
        color: var(--text-main) !important;
    }

    .stApp {
        background-color: var(--bg-canvas) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: var(--text-main) !important;
    }

    /* Header container transparent so sidebar toggle remains accessible */
    header[data-testid="stHeader"],
    [data-testid="stHeader"] {
        background: transparent !important;
        background-color: transparent !important;
        color: var(--text-main) !important;
        z-index: 99999 !important;
    }

    /* Hide Streamlit toolbar (action elements), deploy button & decoration ONLY */
    [data-testid="stToolbarActionElements"],
    [data-testid="stHeaderActionElements"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"],
    div[data-testid="stToast"],
    div[class*="stToast"],
    .stDeployButton {
        display: none !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        display: block !important;
        background-color: var(--bg-card) !important;
        border-right: 1px solid var(--border-light) !important;
    }

    section[data-testid="stSidebar"] .block-container,
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem !important;
        background-color: var(--bg-card) !important;
        color: var(--text-main) !important;
    }

    /* Mobile & Desktop Sidebar Toggle Controls */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarHeader"] button {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999999 !important;
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 8px !important;
    }

    [data-testid="collapsedControl"] *,
    [data-testid="stSidebarCollapsedControl"] *,
    button[data-testid="stSidebarCollapseButton"] *,
    button[data-testid="stExpandSidebarButton"] *,
    [data-testid="stSidebarHeader"] button * {
        color: var(--text-main) !important;
        fill: var(--text-main) !important;
        visibility: visible !important;
    }

    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            height: 100vh !important;
            z-index: 999998 !important;
        }
    }

    .sidebar-profile-card {
        background-color: var(--user-bubble);
        border: 1px solid var(--border-light);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }

    .sidebar-profile-name {
        font-weight: 800;
        font-size: 1.15rem;
        color: var(--text-main);
        margin-bottom: 0.25rem;
    }

    .sidebar-profile-title {
        font-size: 0.85rem;
        color: var(--accent-blue);
        font-weight: 600;
        margin-bottom: 0.75rem;
    }

    .sidebar-section-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Center main container with max-width ~860px and comfortable top padding */
    @media (min-width: 0px) {
        div[data-testid="stMainBlockContainer"],
        .stMainBlockContainer,
        div[data-testid="block-container"],
        .main .block-container {
            max-width: 860px !important;
            margin: 0 auto !important;
            padding-top: 1.25rem !important;
            margin-top: -1.25rem !important;
            padding-bottom: 6rem !important;
        }
    }

    /* Header styling - Centered */
    .portfolio-header {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 0 0 0.5rem 0 !important;
        border-bottom: 1px solid var(--border-light);
        margin-bottom: 1rem !important;
        margin-top: 0 !important;
    }

    .header-title-group {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.25rem;
    }

    .header-title-group h1 {
        font-size: 2.25rem;
        font-weight: 800;
        color: var(--text-main);
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.03em;
    }

    .header-title-group p {
        font-size: 1rem;
        color: var(--text-muted);
        margin: 0;
        font-weight: 400;
    }

    /* System Info & Disclaimer Boxes */
    .system-info-box {
        background-color: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 2rem;
        text-align: left;
    }

    .system-info-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: var(--text-main);
        margin-bottom: 0.5rem;
    }

    .disclaimer-banner {
        background-color: rgba(245, 158, 11, 0.08) !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        color: #fbbf24 !important;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.825rem;
        margin-top: 1rem;
        line-height: 1.5;
    }

    .disclaimer-banner * {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* Hero section & suggestion prompts */
    .hero-container {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        text-align: center !important;
        padding: 1.5rem 1rem;
        margin-bottom: 2rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 0.75rem;
        letter-spacing: -0.02em;
        text-align: center !important;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: var(--text-muted);
        max-width: 620px;
        margin: 0 auto;
        line-height: 1.6;
        text-align: center !important;
    }

    .suggestions-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 1rem;
        text-align: center;
    }

    /* Streamlit Button overrides (Dark Cards) */
    div.stButton > button {
        background-color: var(--bg-card) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 10px !important;
        padding: 0.85rem 1rem !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        transition: all 0.15s ease-in-out !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3) !important;
        width: 100% !important;
        text-align: center !important;
        margin-bottom: 0.5rem !important;
    }

    div.stButton > button:hover {
        border-color: var(--accent-blue) !important;
        color: var(--accent-blue) !important;
        box-shadow: 0 0 14px rgba(59, 130, 246, 0.25) !important;
        background-color: #1a263d !important;
    }

    /* Chat message styling */
    div[data-testid="stChatMessage"] {
        background-color: transparent !important;
        border-radius: 8px;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
        color: #ffffff !important;
    }

    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] span,
    div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] div,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span {
        color: #ffffff !important;
        line-height: 1.6;
    }

    /* User chat bubble styling */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background-color: var(--user-bubble) !important;
        border-radius: 12px;
        padding: 1rem 1.25rem !important;
        border: 1px solid var(--border-light);
    }

    /* Code styling */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: #0d131f !important;
        color: #e2e8f0 !important;
    }

    /* Expander & source snippet styling - Fix white background on click/hover */
    [data-testid="stExpander"],
    .stExpander,
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] details summary,
    [data-testid="stExpander"] details[open] summary,
    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:focus,
    [data-testid="stExpander"] summary:focus-visible,
    [data-testid="stExpander"] summary:active,
    [data-testid="stExpanderDetails"] {
        background-color: var(--bg-card) !important;
        color: var(--text-main) !important;
        border-color: var(--border-light) !important;
        outline: none !important;
        box-shadow: none !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--border-light) !important;
        border-radius: 8px !important;
        margin-top: 0.5rem !important;
    }

    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary svg,
    [data-testid="stExpander"] details summary p,
    [data-testid="stExpander"] details summary span,
    [data-testid="stExpander"] details summary svg {
        color: var(--text-main) !important;
        fill: var(--text-main) !important;
    }

    .source-snippet {
        background-color: var(--user-bubble);
        border-left: 3px solid var(--accent-blue);
        padding: 0.75rem 1rem;
        border-radius: 0 6px 6px 0;
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
        color: var(--text-main);
    }

    .source-snippet-header {
        font-weight: 600;
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-bottom: 0.25rem;
    }

    /* Streamlit Bottom Container & Chat Input Fixes */
    div[data-testid="stBottomBlockContainer"] {
        background-color: var(--bg-canvas) !important;
        border-top: 1px solid var(--border-light) !important;
        padding-top: 0.75rem !important;
        padding-bottom: 1rem !important;
    }

    div[data-testid="stChatInput"] {
        max-width: 860px !important;
        margin: 0 auto !important;
        background-color: transparent !important;
    }

    div[data-testid="stChatInput"] > div {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 12px !important;
    }

    div[data-testid="stChatInput"] textarea,
    div[data-testid="stChatInput"] input {
        background-color: transparent !important;
        color: var(--text-main) !important;
        font-size: 0.95rem !important;
    }

    div[data-testid="stChatInput"] input::placeholder,
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #cbd5e1 !important;
        opacity: 0.95 !important;
    }

    div[data-testid="stChatInput"] input::-webkit-input-placeholder,
    div[data-testid="stChatInput"] textarea::-webkit-input-placeholder {
        color: #cbd5e1 !important;
        opacity: 0.95 !important;
    }

    div[data-testid="stChatInput"] input:focus,
    div[data-testid="stChatInput"] textarea:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


