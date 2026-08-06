"""
RAG Engine Package for CV Question-Answering system.
Exposes core modules for configuration, document chunking/tokenization, vector storage, and conversational LLM engine.
"""

import sys
import site

user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)

from rag.config import Config
from rag.document_loader import CVDocumentLoader
from rag.vector_store import VectorStoreManager
from rag.engine import CVChatEngine

__all__ = ["Config", "CVDocumentLoader", "VectorStoreManager", "CVChatEngine"]
