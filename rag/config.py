import os
import sys
import site
from pathlib import Path
from dotenv import load_dotenv

# Ensure user-installed site-packages are on sys.path
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration manager for the CV RAG Pipeline.
    Encapsulates API keys, model selections, path settings, and chunking hyperparameters.
    """

    # Base project directory
    BASE_DIR = Path(__file__).resolve().parent.parent

    # NVIDIA AI Foundation Endpoint API Key
    # Checks both NVIDIA_API_KEY and api_key environment variables
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("api_key")

    # LLM Model Name for generating answers
    # Uses Meta Llama 3.3 70B Instruct model via NVIDIA Endpoints
    LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.3-70b-instruct")

    # Embedding Model Name for converting text chunks into dense vector representations
    # Uses NVIDIA Code & Text Embeddings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedcode-7b-v1")

    # Path to directory containing target documents (CV PDF, TXT, MD)
    DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", BASE_DIR / "documents"))

    # Path to store the persistent Chroma vector database
    VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_DIR", BASE_DIR / "vector_db"))

    # Text Splitting & Tokenization Parameters
    # CHUNK_SIZE: Maximum characters per chunk (represents token budget per vector snippet)
    CHUNK_SIZE = 700
    # CHUNK_OVERLAP: Character overlap between consecutive chunks to preserve contextual continuity
    CHUNK_OVERLAP = 140
    # RETRIEVAL_K: Default number of top matching document chunks to retrieve per query
    RETRIEVAL_K = 6

    # API Network Resilience Parameters
    # REQUEST_TIMEOUT: HTTP read timeout in seconds to prevent ReadTimeout exceptions during peak loads
    REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))
    # MAX_RETRIES: Number of automatic retries for transient NVIDIA API network timeouts
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
    # EMBEDDING_BATCH_SIZE: Max document chunks sent per embedding request to prevent API timeouts
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "15"))

    @classmethod
    def validate(cls):
        """
        Validates that necessary configuration settings (like API key) are present.
        Raises ValueError if API key is missing.
        """
        if not cls.NVIDIA_API_KEY:
            raise ValueError(
                "NVIDIA API Key not found! Please set NVIDIA_API_KEY or api_key in your .env file."
            )
        
        # Ensure directories exist
        cls.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
