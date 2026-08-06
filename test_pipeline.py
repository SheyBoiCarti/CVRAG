import sys
import site
from pathlib import Path

# Ensure UTF-8 output encoding for Windows terminal compatibility
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.config import Config
from rag.document_loader import CVDocumentLoader
from rag.vector_store import VectorStoreManager
from rag.engine import CVChatEngine

def run_diagnostics():
    """
    Test suite for validating each component of the CV RAG Pipeline.
    """
    print("=" * 60)
    print("Running RAG Pipeline Component Diagnostics")
    print("=" * 60)

    # 1. Config Validation
    print("\n1. Checking Environment & Config...")
    try:
        Config.validate()
        key_str = Config.NVIDIA_API_KEY or ""
        masked_key = key_str[:6] + "..." + key_str[-4:] if len(key_str) > 10 else "***"
        print(f"   [OK] API Key detected: {masked_key}")
        print(f"   [OK] LLM Model: {Config.LLM_MODEL}")
        print(f"   [OK] Embedding Model: {Config.EMBEDDING_MODEL}")
        print(f"   [OK] Documents Dir: {Config.DOCUMENTS_DIR}")
    except Exception as e:
        print(f"   [FAIL] Config error: {e}")
        return

    # 2. Document Loader & Tokenizer / Chunker Test
    print("\n2. Testing Document Loader & Tokenizer...")
    loader = CVDocumentLoader()
    raw_docs = loader.load_documents()
    print(f"   [OK] Loaded {len(raw_docs)} raw document page(s).")
    
    chunks = loader.split_documents(raw_docs)
    print(f"   [OK] Chunking created {len(chunks)} text chunks.")
    if chunks:
        print(f"   Sample Chunk #1 snippet:\n   \"{chunks[0].page_content[:150]}...\"")

    # 3. Vector Database Test
    print("\n3. Testing Vector Database & NVIDIA Embeddings...")
    vector_mgr = VectorStoreManager()
    vector_mgr.build_vector_store(chunks)
    count = vector_mgr.get_document_count()
    print(f"   [OK] ChromaDB holds {count} indexed vector chunks.")

    # 4. Similarity Search Test
    print("\n4. Testing Similarity Search...")
    test_query = "What technical skills, programming languages, or degree does the candidate have?"
    results = vector_mgr.similarity_search_with_scores(test_query, k=2)
    print(f"   Query: '{test_query}'")
    for doc, score in results:
        print(f"   - Match Score: {score:.4f} | Snippet: {doc.page_content[:120]}...")

    # 5. End-to-End RAG Engine Test
    print("\n5. Testing End-to-End RAG Answer Generation...")
    engine = CVChatEngine(auto_index=False)
    output = engine.query("Give me a quick 2-sentence summary of the candidate.")
    print("\n   LLM Answer:")
    print(f"   {output['answer']}")
    print(f"\n   Cited Sources: {output['sources']}")

    print("\n" + "=" * 60)
    print("Diagnostics complete! All pipeline stages operating normally.")
    print("=" * 60)

if __name__ == "__main__":
    run_diagnostics()
