"""Opt-in live diagnostic utility for testing ChromaDB vector indexing and NVIDIA endpoints."""

import argparse
import sys

from rag.config import Config
from rag.document_loader import CVDocumentLoader
from rag.engine import CVChatEngine
from rag.vector_store import VectorStoreManager


def run_diagnostics(with_llm: bool = False) -> int:
    """Executes live diagnostic checks against local documents, ChromaDB vector store, and optional LLM generation."""
    try:
        Config.validate()
        chunks = CVDocumentLoader().load_and_split()
        if not chunks:
            raise RuntimeError("No CV chunks were loaded from the documents directory")

        vector_manager = VectorStoreManager()
        count = vector_manager.ensure_documents(chunks)
        if count < 1:
            raise RuntimeError("The vector index is empty after indexing")

        results = vector_manager.similarity_search(
            "What skills and experience are listed?", k=2
        )
        if not results:
            raise RuntimeError("The retrieval smoke test returned no results")

        print("Configuration: OK")
        print(f"Loaded chunks: {len(chunks)}")
        print(f"Indexed chunks: {count}")
        print(f"Retrieval results: {len(results)}")

        if with_llm:
            response = CVChatEngine(auto_index=False).query(
                "Give a one-sentence professional summary."
            )
            if not response["answer"]:
                raise RuntimeError("The LLM returned an empty answer")
            print("LLM generation: OK")
        else:
            print("LLM generation: SKIPPED (use --with-llm to enable)")
        return 0
    except Exception as exc:
        print(f"Diagnostics failed: {type(exc).__name__}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Perform one live NVIDIA chat-generation request.",
    )
    args = parser.parse_args()
    return run_diagnostics(with_llm=args.with_llm)


if __name__ == "__main__":
    raise SystemExit(main())

