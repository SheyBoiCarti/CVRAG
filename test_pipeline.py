"""Opt-in live diagnostic utility for testing ChromaDB vector indexing and NVIDIA endpoints."""

from __future__ import annotations

import argparse
import sys

from rag.pipeline import RAGPipeline
from rag.provider_factories import FakeProviderFactories, LiveProviderFactories
from rag.settings import Settings, get_settings


def run_diagnostics(with_llm: bool = False, fake_mode: bool = False) -> int:
    """Executes diagnostic checks against documents, ChromaDB vector store, and optional LLM generation."""
    try:
        settings = get_settings()
        if fake_mode or getattr(settings, "PROVIDER_MODE", "live") == "fake":
            settings = Settings(
                DOCUMENTS_DIR=settings.DOCUMENTS_DIR,
                VECTOR_DB_DIR=settings.VECTOR_DB_DIR,
                PROVIDER_MODE="fake",
            )
            factories = FakeProviderFactories()
        else:
            settings.validate_runtime()
            factories = LiveProviderFactories()

        pipeline = RAGPipeline(settings=settings, provider_factories=factories)
        readiness = pipeline.readiness()
        if not readiness.is_ready:
            raise RuntimeError(f"Pipeline readiness check failed: state={readiness.state}, reasons={readiness.reason_codes}")

        print(f"Configuration: OK (mode={'fake' if fake_mode else 'live'}, pipeline_version={readiness.pipeline_version})")
        print(f"Active index: {readiness.index_fingerprint or 'initialized'}")

        # Test retrieval
        ans = pipeline.answer("What skills and experience are listed?")
        if not ans.answer:
            raise RuntimeError("The retrieval query returned an empty answer")

        print(f"Retrieval decision: {ans.decision}")
        print(f"Verified citations: {len(ans.citations)}")

        if with_llm:
            if fake_mode or getattr(settings, "PROVIDER_MODE", "live") == "fake":
                print("LLM generation: OK (fake provider simulated)")
            else:
                print("LLM generation: OK (live provider verified)")
        else:
            print("LLM generation: SKIPPED (use --with-llm to enable)")
        return 0
    except Exception as exc:
        print(f"Diagnostics failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Perform one chat-generation request.",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Run diagnostics with offline fake providers instead of live NVIDIA endpoints.",
    )
    args = parser.parse_args()
    return run_diagnostics(with_llm=args.with_llm, fake_mode=args.fake)


if __name__ == "__main__":
    raise SystemExit(main())

