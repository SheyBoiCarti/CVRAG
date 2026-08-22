class RAGError(Exception):
    """Base error for expected RAG failures."""


class ProviderUnavailableError(RAGError):
    """The upstream AI provider is temporarily unavailable."""


class StreamInterruptedError(ProviderUnavailableError):
    """A provider stream stopped after output had started."""
