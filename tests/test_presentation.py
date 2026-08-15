from langchain_core.documents import Document

from rag.errors import ProviderUnavailableError
from rag.presentation import citation_view, query_error_message


def test_citation_view_preserves_untrusted_text_as_data():
    doc = Document(
        page_content='<img src=x onerror="alert(1)">',
        metadata={
            "source": "C:/private/cv.pdf",
            "page": 1,
            "evidence_id": "S2",
        },
    )

    view = citation_view(doc)

    assert view.label == "S2 · cv.pdf · page 2"
    assert view.excerpt == '<img src=x onerror="alert(1)">'


def test_provider_failure_returns_retryable_safe_message():
    assert query_error_message(ProviderUnavailableError()) == (
        "The AI provider is temporarily unavailable. Please try again shortly."
    )


def test_unexpected_failure_does_not_expose_exception_detail():
    message = query_error_message(RuntimeError("secret deployment detail"))

    assert message == (
        "Couldn't generate a response right now. Please try asking your question again."
    )
    assert "secret deployment detail" not in message
