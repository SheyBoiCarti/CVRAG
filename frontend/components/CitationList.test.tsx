import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CitationList } from "./CitationList";
import { Citation } from "../lib/api/types";

describe("CitationList component", () => {
  it("renders nothing when citation array is empty", () => {
    const { container } = render(<CitationList citations={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders structured citations with source, page, and status badges", () => {
    const citations: Citation[] = [
      {
        evidence_id: "E1",
        chunk_id: "chk-1",
        source: "sheharyar_cv.md",
        page: 1,
        excerpt: "Graduated with Honours Computer Science (Cum Laude).",
        support_status: "supported",
      },
      {
        evidence_id: "E2",
        chunk_id: "chk-2",
        source: "project_architecture.pdf",
        page: 3,
        excerpt: "Implemented distributed RAG with ChromaDB.",
        support_status: "unsupported",
      },
      {
        evidence_id: "E3",
        chunk_id: "chk-3",
        source: "notes.txt",
        page: 2,
        excerpt: "General notes on deep learning.",
        support_status: "not_checked",
      },
    ];

    render(<CitationList citations={citations} />);

    expect(
      screen.getByText(/Grounded RAG Context/i)
    ).toBeInTheDocument();

    expect(screen.getByText("Citation [E1]")).toBeInTheDocument();
    expect(screen.getByText("Supported")).toBeInTheDocument();
    expect(screen.getByText("Source: sheharyar_cv.md (Page 1)")).toBeInTheDocument();
    expect(
      screen.getByText("Graduated with Honours Computer Science (Cum Laude).")
    ).toBeInTheDocument();

    expect(screen.getByText("Citation [E2]")).toBeInTheDocument();
    expect(screen.getByText("Unsupported")).toBeInTheDocument();

    expect(screen.getByText("Citation [E3]")).toBeInTheDocument();
    expect(screen.getByText("Unchecked")).toBeInTheDocument();
  });

  it("safely renders HTML-like citation text as literal text and never creates DOM elements", () => {
    const maliciousExcerpt =
      '<img src="x" onerror="alert(1)"> <script>window.__hacked=true;</script> <b>Bold Attempt</b>';

    const citations: Citation[] = [
      {
        evidence_id: "E1",
        chunk_id: "chk-xss",
        source: "cv.md",
        page: 1,
        excerpt: maliciousExcerpt,
        support_status: "supported",
      },
    ];

    const { container } = render(<CitationList citations={citations} />);

    // Must not contain any <img> or <script> DOM elements
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("script")).toBeNull();

    // The text node itself contains the literal characters
    const excerptEl = screen.getByTestId("citation-excerpt");
    expect(excerptEl.textContent).toBe(maliciousExcerpt);
  });
});
