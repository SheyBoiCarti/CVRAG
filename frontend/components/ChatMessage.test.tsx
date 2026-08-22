import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ChatMessage } from "./ChatMessage";
import { ConversationEntry } from "../lib/chat/state";

describe("ChatMessage component", () => {
  it("renders user question and completed assistant answer", () => {
    const entry: ConversationEntry = {
      id: "entry-1",
      requestId: "req-1",
      lastSequence: 5,
      userText: "Tell me about Sheharyar's education",
      provisionalAnswer: "Graduated from university.",
      result: {
        answer: "Sheharyar holds an Honours Degree in Computer Science.",
        citations: [],
        claim_verifications: [],
        decision: "answered",
        reason_codes: [],
        retrieval_query: "education",
        pipeline_version: "dense-v1",
      },
      status: "complete",
      error: null,
      createdAt: Date.now(),
    };

    render(<ChatMessage entry={entry} />);

    expect(
      screen.getByText("Tell me about Sheharyar's education")
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "Sheharyar holds an Honours Degree in Computer Science."
      )
    ).toBeInTheDocument();
  });

  it("renders degraded banner when decision is degraded", () => {
    const entry: ConversationEntry = {
      id: "entry-2",
      requestId: "req-2",
      lastSequence: 5,
      userText: "What skills does he have?",
      provisionalAnswer: "",
      result: {
        answer: "Skills include Python and TypeScript.",
        citations: [],
        claim_verifications: [],
        decision: "degraded",
        reason_codes: ["dense_unavailable"],
        retrieval_query: "skills",
        pipeline_version: "dense-v1",
      },
      status: "complete",
      error: null,
      createdAt: Date.now(),
    };

    render(<ChatMessage entry={entry} />);

    expect(screen.getByTestId("degraded-banner")).toBeInTheDocument();
    expect(
      screen.getByText(/Response generated under degraded retrieval conditions/i)
    ).toBeInTheDocument();
  });

  it("renders abstained banner when decision is abstained", () => {
    const entry: ConversationEntry = {
      id: "entry-3",
      requestId: "req-3",
      lastSequence: 5,
      userText: "What is his favorite video game?",
      provisionalAnswer: "",
      result: {
        answer: "I do not have evidence regarding favorite video games.",
        citations: [],
        claim_verifications: [],
        decision: "abstained",
        reason_codes: ["insufficient_evidence"],
        retrieval_query: "favorite video game",
        pipeline_version: "dense-v1",
      },
      status: "complete",
      error: null,
      createdAt: Date.now(),
    };

    render(<ChatMessage entry={entry} />);

    expect(screen.getByTestId("abstained-banner")).toBeInTheDocument();
    expect(
      screen.getByText(
        /The assistant abstained from answering this query because insufficient factual evidence/i
      )
    ).toBeInTheDocument();
  });

  it("renders incomplete notice and retry button when stream is interrupted", async () => {
    const handleRetry = vi.fn();
    const user = userEvent.setup();

    const entry: ConversationEntry = {
      id: "entry-4",
      requestId: "req-4",
      lastSequence: 2,
      userText: "Tell me about his backend experience",
      provisionalAnswer: "He has experience building FastAPI and...",
      result: null,
      status: "error",
      error: {
        code: "stream_interrupted",
        message: "The stream was interrupted.",
        retryable: true,
        partialAnswer: "He has experience building FastAPI and...",
      },
      createdAt: Date.now(),
    };

    render(<ChatMessage entry={entry} onRetry={handleRetry} />);

    const notice = screen.getByTestId("incomplete-notice");
    expect(notice).toHaveTextContent(
      "[Response generation incomplete: stream was interrupted]"
    );

    const errorCard = screen.getByTestId("error-card");
    expect(errorCard).toHaveTextContent(
      "The response stream was interrupted before completion. You can retry the question."
    );

    const retryBtn = screen.getByRole("button", { name: /retry question/i });
    expect(retryBtn).toBeInTheDocument();

    await user.click(retryBtn);
    expect(handleRetry).toHaveBeenCalledWith(
      "Tell me about his backend experience"
    );
  });
});
