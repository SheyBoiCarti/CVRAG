import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PipelineStatus } from "./PipelineStatus";

describe("PipelineStatus component", () => {
  it("renders nothing when status is idle or complete", () => {
    const { container: c1 } = render(<PipelineStatus status="idle" />);
    expect(c1).toBeEmptyDOMElement();

    const { container: c2 } = render(<PipelineStatus status="complete" />);
    expect(c2).toBeEmptyDOMElement();
  });

  it("shows Searching badge and status text when retrieving", () => {
    render(
      <PipelineStatus
        status="retrieving"
        metadata={{ evidenceCount: 6 }}
      />
    );

    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(
      screen.getByText("Searching CV corpus (found 6 passages)...")
    ).toBeInTheDocument();
    expect(screen.getByText("Searching")).toBeInTheDocument();
  });

  it("shows Reranking badge and candidate count when reranking", () => {
    render(
      <PipelineStatus
        status="reranking"
        metadata={{ candidateCount: 20 }}
      />
    );

    expect(
      screen.getByText("Reranking 20 candidate passages...")
    ).toBeInTheDocument();
    expect(screen.getByText("Reranking")).toBeInTheDocument();
  });

  it("shows Generating badge when generating", () => {
    render(<PipelineStatus status="generating" />);

    expect(screen.getByText("Synthesizing answer...")).toBeInTheDocument();
    expect(screen.getByText("Generating")).toBeInTheDocument();
  });

  it("shows Verifying badge when verifying", () => {
    render(
      <PipelineStatus
        status="verifying"
        metadata={{ claimCount: 4 }}
      />
    );

    expect(
      screen.getByText("Verifying 4 extracted claims against evidence...")
    ).toBeInTheDocument();
    expect(screen.getByText("Verifying")).toBeInTheDocument();
  });
});
