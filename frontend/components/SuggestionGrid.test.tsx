import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SUGGESTIONS, SuggestionGrid } from "./SuggestionGrid";

describe("SuggestionGrid component", () => {
  it("renders the hero intro and all 6 suggested questions", () => {
    render(<SuggestionGrid onSelectSuggestion={vi.fn()} />);

    expect(
      screen.getByText("Ask about my experience, skills & background")
    ).toBeInTheDocument();
    expect(screen.getByText("Suggested Questions")).toBeInTheDocument();

    expect(SUGGESTIONS.length).toBe(6);
    for (const suggestion of SUGGESTIONS) {
      expect(screen.getByText(suggestion.label)).toBeInTheDocument();
    }
  });

  it("submits the exact intended question when a suggestion button is clicked", async () => {
    const handleSelect = vi.fn();
    const user = userEvent.setup();

    render(<SuggestionGrid onSelectSuggestion={handleSelect} />);

    const summaryButton = screen.getByText("Professional Summary");
    await user.click(summaryButton);

    expect(handleSelect).toHaveBeenCalledTimes(1);
    expect(handleSelect).toHaveBeenCalledWith(
      "Give me a summary of Sheharyar's professional experience and key strengths."
    );

    const mlButton = screen.getByText("AI & Machine Learning");
    await user.click(mlButton);
    expect(handleSelect).toHaveBeenCalledWith(
      "Tell me about Sheharyar's AI, Machine Learning, and RAG project experience."
    );
  });

  it("disables all buttons when disabled is true", async () => {
    const handleSelect = vi.fn();
    const user = userEvent.setup();

    render(<SuggestionGrid onSelectSuggestion={handleSelect} disabled={true} />);

    const button = screen.getByText("Professional Summary");
    expect(button).toBeDisabled();

    await user.click(button);
    expect(handleSelect).not.toHaveBeenCalled();
  });
});
