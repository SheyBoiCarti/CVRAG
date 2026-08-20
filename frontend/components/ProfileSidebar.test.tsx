import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ProfileSidebar } from "./ProfileSidebar";
import { PortfolioHeader } from "./PortfolioHeader";

describe("ProfileSidebar component", () => {
  it("renders candidate profile, external links, and architecture summary", () => {
    render(
      <ProfileSidebar
        hasHistory={false}
        onClearHistory={vi.fn()}
      />
    );

    expect(screen.getByText("Sheharyar Nadeem")).toBeInTheDocument();
    expect(
      screen.getByText("Software Engineer & AI Developer")
    ).toBeInTheDocument();

    const portfolioLink = screen.getByRole("link", { name: /Live Portfolio/i });
    expect(portfolioLink).toHaveAttribute("target", "_blank");
    expect(portfolioLink).toHaveAttribute("rel", "noopener noreferrer");

    const githubLink = screen.getByRole("link", { name: /GitHub Profile/i });
    expect(githubLink).toHaveAttribute("target", "_blank");
    expect(githubLink).toHaveAttribute("rel", "noopener noreferrer");

    const linkedinLink = screen.getByRole("link", { name: /LinkedIn Connect/i });
    expect(linkedinLink).toHaveAttribute("target", "_blank");
    expect(linkedinLink).toHaveAttribute("rel", "noopener noreferrer");

    expect(screen.getByText("RAG Engine Architecture")).toBeInTheDocument();
    expect(screen.getByText("meta/llama-3.3-70b-instruct")).toBeInTheDocument();
  });

  it("disables clear button when hasHistory is false, and enables when true", async () => {
    const handleClear = vi.fn();
    const user = userEvent.setup();

    const { rerender } = render(
      <ProfileSidebar
        hasHistory={false}
        onClearHistory={handleClear}
      />
    );

    const clearButton = screen.getByRole("button", {
      name: /clear conversation history/i,
    });
    expect(clearButton).toBeDisabled();

    rerender(
      <ProfileSidebar
        hasHistory={true}
        onClearHistory={handleClear}
      />
    );

    expect(clearButton).not.toBeDisabled();
    await user.click(clearButton);
    expect(handleClear).toHaveBeenCalledTimes(1);
  });
});

describe("PortfolioHeader component", () => {
  it("renders header title and accordion details", () => {
    render(<PortfolioHeader />);

    expect(
      screen.getByRole("heading", { name: "Sheharyar Nadeem", level: 1 })
    ).toBeInTheDocument();
    expect(
      screen.getByText("Software Developer & AI Engineer")
    ).toBeInTheDocument();
    expect(
      screen.getByText("How this CV RAG Assistant Works & Technical Models")
    ).toBeInTheDocument();
  });
});
