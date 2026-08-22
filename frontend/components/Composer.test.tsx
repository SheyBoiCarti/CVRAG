import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Composer } from "./Composer";

describe("Composer component", () => {
  it("renders textarea and submit button with placeholder", () => {
    render(
      <Composer
        value=""
        onChange={vi.fn()}
        onSubmit={vi.fn()}
      />
    );

    expect(
      screen.getByPlaceholderText(/Ask about Sheharyar's experience/i)
    ).toBeInTheDocument();
    expect(screen.getByTestId("send-button")).toBeDisabled();
  });

  it("calls onChange when typing", async () => {
    const handleChange = vi.fn();
    const user = userEvent.setup();

    render(
      <Composer
        value=""
        onChange={handleChange}
        onSubmit={vi.fn()}
      />
    );

    const input = screen.getByTestId("chat-input");
    await user.type(input, "What is his degree?");

    expect(handleChange).toHaveBeenCalled();
  });

  it("submits on Enter key press", async () => {
    const handleSubmit = vi.fn();
    const user = userEvent.setup();

    render(
      <Composer
        value="What is his degree?"
        onChange={vi.fn()}
        onSubmit={handleSubmit}
      />
    );

    const input = screen.getByTestId("chat-input");
    await user.type(input, "{Enter}");

    expect(handleSubmit).toHaveBeenCalledWith("What is his degree?");
  });

  it("does not submit on Shift+Enter", async () => {
    const handleSubmit = vi.fn();
    const user = userEvent.setup();

    render(
      <Composer
        value="Line 1"
        onChange={vi.fn()}
        onSubmit={handleSubmit}
      />
    );

    const input = screen.getByTestId("chat-input");
    await user.type(input, "{Shift>}{Enter}{/Shift}");

    expect(handleSubmit).not.toHaveBeenCalled();
  });

  it("submits on Send button click", async () => {
    const handleSubmit = vi.fn();
    const user = userEvent.setup();

    render(
      <Composer
        value="Tell me about AI projects"
        onChange={vi.fn()}
        onSubmit={handleSubmit}
      />
    );

    const button = screen.getByTestId("send-button");
    expect(button).not.toBeDisabled();
    await user.click(button);

    expect(handleSubmit).toHaveBeenCalledWith("Tell me about AI projects");
  });

  it("shows validation error and disables submit when length exceeds limit", () => {
    render(
      <Composer
        value={"a".repeat(1001)}
        onChange={vi.fn()}
        onSubmit={vi.fn()}
        validationError="Question exceeds maximum length of 1,000 characters."
      />
    );

    expect(
      screen.getByText("Question exceeds maximum length of 1,000 characters.")
    ).toBeInTheDocument();
    expect(screen.getByTestId("send-button")).toBeDisabled();
  });
});
