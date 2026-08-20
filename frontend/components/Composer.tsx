"use client";

import React, { useEffect, useRef } from "react";
import styles from "./Composer.module.css";

export interface ComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (question: string) => void;
  disabled?: boolean;
  validationError?: string | null;
  maxLength?: number;
}

export const Composer: React.FC<ComposerProps> = ({
  value,
  onChange,
  onSubmit,
  disabled = false,
  validationError,
  maxLength = 1000,
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        160
      )}px`;
    }
  }, [value]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) {
      return;
    }
    onSubmit(trimmed);
  };

  const isOverLimit = value.length > maxLength;
  const isSendDisabled = disabled || !value.trim() || isOverLimit;

  return (
    <div className={styles.composerContainer}>
      <div
        className={`${styles.composerBox} ${
          validationError || isOverLimit ? styles.hasError : ""
        }`}
      >
        <div className={styles.inputRow}>
          <textarea
            ref={textareaRef}
            className={styles.textarea}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about Sheharyar's experience, engineering skills, or projects..."
            disabled={disabled}
            rows={1}
            maxLength={maxLength + 100} // Allow slight overflow to show validation
            aria-label="Ask a question about the CV"
            aria-invalid={Boolean(validationError || isOverLimit)}
            data-testid="chat-input"
          />
          <button
            type="button"
            className={styles.sendButton}
            onClick={handleSend}
            disabled={isSendDisabled}
            aria-label="Send question"
            data-testid="send-button"
          >
            ➤
          </button>
        </div>

        <div className={styles.bottomMeta}>
          <div>
            {validationError && (
              <span className={styles.errorMessage} role="alert">
                {validationError}
              </span>
            )}
          </div>
          <div
            className={`${styles.charCount} ${
              isOverLimit ? styles.limitExceeded : ""
            }`}
          >
            {value.length} / {maxLength}
          </div>
        </div>
      </div>
    </div>
  );
};
