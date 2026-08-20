"use client";

import React from "react";
import { ConversationEntry, getSafeErrorMessage } from "../lib/chat/state";
import { CitationList } from "./CitationList";
import { PipelineStatus } from "./PipelineStatus";
import styles from "./ChatMessage.module.css";

export interface ChatMessageProps {
  entry: ConversationEntry;
  onRetry?: (question: string) => void;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ entry, onRetry }) => {
  const isStreaming =
    entry.status === "submitting" ||
    entry.status === "retrieving" ||
    entry.status === "reranking" ||
    entry.status === "generating" ||
    entry.status === "verifying";

  const isDegraded = entry.result?.decision === "degraded";
  const isAbstained = entry.result?.decision === "abstained";
  const isInterrupted = entry.error?.code === "stream_interrupted";

  const displayAnswer =
    entry.result?.answer || entry.provisionalAnswer || "";

  return (
    <div className={styles.messageWrapper} data-testid={`chat-turn-${entry.id}`}>
      {/* User Question */}
      <div className={styles.userBubble} data-testid="user-message">
        <div className={styles.authorHeader}>You</div>
        <div className={styles.content}>{entry.userText}</div>
      </div>

      {/* Assistant Response */}
      <div className={styles.assistantBubble} data-testid="assistant-message">
        <div className={styles.authorHeader}>CV RAG Assistant</div>

        {/* Degraded State Indicator */}
        {isDegraded && (
          <div className={styles.degradedBanner} data-testid="degraded-banner">
            <span>⚠️ Note: Response generated under degraded retrieval conditions.</span>
          </div>
        )}

        {/* Abstained Notice */}
        {isAbstained && (
          <div className={styles.abstainedBanner} data-testid="abstained-banner">
            ℹ️ The assistant abstained from answering this query because insufficient factual evidence was found in the indexed CV.
          </div>
        )}

        {/* Pipeline Stage Indicator when in progress */}
        {isStreaming && (
          <PipelineStatus
            status={entry.status}
            metadata={entry.stageMetadata}
          />
        )}

        {/* Answer Text Content */}
        {displayAnswer && (
          <div className={styles.content} data-testid="assistant-answer">
            {displayAnswer}
            {isInterrupted && (
              <div
                className={styles.incompleteNotice}
                data-testid="incomplete-notice"
              >
                [Response generation incomplete: stream was interrupted]
              </div>
            )}
          </div>
        )}

        {/* Error Presentation */}
        {entry.error && (
          <div className={styles.errorCard} data-testid="error-card">
            <div className={styles.errorTitle}>
              <span>⚠️ Error</span>
            </div>
            <div>{getSafeErrorMessage(entry.error.code, entry.error.message)}</div>
            {entry.error.retryable && onRetry && (
              <button
                type="button"
                className={styles.retryButton}
                onClick={() => onRetry(entry.userText)}
                aria-label="Retry question"
              >
                🔄 Retry
              </button>
            )}
          </div>
        )}

        {/* Authority Citations */}
        {entry.result && entry.result.citations && entry.result.citations.length > 0 && (
          <CitationList citations={entry.result.citations} />
        )}
      </div>
    </div>
  );
};
