"use client";

import React, { useEffect, useRef } from "react";
import { ConversationEntry } from "../lib/chat/state";
import { ChatMessage } from "./ChatMessage";
import { SuggestionGrid } from "./SuggestionGrid";
import styles from "./ChatFeed.module.css";

export interface ChatFeedProps {
  entries: ConversationEntry[];
  onSelectSuggestion: (prompt: string) => void;
  onRetry?: (question: string) => void;
  onClearConversation?: () => void;
  isBusy?: boolean;
}

export const ChatFeed: React.FC<ChatFeedProps> = ({
  entries,
  onSelectSuggestion,
  onRetry,
  onClearConversation,
  isBusy = false,
}) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (entries.length > 0) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [entries]);

  if (entries.length === 0) {
    return (
      <div className={styles.feedContainer}>
        <SuggestionGrid
          onSelectSuggestion={onSelectSuggestion}
          disabled={isBusy}
        />
      </div>
    );
  }

  return (
    <div className={styles.feedContainer} data-testid="chat-feed">
      {onClearConversation && entries.length > 0 && (
        <div className={styles.actionRow}>
          <button
            type="button"
            className={styles.clearChatTopButton}
            onClick={onClearConversation}
            disabled={isBusy}
            aria-label="Clear chat"
            data-testid="clear-chat-feed-button"
          >
            🗑️ Clear Chat
          </button>
        </div>
      )}

      <div className={styles.messagesList}>
        {entries.map((entry) => (
          <ChatMessage
            key={entry.id}
            entry={entry}
            onRetry={onRetry}
          />
        ))}
      </div>
      <div ref={bottomRef} aria-hidden="true" />
    </div>
  );
};
