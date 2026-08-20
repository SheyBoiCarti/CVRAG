"use client";

import React, { useCallback, useReducer, useRef, useState } from "react";
import { streamChatMessage } from "../lib/api/chat";
import {
  chatReducer,
  getCompletedHistory,
  INITIAL_STATE,
} from "../lib/chat/state";
import { ChatFeed } from "../components/ChatFeed";
import { Composer } from "../components/Composer";
import { PortfolioHeader } from "../components/PortfolioHeader";
import { ProfileSidebar } from "../components/ProfileSidebar";
import styles from "./page.module.css";

export default function HomePage() {
  const [state, dispatch] = useReducer(chatReducer, INITIAL_STATE);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const isBusy =
    state.globalStatus !== "idle" && state.globalStatus !== "error";

  const handleClearConversation = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    dispatch({ type: "CLEAR_CONVERSATION" });
  }, []);

  const handleSubmitQuestion = useCallback(
    async (questionText: string) => {
      const normalized = questionText.trim();
      if (!normalized) {
        dispatch({
          type: "SET_VALIDATION_ERROR",
          payload: "Question cannot be empty or whitespace-only.",
        });
        return;
      }

      if (normalized.length > 1000) {
        dispatch({
          type: "SET_VALIDATION_ERROR",
          payload: "Question exceeds maximum length of 1,000 characters.",
        });
        return;
      }

      // Abort any existing in-flight request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      const entryId = `entry-${Date.now()}-${Math.random()
        .toString(36)
        .substring(2, 8)}`;

      dispatch({
        type: "START_SUBMISSION",
        payload: {
          entryId,
          userText: normalized,
        },
      });

      const history = getCompletedHistory(state.entries);

      await streamChatMessage({
        question: normalized,
        history,
        signal: abortController.signal,
        onEvent: (event) => {
          dispatch({
            type: "RECEIVE_SSE_EVENT",
            payload: {
              entryId,
              requestId: event.request_id,
              event,
            },
          });
        },
        onError: (error) => {
          dispatch({
            type: "FAIL_REQUEST",
            payload: {
              entryId,
              code: error.code,
              message: error.message,
              retryable: error.retryable,
            },
          });
        },
      });

      if (abortControllerRef.current === abortController) {
        abortControllerRef.current = null;
      }
    },
    [state.entries]
  );

  return (
    <div className={styles.layoutContainer}>
      <ProfileSidebar
        hasHistory={state.entries.length > 0}
        onClearHistory={handleClearConversation}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <main className={styles.mainContent}>
        <PortfolioHeader
          onToggleSidebar={() => setSidebarOpen((prev) => !prev)}
        />

        <div className={styles.contentWrapper}>
          <ChatFeed
            entries={state.entries}
            onSelectSuggestion={handleSubmitQuestion}
            onRetry={handleSubmitQuestion}
            onClearConversation={handleClearConversation}
            isBusy={isBusy}
          />
        </div>

        <Composer
          value={state.inputValue}
          onChange={(val) => dispatch({ type: "SET_INPUT", payload: val })}
          onSubmit={handleSubmitQuestion}
          disabled={isBusy}
          validationError={state.validationError}
        />
      </main>
    </div>
  );
}
