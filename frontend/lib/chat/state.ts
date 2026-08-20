import {
  ChatResult,
  ChatTurn,
  PipelineSSEEvent,
} from "../api/types";

export type PipelineStage =
  | "idle"
  | "submitting"
  | "retrieving"
  | "reranking"
  | "generating"
  | "verifying"
  | "complete"
  | "error";

export interface StageMetadata {
  evidenceCount?: number;
  candidateCount?: number;
  claimCount?: number | null;
  degradedReasonCodes?: string[];
}

export interface ConversationError {
  code: string;
  message: string;
  retryable: boolean;
  partialAnswer?: string | null;
}

export interface ConversationEntry {
  id: string; // Stable client generation ID
  requestId: string | null; // Active backend request ID
  lastSequence: number; // Monotonic sequence tracking
  userText: string;
  provisionalAnswer: string;
  result: ChatResult | null;
  status: PipelineStage;
  stageMetadata?: StageMetadata;
  error: ConversationError | null;
  createdAt: number;
}

export interface ChatState {
  entries: ConversationEntry[];
  activeEntryId: string | null;
  globalStatus: PipelineStage;
  inputValue: string;
  validationError: string | null;
}

export const INITIAL_STATE: ChatState = {
  entries: [],
  activeEntryId: null,
  globalStatus: "idle",
  inputValue: "",
  validationError: null,
};

export type ChatAction =
  | {
      type: "START_SUBMISSION";
      payload: {
        entryId: string;
        userText: string;
      };
    }
  | {
      type: "SET_REQUEST_ID";
      payload: {
        entryId: string;
        requestId: string;
      };
    }
  | {
      type: "RECEIVE_SSE_EVENT";
      payload: {
        entryId: string;
        requestId: string;
        event: PipelineSSEEvent;
      };
    }
  | {
      type: "FAIL_REQUEST";
      payload: {
        entryId: string;
        requestId?: string | null;
        code: string;
        message: string;
        retryable: boolean;
        partialAnswer?: string | null;
      };
    }
  | {
      type: "CLEAR_CONVERSATION";
    }
  | {
      type: "SET_INPUT";
      payload: string;
    }
  | {
      type: "SET_VALIDATION_ERROR";
      payload: string | null;
    };

/**
 * Maps error codes to safe, user-friendly UI messages according to Spec 06 §9.
 */
export function getSafeErrorMessage(code: string, fallbackMessage?: string): string {
  switch (code) {
    case "invalid_question":
      return (
        fallbackMessage ||
        "Please provide a valid question (up to 1,000 characters)."
      );
    case "invalid_history":
      return "Conversational history format is invalid.";
    case "request_too_large":
      return "The request payload exceeded the maximum permitted size.";
    case "provider_unavailable":
      return "The upstream AI provider is temporarily unavailable. Please retry shortly.";
    case "pipeline_not_ready":
      return "The RAG pipeline is still initializing or temporarily offline. Please wait a moment and retry.";
    case "capacity_exceeded":
      return "System capacity is currently reached. Please retry in a few moments.";
    case "stream_interrupted":
      return "The response stream was interrupted before completion. You can retry the question.";
    case "internal_error":
    case "rag_error":
    default:
      return "An unexpected error occurred while processing your request. Please try again.";
  }
}

/**
 * Returns only completed turns for conversational history submission.
 * Bounds history to the most recent 20 turns.
 */
export function getCompletedHistory(
  entries: ConversationEntry[],
  maxTurns: number = 20
): ChatTurn[] {
  const completed = entries
    .filter(
      (entry) =>
        entry.status === "complete" &&
        entry.result !== null &&
        entry.userText.trim().length > 0
    )
    .map((entry) => ({
      user: entry.userText,
      assistant: entry.result?.answer || entry.provisionalAnswer,
    }));

  return completed.slice(-maxTurns);
}

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "START_SUBMISSION": {
      const newEntry: ConversationEntry = {
        id: action.payload.entryId,
        requestId: null,
        lastSequence: 0,
        userText: action.payload.userText,
        provisionalAnswer: "",
        result: null,
        status: "submitting",
        stageMetadata: {},
        error: null,
        createdAt: Date.now(),
      };
      return {
        ...state,
        entries: [...state.entries, newEntry],
        activeEntryId: action.payload.entryId,
        globalStatus: "submitting",
        inputValue: "",
        validationError: null,
      };
    }

    case "SET_REQUEST_ID": {
      if (state.activeEntryId !== action.payload.entryId) {
        return state;
      }
      return {
        ...state,
        entries: state.entries.map((entry) =>
          entry.id === action.payload.entryId
            ? { ...entry, requestId: action.payload.requestId }
            : entry
        ),
      };
    }

    case "RECEIVE_SSE_EVENT": {
      const { entryId, requestId, event } = action.payload;

      // Ignore event if this is not the active entry
      if (state.activeEntryId !== entryId) {
        return state;
      }

      const activeEntry = state.entries.find((e) => e.id === entryId);
      if (!activeEntry) {
        return state;
      }

      // Ignore event if request ID does not match active request ID (when established)
      if (
        activeEntry.requestId &&
        activeEntry.requestId !== requestId &&
        event.request_id !== activeEntry.requestId
      ) {
        return state;
      }

      // Enforce monotonic sequence numbers
      if (event.sequence <= activeEntry.lastSequence) {
        return state;
      }

      let updatedStatus: PipelineStage = activeEntry.status;
      let updatedProvisional = activeEntry.provisionalAnswer;
      let updatedResult = activeEntry.result;
      let updatedError: ConversationError | null = activeEntry.error;
      const updatedMetadata: StageMetadata = { ...activeEntry.stageMetadata };

      switch (event.event) {
        case "retrieval_started":
          updatedStatus = "retrieving";
          break;

        case "retrieval_completed":
          updatedMetadata.evidenceCount = event.data.evidence_count;
          if (event.data.degraded_reason_codes) {
            updatedMetadata.degradedReasonCodes = event.data.degraded_reason_codes;
          }
          break;

        case "reranking_started":
          updatedStatus = "reranking";
          updatedMetadata.candidateCount = event.data.candidate_count;
          break;

        case "reranking_completed":
          updatedMetadata.evidenceCount = event.data.evidence_count;
          if (event.data.degraded_reason_codes) {
            updatedMetadata.degradedReasonCodes = event.data.degraded_reason_codes;
          }
          break;

        case "generation_started":
          updatedStatus = "generating";
          break;

        case "answer_delta":
          updatedStatus = "generating";
          updatedProvisional += event.data.delta;
          break;

        case "verification_started":
          updatedStatus = "verifying";
          updatedMetadata.claimCount = event.data.claim_count;
          break;

        case "answer_completed":
          updatedStatus = "complete";
          updatedResult = event.data.result;
          if (event.data.replace_streamed_answer) {
            updatedProvisional = event.data.result.answer;
          } else if (!updatedProvisional) {
            updatedProvisional = event.data.result.answer;
          }
          break;

        case "error":
          updatedStatus = "error";
          updatedError = {
            code: event.data.code,
            message: event.data.message,
            retryable: event.data.retryable,
            partialAnswer: event.data.partial_answer ?? updatedProvisional,
          };
          if (event.data.partial_answer) {
            updatedProvisional = event.data.partial_answer;
          }
          break;
      }

      const isTerminal =
        event.event === "answer_completed" || event.event === "error";

      return {
        ...state,
        globalStatus: isTerminal ? "idle" : updatedStatus,
        activeEntryId: isTerminal ? null : state.activeEntryId,
        entries: state.entries.map((entry) =>
          entry.id === entryId
            ? {
                ...entry,
                requestId: entry.requestId || requestId,
                lastSequence: event.sequence,
                status: updatedStatus,
                provisionalAnswer: updatedProvisional,
                result: updatedResult,
                error: updatedError,
                stageMetadata: updatedMetadata,
              }
            : entry
        ),
      };
    }

    case "FAIL_REQUEST": {
      const { entryId, code, message, retryable, partialAnswer } =
        action.payload;

      const isMatchingActive = state.activeEntryId === entryId;

      return {
        ...state,
        globalStatus: "idle",
        activeEntryId: isMatchingActive ? null : state.activeEntryId,
        entries: state.entries.map((entry) => {
          if (entry.id === entryId) {
            return {
              ...entry,
              status: "error",
              error: {
                code,
                message,
                retryable,
                partialAnswer: partialAnswer ?? entry.provisionalAnswer,
              },
            };
          }
          return entry;
        }),
      };
    }

    case "CLEAR_CONVERSATION": {
      return {
        ...INITIAL_STATE,
      };
    }

    case "SET_INPUT": {
      return {
        ...state,
        inputValue: action.payload,
        validationError: null,
      };
    }

    case "SET_VALIDATION_ERROR": {
      return {
        ...state,
        validationError: action.payload,
      };
    }

    default:
      return state;
  }
}
