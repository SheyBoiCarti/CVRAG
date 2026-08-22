"use client";

import React from "react";
import { PipelineStage, StageMetadata } from "../lib/chat/state";
import styles from "./PipelineStatus.module.css";

export interface PipelineStatusProps {
  status: PipelineStage;
  metadata?: StageMetadata;
}

export const PipelineStatus: React.FC<PipelineStatusProps> = ({
  status,
  metadata,
}) => {
  if (status === "idle" || status === "complete") {
    return null;
  }

  const getStatusText = (): string => {
    switch (status) {
      case "submitting":
        return "Submitting request...";
      case "retrieving":
        return metadata?.evidenceCount !== undefined
          ? `Searching CV corpus (found ${metadata.evidenceCount} passages)...`
          : "Searching CV corpus...";
      case "reranking":
        return metadata?.candidateCount !== undefined
          ? `Reranking ${metadata.candidateCount} candidate passages...`
          : "Reranking candidate passages...";
      case "generating":
        return "Synthesizing answer...";
      case "verifying":
        return metadata?.claimCount !== undefined && metadata.claimCount !== null
          ? `Verifying ${metadata.claimCount} extracted claims against evidence...`
          : "Verifying extracted claims...";
      case "error":
        return "Pipeline error";
      default:
        return "Processing...";
    }
  };

  const getStageLabel = (stage: PipelineStage): string => {
    switch (stage) {
      case "retrieving":
        return "Searching";
      case "reranking":
        return "Reranking";
      case "generating":
        return "Generating";
      case "verifying":
        return "Verifying";
      default:
        return stage;
    }
  };

  const isStageActive = (targetStage: PipelineStage): boolean => {
    return status === targetStage;
  };

  return (
    <div
      className={styles.container}
      role="status"
      aria-live="polite"
      aria-atomic="true"
    >
      <div className={styles.spinner} aria-hidden="true" />
      <span className={styles.statusMessage}>{getStatusText()}</span>
      <div className={styles.stagesList}>
        {status === "retrieving" && (
          <span
            className={`${styles.stageBadge} ${
              isStageActive("retrieving") ? styles.active : ""
            }`}
          >
            {getStageLabel("retrieving")}
          </span>
        )}
        {status === "reranking" && (
          <span
            className={`${styles.stageBadge} ${
              isStageActive("reranking") ? styles.active : ""
            }`}
          >
            {getStageLabel("reranking")}
          </span>
        )}
        {status === "generating" && (
          <span
            className={`${styles.stageBadge} ${
              isStageActive("generating") ? styles.active : ""
            }`}
          >
            {getStageLabel("generating")}
          </span>
        )}
        {status === "verifying" && (
          <span
            className={`${styles.stageBadge} ${
              isStageActive("verifying") ? styles.active : ""
            }`}
          >
            {getStageLabel("verifying")}
          </span>
        )}
      </div>
    </div>
  );
};
