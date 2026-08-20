"use client";

import React from "react";
import { Citation, EvidenceSupportStatus } from "../lib/api/types";
import styles from "./CitationList.module.css";

export interface CitationListProps {
  citations: Citation[];
}

export const CitationList: React.FC<CitationListProps> = ({ citations }) => {
  if (!citations || citations.length === 0) {
    return null;
  }

  const getSourceSummary = (): string => {
    const sources = Array.from(
      new Set(
        citations.map((c) => {
          const lower = c.source.toLowerCase();
          if (lower.includes("cv") || lower.includes("resume")) {
            return "CV Document";
          }
          return c.source;
        })
      )
    );
    return sources.join(", ");
  };

  const getBadgeClass = (status: EvidenceSupportStatus): string => {
    switch (status) {
      case "supported":
        return styles.badgeSupported;
      case "unsupported":
        return styles.badgeUnsupported;
      case "not_checked":
      default:
        return styles.badgeNotChecked;
    }
  };

  const getBadgeLabel = (status: EvidenceSupportStatus): string => {
    switch (status) {
      case "supported":
        return "Supported";
      case "unsupported":
        return "Unsupported";
      case "not_checked":
      default:
        return "Unchecked";
    }
  };

  return (
    <div className={styles.container} data-testid="citation-list">
      <details className={styles.expander}>
        <summary className={styles.summary}>
          <span>Grounded RAG Context ({getSourceSummary()})</span>
          <span aria-hidden="true">({citations.length} sources)</span>
        </summary>

        <div className={styles.citationCardList}>
          {citations.map((citation, index) => (
            <div
              key={citation.evidence_id || `citation-${index}`}
              className={styles.citationCard}
              data-testid={`citation-card-${citation.evidence_id || index}`}
            >
              <div className={styles.cardHeader}>
                <span className={styles.headerTitle}>
                  Citation [{citation.evidence_id || `E${index + 1}`}]
                </span>
                <span
                  className={`${styles.badge} ${getBadgeClass(
                    citation.support_status
                  )}`}
                  data-testid="citation-status-badge"
                >
                  {getBadgeLabel(citation.support_status)}
                </span>
              </div>

              <div className={styles.sourceLocation}>
                Source: {citation.source} (Page {citation.page})
              </div>

              {/* Secure text node rendering - NEVER dangerouslySetInnerHTML */}
              <div className={styles.excerptText} data-testid="citation-excerpt">
                {citation.excerpt}
              </div>
            </div>
          ))}
        </div>
      </details>
    </div>
  );
};
