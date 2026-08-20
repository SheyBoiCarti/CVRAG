"use client";

import React from "react";
import styles from "./PortfolioHeader.module.css";

export interface PortfolioHeaderProps {
  onToggleSidebar?: () => void;
}

export const PortfolioHeader: React.FC<PortfolioHeaderProps> = ({
  onToggleSidebar,
}) => {
  return (
    <header className={styles.headerContainer}>
      <div className={styles.topBar}>
        {onToggleSidebar && (
          <button
            type="button"
            className={styles.menuButton}
            onClick={onToggleSidebar}
            aria-label="Open sidebar menu"
          >
            ☰ Profile
          </button>
        )}
        <div className={styles.titleGroup}>
          <h1 className={styles.title}>Sheharyar Nadeem</h1>
          <p className={styles.subtitle}>Software Developer & AI Engineer</p>
        </div>
      </div>

      <details className={styles.accordion}>
        <summary className={styles.accordionSummary}>
          How this CV RAG Assistant Works & Technical Models
        </summary>
        <div className={styles.accordionContent}>
          <p>
            <strong>1. Document Chunking</strong>
            <br />
            The candidate&apos;s CV document is split into semantic text chunks (
            <code>chunk_size=700</code>, <code>overlap=140</code>) to preserve
            context continuity.
          </p>
          <p>
            <strong>2. Embeddings & Vector Storage</strong>
            <br />
            Chunks are converted into dense vector embeddings using{" "}
            <strong>NVIDIA QA & Text Embeddings</strong> (
            <code>nvidia/nv-embedqa-e5-v5</code>) and indexed in a local{" "}
            <strong>ChromaDB</strong> vector database.
          </p>
          <p>
            <strong>3. Context Retrieval & LLM Generation</strong>
            <br />
            Top relevant CV snippets are retrieved via similarity search and
            passed to <strong>Meta Llama 3.3 70B Instruct</strong> (
            <code>meta/llama-3.3-70b-instruct</code>) to generate factually
            grounded responses.
          </p>
          <div className={styles.disclaimerBanner}>
            <strong>System & Performance Disclaimer:</strong>
            <br />
            Powered by <strong>NVIDIA AI Foundation Free Endpoints</strong>.
            Responses and performance may be slow, and users can experience rate
            limits during peak usage. If an API request times out, please wait a
            moment and try again.
          </div>
        </div>
      </details>
    </header>
  );
};
