"use client";

import React from "react";
import styles from "./ProfileSidebar.module.css";

export interface ProfileSidebarProps {
  hasHistory: boolean;
  onClearHistory: () => void;
  isOpen?: boolean;
  onClose?: () => void;
}

export const ProfileSidebar: React.FC<ProfileSidebarProps> = ({
  hasHistory,
  onClearHistory,
  isOpen = false,
  onClose,
}) => {
  return (
    <>
      {isOpen && (
        <div
          className={`${styles.backdrop} ${styles.open}`}
          onClick={onClose}
          aria-hidden="true"
        />
      )}
      <aside
        className={`${styles.sidebar} ${isOpen ? styles.open : ""}`}
        aria-label="Candidate Profile and Controls"
      >
        {onClose && (
          <button
            type="button"
            className={styles.mobileCloseButton}
            onClick={onClose}
            aria-label="Close sidebar"
          >
            ✕ Close
          </button>
        )}

        <div className={styles.profileCard}>
          <div className={styles.profileName}>Sheharyar Nadeem</div>
          <div className={styles.profileTitle}>Software Engineer & AI Developer</div>
          <div className={styles.profileDetails}>
            🎓 <strong>Honours Computer Science</strong> (Cum Laude)<br />
            📍 <strong>Location</strong>: Open to Remote & Relocation
          </div>
        </div>

        <div className={styles.sectionTitle}>Quick Links</div>
        <ul className={styles.linkList}>
          <li className={styles.linkItem}>
            <a
              href="https://sheharyar-nadeem.vercel.app/"
              target="_blank"
              rel="noopener noreferrer"
            >
              🌐 <span>Live Portfolio</span>
            </a>
          </li>
          <li className={styles.linkItem}>
            <a
              href="https://github.com/SheyBoiCarti"
              target="_blank"
              rel="noopener noreferrer"
            >
              🐙 <span>GitHub Profile</span>
            </a>
          </li>
          <li className={styles.linkItem}>
            <a
              href="https://linkedin.com/in/sheharyar-nadeem/"
              target="_blank"
              rel="noopener noreferrer"
            >
              💼 <span>LinkedIn Connect</span>
            </a>
          </li>
        </ul>

        <hr className={styles.divider} />

        <div className={styles.sectionTitle}>Session Controls</div>
        <button
          type="button"
          className={styles.clearButton}
          onClick={onClearHistory}
          disabled={!hasHistory}
          aria-label="Clear conversation history"
        >
          🗑️ Clear Conversation
        </button>

        <hr className={styles.divider} />

        <div className={styles.sectionTitle}>RAG Engine Architecture</div>
        <ul className={styles.archList}>
          <li className={styles.archItem}>
            <strong>LLM Backbone</strong>: <code>meta/llama-3.3-70b-instruct</code>
          </li>
          <li className={styles.archItem}>
            <strong>Embeddings</strong>: <code>nvidia/nv-embedqa-e5-v5</code>
          </li>
          <li className={styles.archItem}>
            <strong>Vector Storage</strong>: <code>ChromaDB</code> (Persistent)
          </li>
        </ul>
      </aside>
    </>
  );
};
