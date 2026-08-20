"use client";

import React from "react";
import styles from "./SuggestionGrid.module.css";

export interface SuggestionItem {
  id: string;
  label: string;
  prompt: string;
}

export const SUGGESTIONS: SuggestionItem[] = [
  {
    id: "summary",
    label: "Professional Summary",
    prompt:
      "Give me a summary of Sheharyar's professional experience and key strengths.",
  },
  {
    id: "skills",
    label: "Technical Skills",
    prompt:
      "What technical skills, programming languages, and tools does Sheharyar possess?",
  },
  {
    id: "experience",
    label: "Work Experience",
    prompt:
      "What are Sheharyar's key employment roles, engineering experience, and achievements?",
  },
  {
    id: "ai_ml",
    label: "AI & Machine Learning",
    prompt:
      "Tell me about Sheharyar's AI, Machine Learning, and RAG project experience.",
  },
  {
    id: "education",
    label: "Education & Degrees",
    prompt:
      "What education, academic degrees, or certifications does Sheharyar hold?",
  },
  {
    id: "backend",
    label: "Backend & Architecture",
    prompt:
      "What experience does Sheharyar have in backend systems and software development?",
  },
];

export interface SuggestionGridProps {
  onSelectSuggestion: (prompt: string) => void;
  disabled?: boolean;
}

export const SuggestionGrid: React.FC<SuggestionGridProps> = ({
  onSelectSuggestion,
  disabled = false,
}) => {
  return (
    <section className={styles.heroContainer} aria-label="Suggested Questions">
      <h2 className={styles.heroTitle}>
        Ask about my experience, skills & background
      </h2>
      <p className={styles.heroSubtitle}>
        Explore my software engineering journey, technical projects, education,
        and professional background through an AI assistant powered by a
        Retrieval-Augmented Generation (RAG) pipeline indexing my CV.
      </p>

      <div className={styles.suggestionsLabel}>Suggested Questions</div>

      <div className={styles.grid}>
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion.id}
            type="button"
            className={styles.suggestionButton}
            onClick={() => onSelectSuggestion(suggestion.prompt)}
            disabled={disabled}
            aria-label={`Ask suggestion: ${suggestion.label}`}
          >
            {suggestion.label}
          </button>
        ))}
      </div>
    </section>
  );
};
