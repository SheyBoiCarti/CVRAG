import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sheharyar Nadeem — AI Portfolio & CV RAG Assistant",
  description:
    "Interactive CV knowledge base and retrieval-augmented assistant powered by FastAPI and Next.js.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
