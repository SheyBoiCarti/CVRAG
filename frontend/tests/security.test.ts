import { describe, expect, it } from "vitest";
import fs from "fs";
import path from "path";

describe("Frontend Security and Environment Constraints", () => {
  it("ensures no client components or client bundles reference NVIDIA_API_KEY", () => {
    const frontendDir = path.resolve(__dirname, "..");
    const clientFiles: string[] = [];

    function scanDir(dir: string) {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        if (
          entry.name === "node_modules" ||
          entry.name === ".next" ||
          entry.name === "dist" ||
          entry.name === "route.ts" ||
          entry.name === "security.test.ts"
        ) {
          continue;
        }
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          scanDir(fullPath);
        } else if (/\.(tsx|ts|js|jsx|css)$/.test(entry.name)) {
          clientFiles.push(fullPath);
        }
      }
    }

    scanDir(frontendDir);

    for (const file of clientFiles) {
      const content = fs.readFileSync(file, "utf-8");
      expect(content).not.toContain("NVIDIA_API_KEY");
      expect(content).not.toContain("NEXT_PUBLIC_BACKEND_URL");
      expect(content).not.toContain("NEXT_PUBLIC_API_KEY");
      expect(content).not.toContain("NEXT_PUBLIC_NVIDIA_API_KEY");
    }
  });

  it("verifies BACKEND_URL is only referenced in server-only route", () => {
    const apiRoutePath = path.resolve(
      __dirname,
      "../app/api/chat/route.ts"
    );
    const content = fs.readFileSync(apiRoutePath, "utf-8");
    expect(content).toContain("process.env.BACKEND_URL");
    expect(content).not.toContain("NEXT_PUBLIC_");
  });
});
