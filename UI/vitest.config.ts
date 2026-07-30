import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    include: ["**/*.test.{ts,js}", "tests/**/*.test.{ts,js}"],
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname),
    },
  },
});
