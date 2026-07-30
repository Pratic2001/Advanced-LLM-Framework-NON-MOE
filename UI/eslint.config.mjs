// ESLint v9 flat config
// Uses @eslint/eslintrc to bridge existing Next.js ESLint config

import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({ baseDirectory: __dirname });

const eslintConfig = [
  ...compat.extends("next/core-web-vitals"),
  {
    rules: {
      "no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
      "prefer-const": "warn",
      "no-console": ["warn", { allow: ["warn", "error"] }],
    },
  },
];

export default eslintConfig;
