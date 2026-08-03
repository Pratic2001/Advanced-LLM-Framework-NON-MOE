// ESLint v9 flat config
// eslint-config-next v16 ships flat configs directly — spread its array rather
// than bridging through FlatCompat (that was written for legacy eslintrc-style
// configs and crashes on the newer circular plugin references).

import nextCoreWebVitals from "eslint-config-next/core-web-vitals";

const eslintConfig = [
  ...nextCoreWebVitals,
  {
    rules: {
      "no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
      "prefer-const": "warn",
      "no-console": ["warn", { allow: ["warn", "error"] }],
    },
  },
];

export default eslintConfig;
