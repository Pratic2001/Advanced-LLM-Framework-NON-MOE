import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Standalone output gives us a tiny node_modules tree (just the deps Next
  // traces) plus a server.js entrypoint — used by Dockerfile.ui-router so
  // the production image can `CMD ["node", "server.js"]` without running
  // `npm install` at container start. See Next docs:
  // https://nextjs.org/docs/app/api-reference/config/next-config-js/output
  output: "standalone",

  // This repo has multiple lockfiles (UI, UI_lite, UI_RenderRouter…), which
  // makes Turbopack guess the wrong workspace root. Pin it to this app.
  turbopack: { root: process.cwd() },
};

export default nextConfig;
