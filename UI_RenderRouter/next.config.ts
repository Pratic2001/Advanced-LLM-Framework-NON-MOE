import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // This repo has multiple lockfiles (UI, UI_lite, UI_RenderRouter…), which
  // makes Turbopack guess the wrong workspace root. Pin it to this app.
  turbopack: { root: process.cwd() },
};

export default nextConfig;
