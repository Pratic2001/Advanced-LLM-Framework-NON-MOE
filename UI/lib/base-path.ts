// Client- and server-safe base path for the Tailscale funnel sub-path.
//
// next.config.mjs reads the SAME env var (NEXT_PUBLIC_BASE_PATH) for its
// `basePath`, so a hand-written fetch/WebSocket/EventSource URL built from
// this constant always lines up with where Next actually serves the app. It
// is a NEXT_PUBLIC_* var, so it is inlined at build time into client
// components and also present in process.env when server.ts runs via tsx.
export const BASE_PATH = (process.env.NEXT_PUBLIC_BASE_PATH ?? "").replace(/\/$/, "");

/**
 * Prefix an absolute app path with the base path, e.g. api("/api/jobs")
 * -> "/heavy/api/jobs". next/link, next/router and next/image already get
 * the prefix automatically from Next's own basePath; only raw strings that
 * reach the network (fetch, WebSocket, EventSource) need this helper.
 */
export const api = (p: string) => `${BASE_PATH}${p}`;