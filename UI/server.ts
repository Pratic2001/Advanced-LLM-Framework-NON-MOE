/**
 * Custom server for WebSocket support
 *
 * Next.js App Router route handlers don't support WebSocket upgrades.
 * This custom server wraps Next.js and mounts the WebSocket server alongside it.
 *
 * Usage: npx tsx server.ts        # auto-loads .env.local (see package.json script)
 *        npm run start:ws         # same, via npm script
 * In production: Build first (npm run build) then use node .next/standalone/server.js
 * (which does NOT support WebSocket — must run this custom server instead).
 *
 * Env loading: `tsx` and bare Node do NOT auto-load .env.local the way
 * `next dev` / `next start` do. The `start:ws` npm script passes
 * `--env-file=.env.local` to tsx so env vars are populated before any module
 * reads them. Don't run plain `tsx server.ts` — env will be missing.
 */

import { createServer } from "http";
import { parse } from "url";
import next from "next";
import { getWSManager } from "./lib/ws-manager";
import { env } from "./lib/env";
import { BASE_PATH } from "./lib/base-path";

const dev = env.NODE_ENV !== "production";
const hostname = env.HOSTNAME;
const port = env.PORT;

const app = next({ dev, hostname, port });
const handle = app.getRequestHandler();

app.prepare().then(() => {
  const server = createServer((req, res) => {
    // Basic CORS headers
    const origin = req.headers.origin;
    // Match on scheme+host only. A browser's Origin header never carries a
    // path, so comparing against the full NEXTAUTH_URL (which now ends in
    // /heavy for the funnel) would silently disable these CORS headers.
    const allowedOrigin = (() => {
      const raw = process.env.NEXTAUTH_URL || `http://${hostname}:${port}`;
      try {
        return new URL(raw).origin;
      } catch {
        return raw;
      }
    })();
    if (origin && origin === allowedOrigin) {
      res.setHeader("Access-Control-Allow-Origin", origin);
      res.setHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
      res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
    }

    // Handle preflight
    if (req.method === "OPTIONS") {
      res.writeHead(204);
      res.end();
      return;
    }

    const parsedUrl = parse(req.url!, true);

    // Tailscale Funnel mounts this app under BASE_PATH (e.g. /heavy) and
    // strips that prefix before proxying to the backend. The basePath build
    // only serves URLs that carry the prefix, so re-apply it here so that
    // funnel-stripped pages, API routes, and _next assets resolve correctly.
    const bp = BASE_PATH;
    if (bp && parsedUrl.pathname && !parsedUrl.pathname.startsWith(bp)) {
      // A funnel-stripped root ("/") would re-prepend to "/heavy/", which Next
      // 308s to "/heavy" — and since that is the very URL the funnel already
      // delivered, it becomes a redirect loop. Map root to the bare prefix.
      parsedUrl.pathname = parsedUrl.pathname === "/" ? bp : bp + parsedUrl.pathname;
      parsedUrl.path = parsedUrl.pathname + (parsedUrl.search || "");
    }

    handle(req, res, parsedUrl);
  });

  // Initialize WebSocket manager (broadcasts job events to subscribed clients)
  getWSManager().init(server);

  server.listen(port, () => {
    console.log(
      `[server] Listening at http://${hostname}:${port} (WebSocket: ws://${hostname}:${port}${BASE_PATH}/api/ws)`
    );
  });

  // ── Graceful shutdown ──────────────────────────────────────────────────
  const shutdown = (signal: string) => {
    console.log(`[server] Received ${signal}, shutting down gracefully...`);
    server.close(() => {
      console.log("[server] HTTP server closed");
      process.exit(0);
    });

    // Force exit after 10s if graceful shutdown hangs
    setTimeout(() => {
      console.error("[server] Forced shutdown after timeout");
      process.exit(1);
    }, 10000).unref();
  };

  process.on("SIGTERM", () => shutdown("SIGTERM"));
  process.on("SIGINT", () => shutdown("SIGINT"));
}).catch((err) => {
  console.error("[server] Failed to start:", err);
  process.exit(1);
});
