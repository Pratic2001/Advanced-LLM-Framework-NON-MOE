/**
 * Custom server for WebSocket support
 *
 * Next.js App Router route handlers don't support WebSocket upgrades.
 * This custom server wraps Next.js and mounts the WebSocket server alongside it.
 *
 * Usage: npx tsx server.ts
 * In production: Build first (npm run build) then use compiled version
 */

import { createServer } from "http";
import { parse } from "url";
import next from "next";
import { getWSManager } from "./lib/ws-manager";

const dev = process.env.NODE_ENV !== "production";
const hostname = "localhost";
const port = parseInt(process.env.PORT || "3000", 10);

const app = next({ dev, hostname, port });
const handle = app.getRequestHandler();

app.prepare().then(() => {
  const server = createServer((req, res) => {
    const parsedUrl = parse(req.url!, true);
    handle(req, res, parsedUrl);
  });

  // Initialize WebSocket manager (broadcasts job events to subscribed clients)
  getWSManager().init(server);

  server.listen(port, () => {
    console.log(
      `> Server listening at http://${hostname}:${port} (WebSocket: ws://${hostname}:${port}/api/ws)`
    );
  });
});
