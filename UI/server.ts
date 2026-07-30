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
import { WebSocketServer } from "ws";
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

  // Initialize WebSocket server
  const wss = new WebSocketServer({ server, path: "/api/ws" });

  wss.on("connection", (ws, req) => {
    console.log("[WS] Client connected");

    ws.on("message", (data: Buffer) => {
      try {
        const message = JSON.parse(data.toString());

        switch (message.type) {
          case "subscribe":
            console.log(`[WS] Client subscribed to: ${message.jobId || message.channel}`);
            ws.send(
              JSON.stringify({
                type: "subscribed",
                channel: message.jobId || message.channel,
                timestamp: new Date().toISOString(),
              })
            );
            break;
          case "unsubscribe":
            console.log(`[WS] Client unsubscribed from: ${message.jobId || message.channel}`);
            break;
          default:
            ws.send(
              JSON.stringify({
                type: "echo",
                ...message,
                timestamp: new Date().toISOString(),
              })
            );
        }
      } catch {
        ws.send(JSON.stringify({ type: "error", message: "Invalid JSON" }));
      }
    });

    ws.on("close", () => {
      console.log("[WS] Client disconnected");
    });

    // Send connection confirmation
    ws.send(
      JSON.stringify({
        type: "connected",
        timestamp: new Date().toISOString(),
      })
    );
  });

  server.listen(port, () => {
    console.log(
      `> Server listening at http://${hostname}:${port} (WebSocket: ws://${hostname}:${port}/api/ws)`
    );
  });
});
