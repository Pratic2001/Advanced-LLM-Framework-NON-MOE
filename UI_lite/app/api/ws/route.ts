import { NextResponse } from "next/server";
import { BASE_PATH } from "@/lib/base-path";

/**
 * WebSocket upgrade endpoint
 * Note: Next.js App Router route handlers cannot directly upgrade to WebSocket.
 * This handler returns a 400 with instructions — the WebSocket server is
 * mounted at the HTTP server level in next.config.ts or a custom server.
 *
 * For production, use a custom server (server.ts) that wraps Next.js
 * and mounts the WebSocket server alongside it.
 */

export function GET() {
  return NextResponse.json(
    {
      error: "WebSocket connections must use the ws:// or wss:// protocol",
      hint: `Connect to ws://hostname${BASE_PATH}/api/ws instead`,
    },
    { status: 400 }
  );
}
