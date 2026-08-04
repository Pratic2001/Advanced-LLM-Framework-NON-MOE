/**
 * GET /api/shell/stream?sessionId=...
 *
 * Server-Sent Events stream of bash PTY output for a previously opened
 * shell session. This replaces the WebSocket transport for the interactive
 * shell — needed because Tailscale Funnel aggressively kills WebSocket
 * upgrades within milliseconds of opening, while normal HTTP requests
 * (including long-lived SSE) pass through.
 *
 * Wire format (text/event-stream):
 *   event: opened\n
 *   data: {"pid": 12345, "activePtys": 2}\n
 *   \n
 *
 *   event: data
 *   data: {"data": "<raw pty chunk>"}
 *
 *   event: exit
 *   data: {"exitCode": 0, "signal": 0}
 *
 *   event: error
 *   data: {"error": "..."}
 *
 *   :keepalive\n\n   (comment lines every 15s — proxies keep the connection
 *                     open when they see activity)
 *
 * The stream closes when the PTY exits, the client disconnects
 * (AbortSignal), or the idle timer fires.
 *
 * Auth: same as /api/shell/open — session cookie via `auth()`. The
 * sessionId is also verified against the requesting user's userId before
 * any PTY data is sent.
 */
import { auth } from "@/lib/auth";
import {
  getSession,
  subscribeOutput,
  subscribeExit,
} from "@/lib/shell-sessions";
import { getActivePtyCount } from "@/lib/pty-manager";

export const dynamic = "force-dynamic";

// Node runtime — we use Node's `TextEncoder` and need long-lived
// connections. Edge would cap them too aggressively.
export const runtime = "nodejs";

const KEEPALIVE_INTERVAL_MS = 15_000;

export async function GET(req: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return new Response(JSON.stringify({ error: "Unauthorized" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  const url = new URL(req.url);
  const sessionId = url.searchParams.get("sessionId");
  if (!sessionId) {
    return new Response(JSON.stringify({ error: "Missing sessionId" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const rec = getSession(sessionId, session.user.id);
  if (!rec) {
    return new Response(JSON.stringify({ error: "Unknown session" }), {
      status: 404,
      headers: { "Content-Type": "application/json" },
    });
  }

  const encoder = new TextEncoder();
  let unsubscribeOutput: (() => void) | null = null;
  let unsubscribeExit: (() => void) | null = null;
  let keepaliveTimer: NodeJS.Timeout | null = null;
  let closed = false;

  const stream = new ReadableStream({
    start(controller) {
      function send(event: string, payload: unknown) {
        if (closed) return;
        try {
          const chunk = `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`;
          controller.enqueue(encoder.encode(chunk));
        } catch {
          // Controller already closed — ignore.
        }
      }

      function close() {
        if (closed) return;
        closed = true;
        if (unsubscribeOutput) unsubscribeOutput();
        if (unsubscribeExit) unsubscribeExit();
        if (keepaliveTimer) clearInterval(keepaliveTimer);
        try {
          controller.close();
        } catch {}
      }

      // Send the initial 'opened' frame so the client knows the PTY is up.
      send("opened", { pid: rec.pty.pid, activePtys: getActivePtyCount() });

      // Fan out PTY stdout/stderr.
      unsubscribeOutput = subscribeOutput(rec, (chunk) => {
        send("data", { data: chunk });
      });

      // PTY exit — notify once, then close the stream.
      unsubscribeExit = subscribeExit(rec, ({ exitCode, signal }) => {
        send("exit", { exitCode, signal: signal ?? 0 });
        close();
      });

      // Keepalive — SSE comment lines don't fire client events but they do
      // keep proxies from killing the connection on idle timeouts. Critically,
      // this also resets any aggressive Funnel/proxy read timeout.
      keepaliveTimer = setInterval(() => {
        if (closed) return;
        try {
          controller.enqueue(encoder.encode(`:keepalive\n\n`));
        } catch {
          close();
        }
      }, KEEPALIVE_INTERVAL_MS);
      keepaliveTimer.unref?.();

      // Client disconnected (browser closed EventSource, navigated away,
      // closed the tab). Tear down our subscriptions and stop the keepalive.
      req.signal.addEventListener("abort", () => {
        close();
      });
    },
    cancel() {
      // The consumer of the ReadableStream (Node's HTTP response) cancelled
      // — usually because the client disconnected. Run the same teardown.
      closed = true;
      if (unsubscribeOutput) unsubscribeOutput();
      if (unsubscribeExit) unsubscribeExit();
      if (keepaliveTimer) clearInterval(keepaliveTimer);
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      // Disable proxy buffering — nginx/Tailscale Funnel will otherwise
      // buffer the entire response and only flush at the end, which
      // defeats the point of SSE.
      "X-Accel-Buffering": "no",
      "Connection": "keep-alive",
    },
  });
}