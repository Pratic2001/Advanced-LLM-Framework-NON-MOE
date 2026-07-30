/**
 * WebSocket Manager
 *
 * Manages the single WebSocket endpoint at `/api/ws` used to stream
 * job events (logs, metrics, status, errors) to subscribed clients.
 *
 * Note: the interactive shell is intentionally NOT over WebSocket — Tailscale
 * Funnel aggressively kills WebSocket upgrades within milliseconds of
 * opening, so a public HTTPS funnel can't carry them. The shell uses SSE
 * (server → client) + HTTP POST (client → server) instead. See
 * `app/api/shell/*` and `components/InteractiveShell.tsx`.
 */
import { WebSocketServer, WebSocket } from "ws";
import { IncomingMessage } from "http";
import { JobManager } from "./job-manager";
import type { JobEvent } from "./job-manager";
import { decode } from "next-auth/jwt";
import { env as appEnv } from "./env";

interface WsClient {
  ws: WebSocket;
  userId: string;
  subscriptions: Set<string>; // jobIds or "dashboard"
  id: string;
}

export class WSManager {
  private wss: WebSocketServer | null = null;
  private clients: Map<string, WsClient> = new Map();
  private jobManager: JobManager;

  constructor() {
    this.jobManager = JobManager.getInstance();
  }

  /**
   * Initialize the WebSocket server
   */
  init(server: any): void {
    this.wss = new WebSocketServer({ server, path: "/api/ws" });

    this.wss.on("connection", (ws: WebSocket, req: IncomingMessage) => {
      const clientId = this.generateId();

      // The client (see hooks/use-websocket.ts) sends `subscribe` immediately
      // in its `onopen` handler — often within the same tick the upgrade
      // completes. Auth below is async (Auth.js's `decode()` does real JWE/JWT
      // crypto work), so if we don't attach the `message` listener until
      // *after* that `await`, any frames the client sent in the meantime
      // arrive with nobody listening and are silently dropped.
      //
      // Fix: attach listeners synchronously, right here, and buffer any
      // messages that arrive before we know whether the connection is
      // authenticated. Once auth resolves we either replay the buffer
      // through the normal handler or close the socket.
      let authResolved = false;
      const pendingMessages: Buffer[] = [];

      const client: WsClient = {
        ws,
        userId: "",
        subscriptions: new Set(),
        id: clientId,
      };

      const handleMessage = (data: Buffer) => {
        try {
          const message = JSON.parse(data.toString());

          switch (message.type) {
            case "auth":
              client.userId = message.userId;
              break;
            case "subscribe":
              if (message.jobId) {
                client.subscriptions.add(message.jobId);
              }
              if (message.channel === "dashboard") {
                client.subscriptions.add("dashboard");
              }
              break;
            case "unsubscribe":
              if (message.jobId) {
                client.subscriptions.delete(message.jobId);
              }
              if (message.channel === "dashboard") {
                client.subscriptions.delete("dashboard");
              }
              break;
          }
        } catch (err) {
          console.error("[WSManager] Invalid message:", err);
        }
      };

      // Attached immediately (synchronously) so nothing sent right after
      // `onopen` can be missed. Until auth resolves, frames are buffered
      // instead of handled.
      ws.on("message", (data: Buffer) => {
        if (!authResolved) {
          pendingMessages.push(data);
          return;
        }
        handleMessage(data);
      });

      ws.on("close", () => {
        this.clients.delete(clientId);
      });

      ws.on("error", () => {
        this.clients.delete(clientId);
      });

      // Authenticate from the session cookie on the upgrade request. Auth.js
      // v5 sets `httpOnly: true`, so the browser can't read the JWT from JS;
      // it must be verified server-side from `req.headers.cookie`. Same-origin
      // WS upgrades forward cookies automatically.
      this.authenticateFromCookies(req)
        .then((userId) => {
          if (ws.readyState !== WebSocket.OPEN) {
            // Client disconnected mid-auth (e.g. page re-mount tore down
            // the WS before the async JWT decode returned). Nothing to do
            // — the socket is already gone. The buffered messages are
            // dropped, which is correct: the client that sent them is no
            // longer listening.
            return;
          }

          if (!userId) {
            // No valid session cookie — refuse the connection so the client
            // gets a clear error instead of silently rejecting every frame.
            authResolved = true;
            try {
              ws.close(1008, "Not authenticated");
            } catch {}
            return;
          }

          client.userId = userId;
          this.clients.set(clientId, client);
          authResolved = true;

          // Send connection confirmation
          try {
            ws.send(
              JSON.stringify({
                type: "connected",
                clientId,
                timestamp: new Date().toISOString(),
              })
            );
          } catch {}

          // Replay anything that arrived while we were still authenticating.
          while (pendingMessages.length > 0) {
            const data = pendingMessages.shift()!;
            try {
              handleMessage(data);
            } catch (err) {
              console.error("[WSManager] handleMessage threw:", err);
            }
          }
        })
        .catch((err) => {
          // If anything in the auth pipeline rejects (e.g. JWE decode blew
          // up, or a downstream `.then` step threw), don't let it vanish
          // into an unhandled rejection — log it and close the socket so
          // the client at least sees a clear error instead of a silent drop.
          console.error("[WSManager] auth pipeline rejected:", err);
          authResolved = true;
          try {
            ws.close(1011, "Auth error");
          } catch {}
        });
    });

    // Listen to JobManager events and broadcast
    this.jobManager.on("job:log", (event: JobEvent) =>
      this.broadcast(event.jobId, event)
    );
    this.jobManager.on("job:metric", (event: JobEvent) =>
      this.broadcast(event.jobId, event)
    );
    this.jobManager.on("job:status", (event: JobEvent) => {
      this.broadcast(event.jobId, event);
      this.broadcast("dashboard", event);
    });
    this.jobManager.on("job:error", (event: JobEvent) => {
      this.broadcast(event.jobId, event);
      this.broadcast("dashboard", event);
    });
  }

  /**
   * Broadcast an event to subscribers
   */
  private broadcast(channel: string, event: JobEvent): void {
    Array.from(this.clients.values()).forEach((client) => {
      if (client.subscriptions.has(channel)) {
        try {
          client.ws.send(JSON.stringify(event));
        } catch {
          // Client disconnected
        }
      }
    });
  }

  /**
   * Get the number of connected clients
   */
  get clientCount(): number {
    return this.clients.size;
  }

  private generateId(): string {
    return `ws_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  }

  /**
   * Verify a NextAuth session token (JWT) and return the userId if valid.
   * Returns null on any failure. The token comes from a session cookie on
   * the WS upgrade request; the cookie name is used as the JWE salt.
   */
  private async verifySessionToken(
    token: string,
    cookieName: string
  ): Promise<string | null> {
    try {
      // Auth.js v5 encrypts the JWT (A256CBC-HS512 JWE) using NEXTAUTH_SECRET
      // keyed by the cookie name (`salt`). Salt MUST match the name the
      // cookie was originally issued with.
      const decoded = await decode({
        token,
        secret: appEnv.NEXTAUTH_SECRET,
        salt: cookieName,
      });

      if (!decoded) return null;
      // The userId is stored as `sub` in the JWT payload.
      const userId = (decoded as any).sub || (decoded as any).userId;
      return typeof userId === "string" ? userId : null;
    } catch (err) {
      console.error("[WSManager] verifySessionToken failed:", err);
      return null;
    }
  }

  /**
   * Read the NextAuth session cookie from the WS upgrade request headers
   * and verify it. Tries the v5 (`authjs.session-token`) name first, then
   * the v4 (`next-auth.session-token`) fallback for sessions minted before
   * the Auth.js upgrade.
   *
   * Returns the authenticated userId, or null if no valid session is found.
   */
  private async authenticateFromCookies(
    req: IncomingMessage
  ): Promise<string | null> {
    const raw = req.headers.cookie ?? "";
    if (!raw) return null;

    const cookies = new Map<string, string>();
    for (const part of raw.split(";")) {
      const i = part.indexOf("=");
      if (i < 0) continue;
      const k = part.slice(0, i).trim();
      const v = decodeURIComponent(part.slice(i + 1).trim());
      cookies.set(k, v);
    }

    // Detect HTTPS so we know whether to try the `__Secure-` prefixed names.
    const isHttps =
      ((req.headers["x-forwarded-proto"] as string) || "").includes("https") ||
      Boolean((req.socket as any)?.encrypted);

    const candidates = isHttps
      ? [
          "__Secure-authjs.session-token",
          "__Secure-next-auth.session-token",
        ]
      : ["authjs.session-token", "next-auth.session-token"];

    for (const name of candidates) {
      const token = cookies.get(name);
      if (!token) continue;
      const userId = await this.verifySessionToken(token, name);
      if (userId) return userId;
    }
    return null;
  }
}

// Singleton
let wsManagerInstance: WSManager | null = null;

export function getWSManager(): WSManager {
  if (!wsManagerInstance) {
    wsManagerInstance = new WSManager();
  }
  return wsManagerInstance;
}