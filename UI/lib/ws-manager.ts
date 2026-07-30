/**
 * WebSocket Manager
 * Manages WebSocket connections for real-time job updates
 */

import { WebSocketServer, WebSocket } from "ws";
import { IncomingMessage } from "http";
import { JobManager } from "./job-manager";
import type { JobEvent } from "./job-manager";
import {
  spawnShellPty,
  getActivePtyCount,
  type ShellPty,
} from "./pty-manager";
import { decode } from "next-auth/jwt";
import { env as appEnv } from "./env";

interface WsClient {
  ws: WebSocket;
  userId: string;
  subscriptions: Set<string>; // jobIds or "dashboard"
  id: string;
  /** Live PTY for this client, if the interactive shell is currently open. */
  pty: ShellPty | null;
  /** Authenticated for shell access. */
  shellAuthed: boolean;
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

    this.wss.on("connection", async (ws: WebSocket, req: IncomingMessage) => {
      const clientId = this.generateId();

      // Authenticate from the session cookie on the upgrade request. Auth.js
      // v5 sets `httpOnly: true`, so the browser can't read the JWT from JS;
      // it must be verified server-side from `req.headers.cookie`. Same-origin
      // WS upgrades forward cookies automatically.
      const userId = await this.authenticateFromCookies(req);
      const client: WsClient = {
        ws,
        userId: userId ?? "",
        subscriptions: new Set(),
        id: clientId,
        pty: null,
        // Shell access is gated by userId being non-empty; the legacy
        // `shellAuthed` flag is kept for backwards compat but is no longer
        // the gate — auth now happens at connection time.
        shellAuthed: Boolean(userId),
      };

      if (!userId) {
        // No valid session cookie — refuse the connection so the client
        // gets a clear error instead of silently rejecting every pty:open.
        this.sendToClient(clientId, {
          type: "shellAuth:err",
          error: "Not signed in",
        });
        try {
          ws.close(1008, "Not authenticated");
        } catch {}
        return;
      }

      this.clients.set(clientId, client);

      ws.on("message", (data: Buffer) => {
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

            // ── Interactive shell ─────────────────────────────────────
            // Auth gate happens at WS-connection time (see above), so the
            // client no longer needs to send a `shellAuth` round-trip. We
            // keep the case as a no-op for backwards compatibility with any
            // older client that still sends it.
            case "shellAuth":
              break;

            case "pty:open":
              if (!client.shellAuthed) {
                this.sendToClient(clientId, {
                  type: "pty:err",
                  error: "Not authenticated",
                });
                break;
              }
              if (client.pty) {
                // Already open — ignore duplicate open requests.
                break;
              }
              try {
                const pty = spawnShellPty({
                  cols: message.cols || 80,
                  rows: message.rows || 24,
                  userId: client.userId,
                });
                client.pty = pty;

                pty.onData((chunk) => {
                  this.sendToClient(clientId, {
                    type: "pty:data",
                    data: chunk,
                  });
                });
                pty.onExit(({ exitCode, signal }) => {
                  this.sendToClient(clientId, {
                    type: "pty:exit",
                    exitCode,
                    signal,
                  });
                  client.pty = null;
                });

                this.sendToClient(clientId, {
                  type: "pty:opened",
                  pid: pty.pid,
                  activePtys: getActivePtyCount(),
                });
              } catch (err: any) {
                this.sendToClient(clientId, {
                  type: "pty:err",
                  error: err?.message || "Failed to spawn PTY",
                });
              }
              break;

            case "pty:input":
              if (client.pty && typeof message.data === "string") {
                client.pty.write(message.data);
              }
              break;

            case "pty:resize":
              if (
                client.pty &&
                typeof message.cols === "number" &&
                typeof message.rows === "number"
              ) {
                client.pty.resize(message.cols, message.rows);
              }
              break;

            case "pty:close":
              if (client.pty) {
                client.pty.kill();
                client.pty = null;
              }
              break;
          }
        } catch (err) {
          console.error("[WSManager] Invalid message:", err);
        }
      });

      ws.on("close", () => {
        if (client.pty) {
          try {
            client.pty.kill();
          } catch {}
          client.pty = null;
        }
        this.clients.delete(clientId);
      });

      ws.on("error", () => {
        if (client.pty) {
          try {
            client.pty.kill();
          } catch {}
          client.pty = null;
        }
        this.clients.delete(clientId);
      });

      // Send connection confirmation
      ws.send(
        JSON.stringify({
          type: "connected",
          clientId,
          timestamp: new Date().toISOString(),
        })
      );
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
   * Send a message to a specific client
   */
  sendToClient(clientId: string, data: any): void {
    const client = this.clients.get(clientId);
    if (client?.ws.readyState === WebSocket.OPEN) {
      client.ws.send(JSON.stringify(data));
    }
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
