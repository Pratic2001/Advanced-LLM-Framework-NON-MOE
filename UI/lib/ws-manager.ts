/**
 * WebSocket Manager
 * Manages WebSocket connections for real-time job updates
 */

import { WebSocketServer, WebSocket } from "ws";
import { IncomingMessage } from "http";
import { JobManager } from "./job-manager";
import type { JobEvent } from "./job-manager";

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
      const client: WsClient = {
        ws,
        userId: "",
        subscriptions: new Set(),
        id: clientId,
      };

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
          }
        } catch (err) {
          console.error("[WSManager] Invalid message:", err);
        }
      });

      ws.on("close", () => {
        this.clients.delete(clientId);
      });

      ws.on("error", () => {
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
    for (const client of this.clients.values()) {
      if (client.subscriptions.has(channel)) {
        try {
          client.ws.send(JSON.stringify(event));
        } catch {
          // Client disconnected
        }
      }
    }
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
}

// Singleton
let wsManagerInstance: WSManager | null = null;

export function getWSManager(): WSManager {
  if (!wsManagerInstance) {
    wsManagerInstance = new WSManager();
  }
  return wsManagerInstance;
}
