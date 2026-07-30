"use client";

import { useEffect, useRef, useState, useCallback } from "react";

type MessageHandler = (data: any) => void;

interface UseWebSocketOptions {
  onMessage?: MessageHandler;
  onStatus?: MessageHandler;
  onMetric?: MessageHandler;
  onLog?: MessageHandler;
  onError?: MessageHandler;
  autoConnect?: boolean;
}

export function useWebSocket(
  jobId?: string,
  options: UseWebSocketOptions = {}
) {
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/api/ws`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);

      // Subscribe to job updates
      if (jobId) {
        ws.send(JSON.stringify({ type: "subscribe", jobId }));
      }
      ws.send(JSON.stringify({ type: "subscribe", channel: "dashboard" }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setLastMessage(data);

        switch (data.type) {
          case "status":
            options.onStatus?.(data);
            break;
          case "metric":
            options.onMetric?.(data);
            break;
          case "log":
            options.onLog?.(data);
            break;
          case "error":
            options.onError?.(data);
            break;
          default:
            options.onMessage?.(data);
        }
      } catch {}
    };

    ws.onclose = () => {
      setConnected(false);
      // Reconnect after delay
      reconnectTimeoutRef.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [jobId, options]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnected(false);
  }, []);

  const send = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  useEffect(() => {
    connect();
    return disconnect;
  }, [connect, disconnect]);

  return { connected, lastMessage, send, disconnect, reconnect: connect };
}
