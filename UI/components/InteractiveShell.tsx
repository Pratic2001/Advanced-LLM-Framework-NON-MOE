"use client";

/**
 * InteractiveShell — full bash shell embedded in the page.
 *
 * browser xterm.js <-- text frames --> server WS --stdin--> node-pty (bash)
 *                                                            |
 *                              stdout/stderr <----------------'
 *
 * xterm.js is mounted with `disableStdin: false` so user typing is forwarded
 * to the PTY. The PTY echoes back through `pty:data` frames so we render the
 * same content the user sees in a real terminal.
 *
 * Auth: the WS server reads the Auth.js session cookie (which is httpOnly,
 * so JavaScript can't access it) directly from the upgrade request and
 * authenticates the connection at handshake time. The client never touches
 * the JWT — it just sends `pty:open` after the socket opens.
 *
 * Reconnect: PTY is intentionally NOT auto-reconnected — losing a connection
 * kills the bash process. The user sees a "disconnected" banner and a
 * Reconnect button which starts a fresh shell.
 */

import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
  useCallback,
} from "react";
import {
  Terminal as TerminalIcon,
  Plug,
  RefreshCw,
  AlertTriangle,
  Power,
} from "lucide-react";

type ShellState = "idle" | "connecting" | "connected" | "disconnected" | "error";

export interface InteractiveShellHandle {
  /**
   * Paste text into the terminal's input buffer (does NOT auto-execute).
   * Useful for the example-command chips: click → user sees the command
   * sitting in the prompt, then presses Enter.
   */
  pasteIntoPrompt: (text: string) => void;
  /** Connect if not already connected. Resolves once `pty:opened` is received. */
  connect: () => void;
}

interface InteractiveShellProps {
  /** Optional subtitle shown in the header (e.g. "scoped to REPO_ROOT"). */
  subtitle?: string;
  /** Accent color — cyan for torch, blue for deepspeed. */
  tone: "cyan" | "blue";
}

const TONE_CLASSES = {
  cyan: {
    text: "text-neon-cyan",
    border: "border-neon-cyan/30",
    glow: "neon-glow-cyan",
    headerBorder: "border-neon-cyan/20",
  },
  blue: {
    text: "text-neon-blue",
    border: "border-neon-blue/30",
    glow: "neon-glow-blue",
    headerBorder: "border-neon-blue/20",
  },
} as const;

const ANSI = {
  reset: "\x1b[0m",
  dim: "\x1b[90m",
  red: "\x1b[31m",
  yellow: "\x1b[33m",
  green: "\x1b[32m",
};

export const InteractiveShell = forwardRef<
  InteractiveShellHandle,
  InteractiveShellProps
>(function InteractiveShell({ subtitle, tone }, ref) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const termRef = useRef<any>(null);
  const fitAddonRef = useRef<any>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const colsRef = useRef<number>(80);
  const rowsRef = useRef<number>(24);
  const connectRef = useRef<() => void>(() => {});
  // Mirror of `state` that the connect handler can read synchronously
  // without depending on the closure-captured React state (which would be
  // stale by the time async ws callbacks fire).
  const stateRef = useRef<ShellState>("idle");

  const [state, _setState] = useState<ShellState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [pid, setPid] = useState<number | null>(null);

  /**
   * Wrapper that mirrors every state change into `stateRef` so async
   * WebSocket callbacks can read the *current* connection state without
   * depending on a (potentially stale) closure-captured `state`.
   */
  const setState = useCallback((next: ShellState | ((prev: ShellState) => ShellState)) => {
    _setState((prev) => {
      const resolved =
        typeof next === "function" ? (next as (p: ShellState) => ShellState)(prev) : next;
      stateRef.current = resolved;
      return resolved;
    });
  }, []);

  const accent = TONE_CLASSES[tone];

  // Expose an imperative API so the example-command chips can paste into
  // the terminal's prompt (`pasteIntoPrompt`) and so the parent can trigger
  // a connection from outside (e.g. a "Connect" footer button).
  useImperativeHandle(
    ref,
    () => ({
      pasteIntoPrompt: (text: string) => {
        if (!termRef.current) return;
        if (wsRef.current?.readyState !== WebSocket.OPEN) {
          // Auto-connect, then paste once the shell is ready.
          const tryPasteWhenReady = () => {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify({ type: "pty:input", data: text }));
            } else {
              setTimeout(tryPasteWhenReady, 100);
            }
          };
          connectRef.current();
          tryPasteWhenReady();
          return;
        }
        // Send to the PTY directly so the shell receives the same bytes the
        // user would have typed. The PTY will echo it back through `pty:data`,
        // so we don't write to xterm ourselves — that avoids a double-render.
        wsRef.current.send(JSON.stringify({ type: "pty:input", data: text }));
      },
      connect: () => connectRef.current(),
    }),
    []
  );

  /** Mount xterm.js once. The terminal is reused across reconnects. */
  const mountTerminal = useCallback(async () => {
    if (termRef.current || !containerRef.current) return;

    const [{ Terminal }, { FitAddon }] = await Promise.all([
      import("@xterm/xterm"),
      import("@xterm/addon-fit"),
    ]);
    await import("@xterm/xterm/css/xterm.css");

    if (!containerRef.current) return;

    const term = new Terminal({
      convertEol: true,
      cursorBlink: true,
      fontFamily: '"JetBrains Mono", "Fira Code", Menlo, Consolas, monospace',
      fontSize: 12,
      lineHeight: 1.2,
      scrollback: 10000,
      theme: {
        background: "#000000",
        foreground: "#e5e7eb",
        cursor: tone === "cyan" ? "#00f0ff" : "#0088ff",
        selectionBackground: "rgba(0,240,255,0.25)",
      },
    });

    const fitAddon = new FitAddon();
    term.loadAddon(fitAddon);
    term.open(containerRef.current);

    // Forward typed characters to the PTY via the WebSocket. xterm normalizes
    // keys into printable strings and arrow/control sequences through `onData`.
    term.onData((data: string) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "pty:input", data }));
      }
    });

    // First paint. `fit()` may throw if the container has zero size; the
    // ResizeObserver below retries.
    try {
      fitAddon.fit();
      colsRef.current = term.cols;
      rowsRef.current = term.rows;
    } catch {
      // ignore
    }

    termRef.current = term;
    fitAddonRef.current = fitAddon;

    const observer = new ResizeObserver(() => {
      try {
        fitAddon.fit();
        const cols = term.cols;
        const rows = term.rows;
        if (
          (cols !== colsRef.current || rows !== rowsRef.current) &&
          wsRef.current?.readyState === WebSocket.OPEN
        ) {
          colsRef.current = cols;
          rowsRef.current = rows;
          wsRef.current.send(
            JSON.stringify({ type: "pty:resize", cols, rows })
          );
        }
      } catch {
        // Ignore transient fit failures during layout shifts.
      }
    });
    observer.observe(containerRef.current);

    return () => observer.disconnect();
  }, [tone]);

  /** Connect to the WS, then open the PTY (server authenticates from cookies). */
  const connect = useCallback(async () => {
    // Read the ref, not the closure-captured `state` — by the time a second
    // click handler fires, `state` may be stale and a second ws would be
    // opened in parallel.
    const current = stateRef.current;
    if (current === "connecting" || current === "connected") return;

    setError(null);
    setState("connecting");

    // Mount xterm on first connect.
    if (!termRef.current) {
      await mountTerminal();
    }

    if (!termRef.current) {
      setState("error");
      setError("Terminal failed to initialize");
      return;
    }

    const term = termRef.current;
    term.writeln(`${ANSI.dim}# Connecting to interactive shell...${ANSI.reset}`);

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/api/ws`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    let opened = false;

    ws.onopen = () => {
      // The server reads the Auth.js session cookie from the upgrade request
      // at connection time. Nothing for the client to send for auth.
      // Subscribe to dashboard broadcasts (job updates) anyway.
      ws.send(JSON.stringify({ type: "subscribe", channel: "dashboard" }));
      // Open the PTY immediately.
      ws.send(
        JSON.stringify({
          type: "pty:open",
          cols: colsRef.current,
          rows: rowsRef.current,
        })
      );
    };

    ws.onmessage = (event) => {
      let msg: any;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }

      switch (msg.type) {
        case "pty:opened":
          opened = true;
          setPid(msg.pid);
          setState("connected");
          term.writeln(
            `${ANSI.green}# Connected (pid=${msg.pid}, ${msg.activePtys} active shells)${ANSI.reset}`
          );
          break;

        case "pty:data":
          if (typeof msg.data === "string") {
            term.write(msg.data);
          }
          break;

        case "pty:exit":
          setState("disconnected");
          term.writeln(
            `\r\n${ANSI.yellow}# Shell exited (code=${msg.exitCode}, signal=${msg.signal ?? 0})${ANSI.reset}`
          );
          break;

        case "pty:err":
          setState("error");
          setError(msg.error || "PTY error");
          term.writeln(`${ANSI.red}# ${msg.error || "PTY error"}${ANSI.reset}`);
          break;
      }
    };

    ws.onclose = (event) => {
      if (opened) {
        setState((prev) => {
          if (prev === "disconnected") return prev;
          if (termRef.current) {
            termRef.current.writeln(
              `\r\n${ANSI.yellow}# Connection closed.${ANSI.reset}`
            );
          }
          return "disconnected";
        });
      } else {
        // Closed before the PTY ever opened — most likely an auth failure
        // (the server closes with 1008 "Not authenticated" when the session
        // cookie is missing/invalid). Surface a clear error instead of
        // leaving the user stuck in "connecting".
        const reason = event.reason || "Connection closed before shell opened";
        setState((prev) => {
          if (prev === "connected" || prev === "disconnected") return prev;
          if (termRef.current) {
            termRef.current.writeln(
              `${ANSI.red}# ${reason}${ANSI.reset}`
            );
          }
          setError(reason);
          return "error";
        });
      }
      wsRef.current = null;
    };

    ws.onerror = () => {
      // Functional setter — reads latest state without depending on the
      // closure-captured value (which can be stale by the time onerror fires).
      setState((prev) => {
        if (prev !== "connecting") return prev;
        setError("WebSocket connection failed");
        return "error";
      });
    };
  }, [mountTerminal]);

  /** Tidy disconnect: close WS, kill PTY, leave terminal buffer visible. */
  const disconnect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "pty:close" }));
      try {
        wsRef.current.close();
      } catch {}
    }
    wsRef.current = null;
    setPid(null);
    setState("disconnected");
    if (termRef.current) {
      termRef.current.writeln(
        `\r\n${ANSI.yellow}# Disconnected.${ANSI.reset}`
      );
    }
  }, []);

  /** Tear down the whole terminal (used on unmount). */
  const dispose = useCallback(() => {
    if (wsRef.current) {
      try {
        wsRef.current.send(JSON.stringify({ type: "pty:close" }));
      } catch {}
      try {
        wsRef.current.close();
      } catch {}
      wsRef.current = null;
    }
    if (termRef.current) {
      try {
        termRef.current.dispose();
      } catch {}
      termRef.current = null;
    }
    fitAddonRef.current = null;
    setPid(null);
  }, []);

  // Mount terminal on first render (it persists across reconnects).
  useEffect(() => {
    mountTerminal();
    return () => {
      dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const isConnected = state === "connected";
  const isBusy = state === "connecting";

  // Keep the imperative handle's `connect` callback fresh so the parent can
  // trigger a connection from chips / external buttons.
  connectRef.current = connect;

  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold flex items-center gap-2">
          <TerminalIcon className={`w-4 h-4 ${accent.text}`} />
          Interactive Shell
          <span className="text-xs font-normal text-muted-foreground">
            {subtitle ?? "bash · scoped to REPO_ROOT"}
          </span>
        </h2>
        <div className="flex items-center gap-2">
          <ShellStateBadge state={state} pid={pid} tone={tone} />
          {isConnected ? (
            <button
              onClick={disconnect}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium transition-all bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20`}
            >
              <Power className="w-3 h-3" />
              Disconnect
            </button>
          ) : (
            <button
              onClick={connect}
              disabled={isBusy}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium transition-all ${
                isBusy
                  ? "bg-accent/20 text-muted-foreground/50 cursor-not-allowed"
                  : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/20"
              }`}
            >
              {state === "disconnected" ? (
                <RefreshCw className="w-3 h-3" />
              ) : (
                <Plug className="w-3 h-3" />
              )}
              {isBusy ? "Connecting…" : state === "disconnected" ? "Reconnect" : "Connect"}
            </button>
          )}
        </div>
      </div>

      <div
        className={`glass rounded-xl border ${accent.border} ${accent.glow} overflow-hidden`}
      >
        <div
          ref={containerRef}
          className="h-96 w-full bg-black"
          style={{ minHeight: "24rem" }}
        />
      </div>

      {error && (
        <div className="flex items-start gap-2 glass rounded-lg p-3 border border-red-500/30 bg-red-500/5">
          <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
          <p className="text-xs text-red-400">{error}</p>
        </div>
      )}

      {state === "idle" && (
        <p className="text-xs text-muted-foreground">
          Click <span className={accent.text}>Connect</span> to open a real bash
          shell. It runs on the same machine as the dev server, with the
          working directory locked to <span className="font-mono">REPO_ROOT</span>.
          Disconnect (or close the tab) to terminate the shell.
        </p>
      )}
    </section>
  );
});

function ShellStateBadge({
  state,
  pid,
  tone,
}: {
  state: ShellState;
  pid: number | null;
  tone: "cyan" | "blue";
}) {
  const map: Record<ShellState, { label: string; className: string; pulsing?: boolean }> = {
    idle: {
      label: "idle",
      className: "bg-accent/30 text-muted-foreground border border-border",
    },
    connecting: {
      label: "connecting",
      className: "bg-yellow-500/10 text-yellow-400 border border-yellow-500/30",
      pulsing: true,
    },
    connected: {
      label: pid ? `running · pid ${pid}` : "running",
      className:
        tone === "cyan"
          ? "bg-neon-cyan/10 text-neon-cyan border border-neon-cyan/30"
          : "bg-neon-blue/10 text-neon-blue border border-neon-blue/30",
      pulsing: true,
    },
    disconnected: {
      label: "disconnected",
      className: "bg-orange-500/10 text-orange-400 border border-orange-500/30",
    },
    error: {
      label: "error",
      className: "bg-red-500/10 text-red-400 border border-red-500/30",
    },
  };
  const m = map[state];
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider ${m.className}`}
    >
      {m.pulsing && (
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-current opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-current" />
        </span>
      )}
      {m.label}
    </span>
  );
}
