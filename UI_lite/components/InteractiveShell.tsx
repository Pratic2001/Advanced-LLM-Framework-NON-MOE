/**
 * InteractiveShell
 *
 * xterm.js backed bash shell that runs on the server. The transport is
 * SSE (server → client, PTY output) + HTTP POST (client → server, PTY
 * input / resize / close).
 *
 * Why not WebSocket? Tailscale Funnel aggressively kills WebSocket upgrade
 * requests within milliseconds of opening, so a plain public HTTPS tunnel
 * can't carry them. SSE rides on the same long-lived HTTP semantics as a
 * normal `fetch`, which Funnel lets through.
 *
 * Wire shape:
 *   POST /api/shell/open   { cols, rows }                → { sessionId, pid }
 *   GET  /api/shell/stream?sessionId=…                   → SSE: opened / data / exit
 *   POST /api/shell/input  { sessionId, data }           → 204
 *   POST /api/shell/resize { sessionId, cols, rows }     → 204
 *   POST /api/shell/close  { sessionId }                 → 204 (idempotent)
 *
 * The session cookie is automatically attached to all of these — no extra
 * handshake step, since `auth()` on the server validates per request.
 */
import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import { Power, Plug, RefreshCw, Terminal as TerminalIcon, AlertTriangle } from "lucide-react";
import { api } from "@/lib/base-path";

import "@xterm/xterm/css/xterm.css";

type Tone = "cyan" | "blue";

const TONE_CLASSES: Record<
  Tone,
  { border: string; glow: string; text: string }
> = {
  cyan: {
    border: "border-neon-cyan/40",
    glow: "shadow-[0_0_18px_-6px_rgba(34,211,238,0.35)]",
    text: "text-neon-cyan",
  },
  blue: {
    border: "border-neon-blue/40",
    glow: "shadow-[0_0_18px_-6px_rgba(59,130,246,0.35)]",
    text: "text-neon-blue",
  },
};

type ShellState =
  | "idle"
  | "connecting"
  | "connected"
  | "disconnected"
  | "error";

const ANSI = {
  reset: "\x1b[0m",
  dim: "\x1b[2m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  red: "\x1b[31m",
};

export interface InteractiveShellHandle {
  pasteIntoPrompt: (text: string) => void;
  connect: () => void;
  disconnect: () => void;
}

interface InteractiveShellProps {
  subtitle?: string;
  tone?: Tone;
}

export const InteractiveShell = forwardRef<
  InteractiveShellHandle,
  InteractiveShellProps
>(function InteractiveShell({ subtitle, tone }, ref) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const termRef = useRef<any>(null);
  const fitAddonRef = useRef<any>(null);
  const sessionIdRef = useRef<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  // Track whether the PTY has actually opened so we can distinguish
  // "transport closed before opening" (auth/connect failure) from
  // "shell exited on its own" (normal exit).
  const openedRef = useRef<boolean>(false);
  // Tracks whether the user explicitly disconnected — if so, don't
  // try to auto-reconnect after the EventSource closes.
  const intentionallyClosedRef = useRef<boolean>(false);
  // Holds keystrokes that arrived while we were waiting for `opened`.
  // Drained once the SSE stream reports the PTY is up.
  const pendingInputRef = useRef<string>("");
  // Coalesces rapid keystrokes into one POST every INPUT_BATCH_MS.
  const inputBufferRef = useRef<string>("");
  const inputFlushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const colsRef = useRef<number>(80);
  const rowsRef = useRef<number>(24);
  const connectRef = useRef<() => void>(() => {});

  const [state, _setState] = useState<ShellState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [pid, setPid] = useState<number | null>(null);

  const setState = useCallback(
    (next: ShellState | ((prev: ShellState) => ShellState)) => {
      _setState((prev) => {
        const resolved =
          typeof next === "function"
            ? (next as (p: ShellState) => ShellState)(prev)
            : next;
        return resolved;
      });
    },
    []
  );

  // Mirror of `state` readable synchronously inside async callbacks. Kept in
  // a separate effect-free ref so closures don't capture stale values.
  const stateRef = useRef<ShellState>("idle");
  stateRef.current = state;

  const accent = TONE_CLASSES[tone ?? "cyan"];

  /**
   * POST helper used by all shell endpoints. Uses `fetch` with `keepalive`
   * so requests fire reliably even when the page is navigating away, and
   * swallows 204s (the success response on close/input/resize).
   */
  const postShell = useCallback(
    async (path: string, body: unknown): Promise<boolean> => {
      try {
        const res = await fetch(path, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          keepalive: true,
        });
        if (res.ok || res.status === 204) return true;
        // Surface server-side error messages when present.
        try {
          const errBody = await res.json();
          const message =
            errBody?.error || `Request failed (${res.status})`;
          if (stateRef.current === "connecting") {
            setError(message);
            setState("error");
          }
        } catch {
          if (stateRef.current === "connecting") {
            setError(`Request failed (${res.status})`);
            setState("error");
          }
        }
        return false;
      } catch {
        // Network-level failure — leave the current state alone; the
        // EventSource's `onerror` will report it more accurately.
        return false;
      }
    },
    []
  );

  /**
   * Flush buffered keystrokes to the server. We coalesce very rapid input
   * (a paste of several KB) into one POST so we don't drown the server in
   * tiny requests — but the timer is short (50ms) so it still feels
   * instant to the user.
   */
  const INPUT_BATCH_MS = 50;

  const flushInputBuffer = useCallback(() => {
    const sessionId = sessionIdRef.current;
    const buffered = inputBufferRef.current;
    inputBufferRef.current = "";
    if (inputFlushTimerRef.current) {
      clearTimeout(inputFlushTimerRef.current);
      inputFlushTimerRef.current = null;
    }
    if (!sessionId || buffered.length === 0) return;
    void postShell(api("/api/shell/input"), { sessionId, data: buffered });
  }, [postShell]);

  const queueInput = useCallback(
    (data: string) => {
      // If the PTY isn't open yet, hold the input until it is — otherwise
      // the server will return 404 and the keystrokes would be lost.
      if (!openedRef.current || !sessionIdRef.current) {
        pendingInputRef.current += data;
        return;
      }
      inputBufferRef.current += data;
      if (inputFlushTimerRef.current) return;
      inputFlushTimerRef.current = setTimeout(() => {
        flushInputBuffer();
      }, INPUT_BATCH_MS);
    },
    [flushInputBuffer]
  );

  /**
   * Tear down the current session: close the EventSource, kill the PTY
   * (idempotent on the server), and clear local refs. Used by both
   * `disconnect` and the cleanup path on unmount.
   */
  const tearDownTransport = useCallback(() => {
    const sessionId = sessionIdRef.current;
    if (eventSourceRef.current) {
      try {
        eventSourceRef.current.close();
      } catch {}
      eventSourceRef.current = null;
    }
    if (sessionId) {
      // Fire-and-forget; the server treats this as idempotent.
      void postShell(api("/api/shell/close"), { sessionId });
      sessionIdRef.current = null;
    }
    openedRef.current = false;
    pendingInputRef.current = "";
    if (inputFlushTimerRef.current) {
      clearTimeout(inputFlushTimerRef.current);
      inputFlushTimerRef.current = null;
    }
    inputBufferRef.current = "";
  }, [postShell]);

  /**
   * Open a new PTY on the server, then attach an EventSource to stream
   * its output. Used for the initial connection and for reconnects.
   */
  const connect = useCallback(async () => {
    if (stateRef.current === "connecting" || stateRef.current === "connected") {
      return;
    }
    if (!termRef.current) {
      setState("error");
      setError("Terminal failed to initialize");
      return;
    }

    const term = termRef.current;
    setError(null);
    intentionallyClosedRef.current = false;
    setState("connecting");
    term.writeln(`${ANSI.dim}# Connecting to interactive shell...${ANSI.reset}`);

    // 1. Open a PTY session on the server.
    let sessionId: string;
    try {
      const res = await fetch(api("/api/shell/open"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cols: colsRef.current,
          rows: rowsRef.current,
        }),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => null);
        const message = errBody?.error || `Open failed (${res.status})`;
        term.writeln(`${ANSI.red}# ${message}${ANSI.reset}`);
        setError(message);
        setState("error");
        return;
      }
      const data = await res.json();
      sessionId = data.sessionId;
      if (!sessionId) {
        const message = "Server returned no sessionId";
        term.writeln(`${ANSI.red}# ${message}${ANSI.reset}`);
        setError(message);
        setState("error");
        return;
      }
      sessionIdRef.current = sessionId;
    } catch (err: any) {
      const message = err?.message || "Failed to reach /api/shell/open";
      term.writeln(`${ANSI.red}# ${message}${ANSI.reset}`);
      setError(message);
      setState("error");
      return;
    }

    // 2. Open the SSE stream for this session.
    const es = new EventSource(
      api(`/api/shell/stream?sessionId=${encodeURIComponent(sessionId)}`)
    );
    eventSourceRef.current = es;

    es.addEventListener("opened", (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data);
        openedRef.current = true;
        setPid(typeof payload?.pid === "number" ? payload.pid : null);
        setState("connected");
        term.writeln(
          `${ANSI.green}# Connected (pid=${payload?.pid ?? "?"}${
            typeof payload?.activePtys === "number"
              ? `, ${payload.activePtys} active shells`
              : ""
          })${ANSI.reset}`
        );
        // Drain any keystrokes that arrived while we were waiting.
        const pending = pendingInputRef.current;
        if (pending) {
          pendingInputRef.current = "";
          queueInput(pending);
        }
      } catch {
        // Malformed payload — treat as error so the user sees a message
        // instead of a silently broken terminal.
        setState("error");
        setError("Malformed 'opened' event from server");
      }
    });

    es.addEventListener("data", (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data);
        if (typeof payload?.data === "string") {
          term.write(payload.data);
        }
      } catch {
        // Ignore unparseable data frames — they're transient and the next
        // frame will usually still be useful.
      }
    });

    es.addEventListener("exit", (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data);
        openedRef.current = false;
        term.writeln(
          `\r\n${ANSI.yellow}# Shell exited (code=${payload?.exitCode ?? "?"}, signal=${
            payload?.signal ?? 0
          })${ANSI.reset}`
        );
        // If we hadn't marked this disconnect as intentional, leave the
        // state machine in "disconnected" so the user can reconnect.
        setState((prev) =>
          prev === "error" ? prev : intentionallyClosedRef.current ? prev : "disconnected"
        );
      } catch {
        openedRef.current = false;
        setState((prev) => (prev === "error" ? prev : "disconnected"));
      }
      // The EventSource will close on its own after the server's `close()`
      // call in the stream route — let it.
    });

    es.addEventListener("error", () => {
      // EventSource auto-reconnects internally on transient errors, but
      // closes the connection if it gets a non-recoverable status. We
      // treat an outright close as a transport failure unless the user
      // asked for it.
      if (intentionallyClosedRef.current) return;
      openedRef.current = false;
      setState((prev) => {
        if (prev === "connected") {
          term.writeln(
            `\r\n${ANSI.yellow}# Connection lost. Reconnecting...${ANSI.reset}`
          );
          return "connecting";
        }
        if (prev === "connecting") {
          term.writeln(
            `${ANSI.red}# Connection closed before shell opened${ANSI.reset}`
          );
          setError("Connection closed before shell opened");
          return "error";
        }
        return prev;
      });
    });
  }, [queueInput]);

  /** Tidy disconnect: kill the PTY, leave the terminal buffer visible. */
  const disconnect = useCallback(() => {
    intentionallyClosedRef.current = true;
    tearDownTransport();
    setPid(null);
    setState("disconnected");
    if (termRef.current) {
      termRef.current.writeln(`\r\n${ANSI.yellow}# Disconnected.${ANSI.reset}`);
    }
  }, [tearDownTransport]);

  /** Tear down the whole terminal (used on unmount). */
  const dispose = useCallback(() => {
    tearDownTransport();
    if (termRef.current) {
      try {
        termRef.current.dispose();
      } catch {}
      termRef.current = null;
    }
    fitAddonRef.current = null;
    setPid(null);
  }, [tearDownTransport]);

  useImperativeHandle(
    ref,
    () => ({
      pasteIntoPrompt: (text: string) => {
        if (!termRef.current) return;
        // If the PTY isn't open yet, just queue the text — once `opened`
        // arrives, `connect` will drain pendingInputRef automatically.
        queueInput(text);
        if (stateRef.current !== "connected") {
          connectRef.current();
        }
      },
      connect,
      disconnect,
    }),
    [connect, disconnect, queueInput]
  );

  // Keep the imperative handle's `connect` callback fresh so chips and
  // external buttons always invoke the latest version.
  connectRef.current = connect;

  /** Mount xterm.js once on first render. */
  const mountTerminal = useCallback(async () => {
    if (termRef.current || !containerRef.current) return;

    const [{ Terminal }, { FitAddon }] = await Promise.all([
      import("@xterm/xterm"),
      import("@xterm/addon-fit"),
    ]);

    const fit = new FitAddon();
    const term = new Terminal({
      fontFamily:
        '"JetBrains Mono", "Fira Code", "Cascadia Code", ui-monospace, monospace',
      fontSize: 13,
      lineHeight: 1.2,
      cursorBlink: true,
      convertEol: true,
      scrollback: 5000,
      theme: {
        background: "#000000",
        foreground: "#e5e7eb",
        cursor: "#22d3ee",
        selectionBackground: "#0e7490",
      },
    });
    term.loadAddon(fit);
    term.open(containerRef.current);
    fit.fit();

    termRef.current = term;
    fitAddonRef.current = fit;

    colsRef.current = term.cols;
    rowsRef.current = term.rows;

    // Forward keystrokes to the PTY (queued; flushed in batches).
    term.onData((data: string) => {
      queueInput(data);
    });

    // Initial welcome — gives the user something to see before they hit
    // Connect.
    term.writeln(`${ANSI.dim}# Interactive shell — press Connect.${ANSI.reset}`);
  }, [queueInput]);

  // Mount the terminal on first render, dispose on unmount.
  useEffect(() => {
    mountTerminal();
    return () => {
      dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Resize the PTY whenever the visible size of the terminal changes.
  // We avoid sending a resize on the very first paint — xterm fires one
  // synchronously after open() before the user has had a chance to size
  // anything, and the server's `getSession` check requires the sessionId
  // to be valid, which it isn't until after `connect()`.
  useEffect(() => {
    const handler = () => {
      if (!termRef.current || !fitAddonRef.current) return;
      try {
        fitAddonRef.current.fit();
      } catch {}
      colsRef.current = termRef.current.cols;
      rowsRef.current = termRef.current.rows;
      const sessionId = sessionIdRef.current;
      if (sessionId && openedRef.current) {
        void postShell(api("/api/shell/resize"), {
          sessionId,
          cols: colsRef.current,
          rows: rowsRef.current,
        });
      }
    };
    window.addEventListener("resize", handler);
    return () => window.removeEventListener("resize", handler);
  }, [postShell]);

  const isConnected = state === "connected";
  const isBusy = state === "connecting";

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
          <ShellStateBadge state={state} pid={pid} tone={accent} />
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
  tone: { text: string };
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
      className: `${tone.text} bg-accent/30 border border-current/30`,
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