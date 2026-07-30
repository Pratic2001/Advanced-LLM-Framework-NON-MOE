"use client";

/**
 * TerminalPanel — owns the xterm.js Terminal instance, subscribes to the
 * job's WebSocket log/status stream, and exposes a Stop button via
 * useStopJob. The wrapper component (IntegratedTerminal) handles styling,
 * the header bar, and SSR safety.
 */

import { useEffect, useRef, useState } from "react";
import { useWebSocket } from "@/hooks/use-websocket";
import { useStopJob } from "@/hooks/use-jobs";
import {
  formatStatusBadge,
  isTerminalStatus,
  type JobStatus,
} from "@/lib/terminal-helpers";

interface TerminalPanelProps {
  jobId: string;
  /** Backend controls the accent color (cyan for torch, blue for deepspeed). */
  tone: "cyan" | "blue";
}

const TONE_CLASSES = {
  cyan: {
    border: "border-neon-cyan/30",
    text: "text-neon-cyan",
    glow: "neon-glow-cyan",
    headerBorder: "border-neon-cyan/20",
  },
  blue: {
    border: "border-neon-blue/30",
    text: "text-neon-blue",
    glow: "neon-glow-blue",
    headerBorder: "border-neon-blue/20",
  },
} as const;

export function TerminalPanel({ jobId, tone }: TerminalPanelProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const termRef = useRef<any>(null);
  const fitAddonRef = useRef<any>(null);
  const writeChunk = (text: string) => {
    if (termRef.current) termRef.current.write(text);
  };

  const [status, setStatus] = useState<JobStatus>("QUEUED");
  const [connected, setConnected] = useState(false);
  const [ready, setReady] = useState(false);
  const { stop, stopping } = useStopJob();

  // Mount xterm once on the client. Imported lazily to keep DOM-bound code
  // out of the SSR bundle and to avoid pulling xterm CSS into every page.
  useEffect(() => {
    let disposed = false;
    let observer: ResizeObserver | null = null;

    (async () => {
      const [{ Terminal }, { FitAddon }] = await Promise.all([
        import("@xterm/xterm"),
        import("@xterm/addon-fit"),
      ]);

      // Dynamic CSS import — keeps the file out of the global stylesheet so
      // xterm styles only ship on pages that mount the terminal.
      await import("@xterm/xterm/css/xterm.css");

      if (disposed || !containerRef.current) return;

      const term = new Terminal({
        convertEol: true,
        cursorBlink: true,
        disableStdin: true,
        fontFamily: '"JetBrains Mono", "Fira Code", Menlo, Consolas, monospace',
        fontSize: 12,
        lineHeight: 1.2,
        scrollback: 10000,
        theme: {
          background: "#000000",
          foreground: "#e5e7eb",
          cursor: TONE_CLASSES[tone].text.includes("cyan") ? "#00f0ff" : "#0088ff",
          selectionBackground: "rgba(0,240,255,0.25)",
        },
      });

      const fitAddon = new FitAddon();
      term.loadAddon(fitAddon);
      term.open(containerRef.current);

      // Initial fit — the container has a fixed height set by the wrapper so
      // this should be accurate on first paint.
      try {
        fitAddon.fit();
      } catch {
        // Container not ready yet; the ResizeObserver below will retry.
      }

      term.writeln(
        `\x1b[90m# Connecting to job ${jobId}...\x1b[0m`
      );

      termRef.current = term;
      fitAddonRef.current = fitAddon;

      observer = new ResizeObserver(() => {
        try {
          fitAddon.fit();
        } catch {
          // Ignore transient fit failures during layout shifts.
        }
      });
      observer.observe(containerRef.current);

      setReady(true);
    })();

    return () => {
      disposed = true;
      if (observer) observer.disconnect();
      if (termRef.current) {
        try {
          termRef.current.dispose();
        } catch {}
        termRef.current = null;
      }
      fitAddonRef.current = null;
    };
    // tone only affects the cursor color and other visuals; we don't want
    // to recreate the terminal when the prop changes (it doesn't in practice).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  // Replay log snapshot once the terminal is ready.
  useEffect(() => {
    if (!ready) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}/logs`);
        if (!res.ok) return;
        const data = await res.json();
        if (cancelled) return;
        if (data.logs && termRef.current) {
          termRef.current.write(data.logs);
        }
      } catch {
        // Snapshot is best-effort.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [ready, jobId]);

  // Subscribe to live updates. The hook's onLog is called for every chunk
  // emitted by JobManager; data.text is the raw stdout/stderr buffer.
  useWebSocket(jobId, {
    onLog: (msg: any) => {
      if (msg?.data?.text) writeChunk(msg.data.text);
    },
    onStatus: (msg: any) => {
      if (msg?.data?.status) setStatus(msg.data.status as JobStatus);
    },
    onError: (msg: any) => {
      const text = msg?.data?.message || "Stream error";
      writeChunk(`\r\n\x1b[31m[error] ${text}\x1b[0m\r\n`);
    },
  });

  const handleStop = async () => {
    writeChunk(`\r\n\x1b[33m# Stop requested...\x1b[0m\r\n`);
    await stop(jobId);
  };

  const badge = formatStatusBadge(status);
  const disabled = stopping || isTerminalStatus(status);
  const accent = TONE_CLASSES[tone];

  return (
    <div
      className={`glass rounded-xl border ${accent.border} ${accent.glow} overflow-hidden`}
    >
      <div
        className={`flex items-center justify-between px-4 py-2 border-b ${accent.headerBorder} bg-background/40`}
      >
        <div className="flex items-center gap-3">
          <span className={`text-xs font-mono ${accent.text}`}>●</span>
          <span className="text-xs font-mono text-muted-foreground truncate max-w-[60ch]">
            job: {jobId}
          </span>
          <span
            className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider ${badge.className}`}
          >
            {badge.pulsing && (
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-current opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-current" />
              </span>
            )}
            {badge.label}
          </span>
          <span className="text-[10px] text-muted-foreground">
            {connected ? "WS ✓" : "WS …"}
          </span>
        </div>
        <button
          onClick={handleStop}
          disabled={disabled}
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium transition-all ${
            disabled
              ? "bg-accent/20 text-muted-foreground/50 cursor-not-allowed"
              : "bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20"
          }`}
        >
          {stopping ? "Stopping…" : "■ Stop"}
        </button>
      </div>
      <div
        ref={containerRef}
        className="h-96 w-full bg-black"
        style={{ minHeight: "24rem" }}
      />
      {!ready && (
        <div className="px-4 py-2 text-xs text-muted-foreground border-t border-border/50">
          Initialising terminal…
        </div>
      )}
    </div>
  );
}