"use client";

/**
 * IntegratedTerminal — composes the xterm.js panel with a header that shows
 * the exact command that will run / is running, plus a "View in Jobs" link.
 *
 * Mounts TerminalPanel via next/dynamic with ssr: false because xterm touches
 * the DOM at module load time and Next.js can't render it on the server.
 */

import dynamic from "next/dynamic";
import Link from "next/link";
import { Terminal as TerminalIcon, ExternalLink, Copy, Check } from "lucide-react";
import { useState } from "react";

const TerminalPanel = dynamic(
  () => import("./TerminalPanel").then((m) => m.TerminalPanel),
  {
    ssr: false,
    loading: () => (
      <div className="glass rounded-xl border border-border/50 h-[28rem] flex items-center justify-center text-xs text-muted-foreground">
        Loading terminal…
      </div>
    ),
  }
);

interface IntegratedTerminalProps {
  /** The job returned from POST /api/jobs. */
  jobId: string;
  /** Backend drives the accent color: torch → cyan, deepspeed → blue. */
  backend: "torch" | "deepspeed";
  /** Exact command string the server will execute. */
  commandPreview: string;
}

export function IntegratedTerminal({
  jobId,
  backend,
  commandPreview,
}: IntegratedTerminalProps) {
  const [copied, setCopied] = useState(false);

  const tone = backend === "torch" ? "cyan" : "blue";
  const accentText = tone === "cyan" ? "text-neon-cyan" : "text-neon-blue";
  const jobsHref = `/dashboard/${backend === "torch" ? "torchtab" : "deepspeed"}/jobs/${jobId}`;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(commandPreview);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard may be blocked; silent failure.
    }
  };

  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold flex items-center gap-2">
          <TerminalIcon className={`w-4 h-4 ${accentText}`} />
          Live Output
        </h2>
        <Link
          href={jobsHref}
          className={`inline-flex items-center gap-1.5 text-xs ${accentText} hover:opacity-80 transition-opacity`}
        >
          View in Jobs
          <ExternalLink className="w-3 h-3" />
        </Link>
      </div>

      <div className="glass rounded-xl border border-border/50 overflow-hidden">
        <div className="flex items-center gap-2 px-4 py-2 border-b border-border/50 bg-background/40">
          <span className="text-[10px] font-mono text-muted-foreground select-none">
            $
          </span>
          <code className="flex-1 text-xs font-mono text-foreground/80 truncate">
            {commandPreview}
          </code>
          <button
            onClick={handleCopy}
            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-medium transition-all ${
              copied
                ? "bg-emerald-500/10 text-emerald-400"
                : "bg-accent/30 text-muted-foreground hover:text-foreground"
            }`}
            title="Copy command"
          >
            {copied ? (
              <>
                <Check className="w-3 h-3" /> Copied
              </>
            ) : (
              <>
                <Copy className="w-3 h-3" /> Copy
              </>
            )}
          </button>
        </div>

        <TerminalPanel jobId={jobId} tone={tone} />
      </div>
    </section>
  );
}