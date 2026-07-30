"use client";

import { useEffect } from "react";
import Link from "next/link";
import { AlertTriangle, RefreshCw, LayoutDashboard } from "lucide-react";

export default function DashboardError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[DashboardError]", error);
  }, [error]);

  return (
    <div className="flex items-center justify-center py-20">
      <div className="glass-strong rounded-2xl p-8 max-w-md w-full text-center space-y-4">
        <div className="w-14 h-14 rounded-full bg-destructive/10 flex items-center justify-center mx-auto">
          <AlertTriangle className="w-7 h-7 text-destructive" />
        </div>
        <h1 className="text-xl font-bold">Dashboard Error</h1>
        <p className="text-sm text-muted-foreground">
          Something went wrong in this section.
        </p>
        <div className="flex items-center justify-center gap-3 pt-2">
          <button
            onClick={reset}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-neon-cyan to-neon-blue text-black font-semibold text-sm hover:opacity-90 transition-all"
          >
            <RefreshCw className="w-4 h-4" />
            Retry
          </button>
          <Link
            href="/dashboard"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-border text-sm text-muted-foreground hover:text-foreground transition-all"
          >
            <LayoutDashboard className="w-4 h-4" />
            Dashboard Home
          </Link>
        </div>
      </div>
    </div>
  );
}
