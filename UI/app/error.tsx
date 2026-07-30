"use client";

import { useEffect } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

export default function RootError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[RootError]", error);
  }, [error]);

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-background">
      <div className="glass-strong rounded-2xl p-8 max-w-md w-full text-center space-y-4">
        <div className="w-14 h-14 rounded-full bg-destructive/10 flex items-center justify-center mx-auto">
          <AlertTriangle className="w-7 h-7 text-destructive" />
        </div>
        <h1 className="text-xl font-bold">Something went wrong</h1>
        <p className="text-sm text-muted-foreground">
          An unexpected error occurred. Our team has been notified.
        </p>
        {error.digest && (
          <p className="text-xs text-muted-foreground/60 font-mono">
            Error ID: {error.digest}
          </p>
        )}
        <button
          onClick={reset}
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-neon-cyan to-neon-blue text-black font-semibold text-sm hover:opacity-90 transition-all"
        >
          <RefreshCw className="w-4 h-4" />
          Try again
        </button>
      </div>
    </div>
  );
}
