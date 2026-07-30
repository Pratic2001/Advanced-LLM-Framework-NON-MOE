"use client";

import { useEffect } from "react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[GlobalError]", error);
  }, [error]);

  return (
    <html>
      <body>
        <div
          style={{
            minHeight: "100vh",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: "1rem",
            backgroundColor: "#09090b",
            color: "#fafafa",
            fontFamily:
              'Inter, system-ui, -apple-system, sans-serif',
          }}
        >
          <div
            style={{
              maxWidth: "28rem",
              width: "100%",
              textAlign: "center",
              padding: "2rem",
              borderRadius: "1rem",
              background: "rgba(24, 24, 27, 0.8)",
              backdropFilter: "blur(24px)",
              border: "1px solid rgba(39, 39, 42, 0.5)",
            }}
          >
            <div
              style={{
                width: "3.5rem",
                height: "3.5rem",
                borderRadius: "9999px",
                background: "rgba(239, 68, 68, 0.1)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                margin: "0 auto 1rem",
                fontSize: "1.75rem",
              }}
            >
              ⚠
            </div>
            <h1 style={{ fontSize: "1.25rem", fontWeight: 700, marginBottom: "0.5rem" }}>
              Critical Error
            </h1>
            <p
              style={{
                fontSize: "0.875rem",
                color: "rgba(250, 250, 250, 0.6)",
                marginBottom: "1.5rem",
              }}
            >
              A critical error occurred. Please refresh the page to try again.
            </p>
            {error.digest && (
              <p
                style={{
                  fontSize: "0.75rem",
                  color: "rgba(250, 250, 250, 0.4)",
                  fontFamily: "monospace",
                  marginBottom: "1.5rem",
                }}
              >
                Error ID: {error.digest}
              </p>
            )}
            <button
              onClick={() => reset()}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "0.5rem",
                padding: "0.625rem 1.25rem",
                borderRadius: "0.75rem",
                background: "linear-gradient(135deg, #00f0ff, #0088ff)",
                color: "#000",
                fontWeight: 600,
                fontSize: "0.875rem",
                border: "none",
                cursor: "pointer",
              }}
            >
              Refresh page
            </button>
          </div>
        </div>
      </body>
    </html>
  );
}
