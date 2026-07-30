/**
 * Instrumentation hook for Next.js.
 * Runs once on server startup to set up observability.
 *
 * NOTE: This requires `experimental.instrumentationHook = true` in next.config.mjs
 * to be active. Currently disabled — uncomment in next.config.mjs when needed.
 */

export async function register() {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    console.log(
      `[instrumentation] Server starting — ${process.env.NODE_ENV ?? "development"} mode`
    );

    // In production, initialize OpenTelemetry here:
    // const { NodeSDK } = await import("@opentelemetry/sdk-node");
    // const { OTLPTraceExporter } = await import("@opentelemetry/exporter-trace-otlp-http");
    // const sdk = new NodeSDK({ traceExporter: new OTLPTraceExporter() });
    // await sdk.start();
  }
}
