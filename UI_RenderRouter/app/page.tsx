"use client";

import { useEffect, useState } from "react";
import { runAudit, type AuditResult } from "@/lib/audit";

type Status = "auditing" | "done" | "error";
type Targets = { heavy: string; lite: string };

// Where the heavy/lite UIs live.
//
// NEXT_PUBLIC_* env vars (pinned in .env/.env.local to the Tailscale MagicDNS
// endpoints, e.g. http://<host>.ts.net:3000 / :3001) win. Without them the host
// is derived from wherever the router page itself was opened — localhost, a LAN
// IP, or the tailnet hostname — because all three apps bind 0.0.0.0 on the same
// machine, so the router is reached under the same hostname as the UIs it
// points at. Ports follow NODE_ENV: prod → 3000/3001, dev → 3210/3211.
function resolveTargets(): Targets {
  const envHeavy = process.env.NEXT_PUBLIC_HEAVY_UI_URL;
  const envLite = process.env.NEXT_PUBLIC_LITE_UI_URL;
  if (envHeavy && envLite) return { heavy: envHeavy, lite: envLite };

  // Client-side only: window is undefined during SSR, so this function must be
  // called from a mount effect, never from useMemo (see below).
  const host = window.location.hostname;
  const prod = process.env.NODE_ENV === "production";
  // Always http: the UIs are plain-HTTP processes (next start / server.ts). If
  // this page were ever reached over https (e.g. Tailscale Funnel), putting
  // https on a non-443 port would try a TLS handshake against an HTTP server
  // and fail — so derive http and leave https/funnel paths to the env override.
  return {
    heavy: `http://${host}:${prod ? "3000" : "3210"}`,
    lite: `http://${host}:${prod ? "3001" : "3211"}`,
  };
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <tr>
      <td>{k}</td>
      <td>{v}</td>
    </tr>
  );
}

export default function Page() {
  const [targets, setTargets] = useState<Targets | null>(null);
  const [status, setStatus] = useState<Status>("auditing");
  const [result, setResult] = useState<AuditResult | null>(null);
  const [countdown, setCountdown] = useState(6);
  const [stay, setStay] = useState(false);
  const [forced, setForced] = useState<"heavy" | "lite" | null>(null);

  // Query-param controls:
  //   ?force=heavy|lite  → skip the audit, route immediately
  //   ?audit=1           → run the audit but never auto-redirect (inspection)
  useEffect(() => {
    // Resolve targets here, in the browser. During SSR `window` is undefined;
    // useMemo([]) would freeze the server-computed "localhost" value on the
    // client and the router would always aim at localhost even when it was
    // opened via the tailnet hostname or a LAN IP.
    setTargets(resolveTargets());

    const params = new URLSearchParams(window.location.search);
    const f = params.get("force");
    if (f === "heavy" || f === "lite") {
      setForced(f);
      return;
    }
    if (params.get("audit") === "1") setStay(true);
    let cancelled = false;
    runAudit()
      .then((r) => {
        if (!cancelled) {
          setResult(r);
          setStatus("done");
        }
      })
      .catch(() => {
        if (!cancelled) setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const target = result
    ? result.verdict === "heavy"
      ? targets?.heavy ?? null
      : targets?.lite ?? null
    : null;
  const otherUrl = result
    ? result.verdict === "heavy"
      ? targets?.lite ?? null
      : targets?.heavy ?? null
    : null;

  // Forced routing — brief pause so the landing paints, then navigate.
  useEffect(() => {
    if (!forced || !targets) return;
    const t = setTimeout(
      () => window.location.replace(forced === "heavy" ? targets.heavy : targets.lite),
      500
    );
    return () => clearTimeout(t);
  }, [forced, targets]);

  // Countdown auto-redirect once the audit completes.
  useEffect(() => {
    if (status !== "done" || stay || !target) return;
    if (countdown <= 0) {
      window.location.replace(target);
      return;
    }
    const t = setTimeout(() => setCountdown((c) => c - 1), 1000);
    return () => clearTimeout(t);
  }, [status, countdown, stay, target]);

  const go = (url: string) => window.location.replace(url);

  if (forced) {
    const label = forced === "heavy" ? "Heavy UI" : "Lite UI";
    const url = forced === "heavy" ? targets?.heavy : targets?.lite;
    return (
      <main className="wrap">
        <div className="card center">
          <div className="pill pill--blue">FORCED OVERRIDE</div>
          <h1>Routing to {label}…</h1>
          {url ? (
            <>
              <p className="muted">
                Opening <code>{url}</code>
              </p>
              <a className="btn btn--primary" href={url}>
                Proceed
              </a>
            </>
          ) : (
            <div className="spinner" />
          )}
        </div>
      </main>
    );
  }

  return (
    <main className="wrap">
      <div className="card">
        <div className="card-head">
          <span className="logo">◆</span>
          <span className="brand">LLMForge · Render Router</span>
        </div>

        {status === "auditing" && (
          <>
            <div className="spinner" />
            <p className="muted">Auditing your hardware…</p>
          </>
        )}

        {status === "error" && (
          <>
            <div className="pill pill--red">COULD NOT AUDIT</div>
            <h1>Choose how you&apos;d like to run the UI</h1>
            <div className="actions">
              <button
                className="btn btn--primary"
                onClick={() => targets && go(targets.heavy)}
              >
                Open Heavy UI
              </button>
              <button
                className="btn"
                onClick={() => targets && go(targets.lite)}
              >
                Open Lite UI
              </button>
            </div>
          </>
        )}

        {status === "done" && result && (
          <>
            <div
              className={`pill ${
                result.verdict === "heavy" ? "pill--blue" : "pill--green"
              }`}
            >
              {result.verdict === "heavy" ? "ROUTING TO HEAVY UI" : "ROUTING TO LITE UI"}
            </div>
            <h1>
              {result.verdict === "heavy"
                ? "This machine can handle the full experience."
                : "Routed to the lighter UI for smooth performance."}
            </h1>
            <p className="muted">
              Auto-redirecting to <code>{target}</code>
              {stay ? " (auto-route paused)." : ` in ${countdown}s.`} You can override below.
            </p>

            <table className="hw">
              <tbody>
                <Row k="WebGL" v={result.webgl2 ? "WebGL2" : result.webgl1 ? "WebGL1" : "None"} />
                <Row k="GPU" v={result.renderer || "unknown"} />
                <Row k="Memory" v={result.deviceMemory != null ? `${result.deviceMemory} GB` : "unknown"} />
                <Row k="CPU threads" v={result.cores != null ? String(result.cores) : "unknown"} />
                <Row k="Viewport" v={result.viewport} />
                <Row k="Pixel ratio" v={String(result.dpr)} />
                {result.webglDraws != null && (
                  <Row k="WebGL stress" v={`${result.webglDraws} draws/s`} />
                )}
                {result.canvas2D != null && (
                  <Row k="2D canvas" v={`${result.canvas2D} ops/s`} />
                )}
                <Row k="Score" v={`${result.score}`} />
              </tbody>
            </table>

            <ul className="reasons">
              {result.reasons.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>

            <div className="actions">
              <button className="btn btn--primary" onClick={() => target && go(target)}>
                Continue to {result.verdict === "heavy" ? "Heavy UI" : "Lite UI"}
              </button>
              <button className="btn" onClick={() => otherUrl && go(otherUrl)}>
                Open {result.verdict === "heavy" ? "Lite UI" : "Heavy UI"} anyway
              </button>
              <button
                className="btn btn--ghost"
                onClick={() => setStay(true)}
                disabled={stay}
              >
                {stay ? "Auto-route paused" : "Pause auto-route"}
              </button>
            </div>
          </>
        )}
      </div>
    </main>
  );
}
