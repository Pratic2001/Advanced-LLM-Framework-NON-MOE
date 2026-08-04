/**
 * POST /api/shell/open
 *
 * Spawn a new bash PTY scoped to the authenticated user and the configured
 * REPO_ROOT. Returns a `sessionId` opaque token that the client passes to
 * the other `/api/shell/*` endpoints.
 *
 * Body: `{ cols?: number, rows?: number }`
 * Response: `{ sessionId, pid, activePtys }`
 *
 * Auth: requires a signed-in session (same Auth.js cookie as the rest of
 * the UI). The PTY's `userId` is stamped at spawn time so even a leaked
 * sessionId can't be used by another user.
 */
import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { openShell } from "@/lib/shell-sessions";

// Force dynamic — this route is per-user, no caching.
export const dynamic = "force-dynamic";

// Node runtime — the PTY registry uses Node's `setTimeout`/`Map` and we
// need long-lived idle timers. Edge would cap them.
export const runtime = "nodejs";

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  let body: { cols?: number; rows?: number } = {};
  try {
    body = await req.json();
  } catch {
    // Empty body is fine — we have defaults.
  }

  try {
    const result = openShell({
      userId: session.user.id,
      cols: typeof body.cols === "number" ? body.cols : 80,
      rows: typeof body.rows === "number" ? body.rows : 24,
    });
    return NextResponse.json(result);
  } catch (err: any) {
    const message = err?.message || "Failed to open shell";
    const status = message.includes("Too many") ? 429 : 500;
    return NextResponse.json({ error: message }, { status });
  }
}