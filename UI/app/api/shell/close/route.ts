/**
 * POST /api/shell/close
 *
 * Kill a previously opened shell session's PTY and clean up.
 *
 * Body: `{ sessionId: string }`
 * Response: `{ ok: true }`
 *
 * Idempotent: closing an unknown session is a 404, closing an already-closed
 * session is a no-op success (the client's EventSource will close on its own
 * once the PTY exits, and we want close calls to be safe to retry).
 */
import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { killSession, getSession } from "@/lib/shell-sessions";

export const dynamic = "force-dynamic";

// Node runtime — session registry uses Node timers.
export const runtime = "nodejs";

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  let body: { sessionId?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  if (!body.sessionId) {
    return NextResponse.json({ error: "Missing sessionId" }, { status: 400 });
  }

  // Verify ownership before killing. We use the same helper as the other
  // routes so the userId check is enforced identically.
  const rec = getSession(body.sessionId, session.user.id);
  if (!rec) {
    // Already gone, or never ours. Either way, the caller's intent is
    // satisfied — no PTY is running for that sessionId.
    return new NextResponse(null, { status: 204 });
  }

  killSession(body.sessionId);
  return new NextResponse(null, { status: 204 });
}