/**
 * POST /api/shell/input
 *
 * Write data to a previously opened shell session's PTY stdin.
 *
 * Body: `{ sessionId: string, data: string }`
 * Response: `{ ok: true }` (204 No Content on success)
 *
 * Auth: session cookie via `auth()`. sessionId must belong to the
 * requesting user.
 */
import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { getSession, writeInput } from "@/lib/shell-sessions";

export const dynamic = "force-dynamic";

// Node runtime — session registry uses Node timers.
export const runtime = "nodejs";

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  let body: { sessionId?: string; data?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  if (!body.sessionId || typeof body.data !== "string") {
    return NextResponse.json(
      { error: "Missing sessionId or data" },
      { status: 400 }
    );
  }

  const rec = getSession(body.sessionId, session.user.id);
  if (!rec) {
    return NextResponse.json({ error: "Unknown session" }, { status: 404 });
  }

  try {
    writeInput(rec, body.data);
  } catch (err: any) {
    return NextResponse.json(
      { error: err?.message || "Write failed" },
      { status: 500 }
    );
  }

  return new NextResponse(null, { status: 204 });
}