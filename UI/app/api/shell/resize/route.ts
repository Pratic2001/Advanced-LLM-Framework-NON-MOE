/**
 * POST /api/shell/resize
 *
 * Resize a shell session's PTY (forwards SIGWINCH).
 *
 * Body: `{ sessionId: string, cols: number, rows: number }`
 * Response: `{ ok: true }`
 */
import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { getSession, resizeShell } from "@/lib/shell-sessions";

export const dynamic = "force-dynamic";

// Node runtime — session registry uses Node timers.
export const runtime = "nodejs";

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  let body: { sessionId?: string; cols?: number; rows?: number };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  if (
    !body.sessionId ||
    typeof body.cols !== "number" ||
    typeof body.rows !== "number"
  ) {
    return NextResponse.json(
      { error: "Missing sessionId, cols, or rows" },
      { status: 400 }
    );
  }

  const rec = getSession(body.sessionId, session.user.id);
  if (!rec) {
    return NextResponse.json({ error: "Unknown session" }, { status: 404 });
  }

  try {
    resizeShell(rec, body.cols, body.rows);
  } catch (err: any) {
    return NextResponse.json(
      { error: err?.message || "Resize failed" },
      { status: 500 }
    );
  }

  return new NextResponse(null, { status: 204 });
}