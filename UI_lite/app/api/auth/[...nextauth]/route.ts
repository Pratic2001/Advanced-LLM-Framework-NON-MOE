import { handlers } from "@/lib/auth";
import { BASE_PATH } from "@/lib/base-path";
import type { NextRequest } from "next/server";

// See UI/app/api/auth/[...nextauth]/route.ts for the rationale — this app
// is served at /lite, so the same `${ORIGIN}/api/auth/...` -> `/lite/api/auth/...`
// rebasing applies. The two route files are identical except for which
// dependency they import from.
const ORIGIN = process.env.AUTH_URL ?? process.env.NEXTAUTH_URL ?? "";

function rebaseLocation(loc: string): string {
  if (!BASE_PATH || !loc.startsWith(ORIGIN)) return loc;
  const after = loc.slice(ORIGIN.length);
  if (after.startsWith(BASE_PATH) || !after.startsWith("/")) return loc;
  if (!after.startsWith("/api/auth")) return loc;
  return `${ORIGIN}${BASE_PATH}${after}`;
}

async function handle(req: NextRequest) {
  const res = await handlers.POST(req);
  const loc = res.headers.get("Location");
  if (!loc) return res;
  const fixed = rebaseLocation(loc);
  if (fixed === loc) return res;

  const out = new Response(res.body, {
    status: res.status,
    statusText: res.statusText,
    headers: res.headers,
  });
  out.headers.set("Location", fixed);
  return out;
}

export const GET = handle;
export const POST = handle;
