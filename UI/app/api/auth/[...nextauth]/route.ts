import { handlers } from "@/lib/auth";
import { BASE_PATH } from "@/lib/base-path";
import type { NextRequest } from "next/server";

// next-auth builds its outgoing redirect URLs from `internalRequest.url.origin`
// joined with `pages.error` (or `${basePath}/error`). When the app is served
// under a Tailscale funnel sub-path (/heavy or /lite) the resulting Location
// header is the bare origin + path, e.g.
//
//   https://pratic-battleaxb450mkm2.tail5e5151.ts.net/api/auth/error?error=Configuration
//
// The funnel sends that absolute URL to Next without the /heavy prefix, so it
// lands on the funnel's catch-all (UI_RenderRouter) and 404s. Re-add the
// basePath to any 3xx Location that Auth.js emits so the brokered redirect
// stays inside this app.
//
// Why the wrapper is more conservative than fixing this upstream:
//   - Setting NEXTAUTH_URL to include the basePath makes setEnvDefaults pick
//     basePath = "/heavy" in Auth.js, which then breaks parseActionAndProviderId
//     against Next's basePath-stripped pathname (UnknownAction).
//   - Setting pages.error = "/heavy/api/auth/error" works in the failure path
//     but doesn't help sign-out / verify-request / callback redirects that go
//     through the same `${origin}${pagePath}` template.
//
// Only the Location header is touched; status, cookies, and body are copied
// verbatim.
const ORIGIN = process.env.AUTH_URL ?? process.env.NEXTAUTH_URL ?? "";

function rebaseLocation(loc: string): string {
  if (!BASE_PATH || !loc.startsWith(ORIGIN)) return loc;
  const after = loc.slice(ORIGIN.length);
  // already prefixed or is unrelated path (Callback URL to /dashboard etc.)
  if (after.startsWith(BASE_PATH) || !after.startsWith("/")) return loc;
  if (!after.startsWith("/api/auth")) return loc;
  return `${ORIGIN}${BASE_PATH}${after}`;
}

async function handle(req: NextRequest) {
  // handlers.GET and handlers.POST are the SAME function in next-auth v5
  // (see node_modules/next-auth/index.js); the function routes both verbs
  // internally via the request's method. Calling POST is safe for either.
  const res = await handlers.POST(req);
  const loc = res.headers.get("Location");
  if (!loc) return res;
  const fixed = rebaseLocation(loc);
  if (fixed === loc) return res;

  // Build a new Response copying status/headers/body. `new Response(body, res)`
  // would only forward status/statusText/headers-as-Headers; we want every
  // header including set-cookie, so iterate explicitly.
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
