import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { getToken } from "next-auth/jwt";

export const config = {
  // /api/((?!ws).*) matches everything under /api EXCEPT /api/ws.
  // Next.js's dev router runs middleware against WebSocket *upgrade*
  // requests too when the path matches the matcher, which throws
  // "Cannot read properties of undefined (reading 'bind')" for socket
  // upgrades (see vercel/next.js#56368 and related issues). The terminal
  // socket at /api/ws authenticates itself independently via cookies in
  // lib/ws-manager.ts, so excluding it here does not weaken auth.
  matcher: ["/dashboard/:path*", "/login", "/signup", "/api/((?!ws).*)"],
};

// CSP is intentionally not set here — middleware CSP can cause issues with
// Next.js chunk loading. The CSP baseline is applied in next.config.mjs headers().
const HEADERS = {
  "X-Frame-Options": "DENY",
  "X-Content-Type-Options": "nosniff",
  "Referrer-Policy": "strict-origin-when-cross-origin",
  "Permissions-Policy":
    "camera=(), microphone=(), geolocation=(), interest-cohort=()",
};

const PROD_HSTS =
  process.env.NODE_ENV === "production"
    ? { "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload" }
    : {};

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // ── Route protection ────────────────────────────────────────────────────
  // Protect /dashboard/* and /api/* (except /api/auth which NextAuth handles)
  const isProtectedRoute =
    pathname.startsWith("/dashboard") ||
    (pathname.startsWith("/api") && !pathname.startsWith("/api/auth"));

  const isAuthPage = pathname === "/login" || pathname === "/signup";

  // Auth.js v5 (5.0.0-beta.7+) renamed the session cookie from
  // `next-auth.session-token` to `authjs.session-token`. We try the new name
  // first; if absent, fall back to the legacy name in case a legacy session
  // cookie is still around. We pass the raw cookie value through to
  // `auth()` later so the dashboard layout can decode it server-side.
  const isHttps = request.nextUrl.protocol === "https:";
  const cookieName = isHttps
    ? "__Secure-authjs.session-token"
    : "authjs.session-token";
  const legacyCookieName = isHttps
    ? "__Secure-next-auth.session-token"
    : "next-auth.session-token";

  const token = await getToken({
    req: request,
    secret: process.env.NEXTAUTH_SECRET ?? process.env.AUTH_SECRET,
    cookieName,
    // Surface a clearer error if the cookie name is wrong rather than silent
    // 401s downstream.
    raw: false,
  });

  // If the new-name cookie wasn't present but the legacy one was, try again
  // with the legacy name. This keeps existing sessions valid across the
  // upgrade window.
  const tokenViaLegacy = token
    ? null
    : await getToken({
        req: request,
        secret: process.env.NEXTAUTH_SECRET ?? process.env.AUTH_SECRET,
        cookieName: legacyCookieName,
      });

  const session = token ?? tokenViaLegacy;

  if (isProtectedRoute && !session) {
    const loginUrl = new URL("/login", request.url);
    loginUrl.searchParams.set("callbackUrl", pathname);
    return NextResponse.redirect(loginUrl);
  }

  if (isAuthPage && token) {
    return NextResponse.redirect(new URL("/dashboard", request.url));
  }

  // ── Security headers ────────────────────────────────────────────────────
  const response = NextResponse.next();
  for (const [key, value] of Object.entries({ ...HEADERS, ...PROD_HSTS })) {
    response.headers.set(key, value);
  }

  return response;
}
