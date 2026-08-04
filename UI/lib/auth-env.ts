"use client";

// next-auth/react computes its client `basePath` from the PATH of
// process.env.NEXTAUTH_URL at module-evaluation time (parseUrl(NEXTAUTH_URL).path)
// — see node_modules/next-auth/react.js / lib/client.js.
//
// Next.js only inlines NEXT_PUBLIC_* vars into browser bundles, so
// process.env.NEXTAUTH_URL is otherwise `undefined` in the client, and
// parseUrl(undefined) falls back to the default URL `http://localhost:3000/api/auth`,
// whose path /api/auth coincidentally matches the real endpoints at the
// root — but silently breaks the moment the app is served under a basePath.
//
// To make Auth.js aware of /heavy we materialize NEXTAUTH_URL from the
// NEXT_PUBLIC copy BEFORE next-auth/react evaluates. This module must be
// imported ahead of `import { SessionProvider } from "next-auth/react"` so
// ESM evaluation order sets process.env first. Imported for its side effect.
if (
  typeof window !== "undefined" &&
  typeof process !== "undefined" &&
  process.env &&
  !process.env.NEXTAUTH_URL &&
  process.env.NEXT_PUBLIC_NEXTAUTH_URL
) {
  process.env.NEXTAUTH_URL = process.env.NEXT_PUBLIC_NEXTAUTH_URL;
}

export {};