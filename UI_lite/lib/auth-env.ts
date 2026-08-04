"use client";

// See UI/lib/auth-env.ts. This app is served at /lite.
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