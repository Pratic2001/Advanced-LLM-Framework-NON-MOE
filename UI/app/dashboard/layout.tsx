import { redirect } from "next/navigation";
import type { ReactNode } from "react";
import { auth } from "@/lib/auth";
import DashboardShell from "./_shell";

// Server-side guard. Auth.js v5's `auth()` decodes the JWT from the
// `authjs.session-token` cookie (or the legacy `next-auth.session-token`
// name as a fallback) and returns the session or null. If unauthenticated,
// we redirect before any client component mounts — this prevents the
// perpetual "loading" state where a slow/hung SessionProvider leaves the
// dashboard stuck on its spinner right after sign-in.
export default async function DashboardLayout({
  children,
}: {
  children: ReactNode;
}) {
  const session = await auth();
  if (!session) {
    redirect("/login");
  }

  return <DashboardShell>{children}</DashboardShell>;
}
