"use client";

// IMPORT-effect first: materializes process.env.NEXTAUTH_URL (from the
// NEXT_PUBLIC copy) so next-auth derives the /lite basePath when it evaluates.
import "@/lib/auth-env";
import { SessionProvider } from "next-auth/react";
import { PaletteProvider } from "@/components/PaletteProvider";
import { ShellClient } from "@/components/ShellClient";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <SessionProvider>
      <PaletteProvider>
        <ShellClient>{children}</ShellClient>
      </PaletteProvider>
    </SessionProvider>
  );
}