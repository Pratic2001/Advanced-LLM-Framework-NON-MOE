"use client";

// Side-effect first: materializes process.env.NEXTAUTH_URL (from the
// NEXT_PUBLIC copy) so next-auth/react derives the correct /heavy basePath
// when it evaluates below. Must precede the next-auth/react import.
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