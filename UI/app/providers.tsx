"use client";

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