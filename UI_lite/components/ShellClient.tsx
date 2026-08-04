"use client";

import { ReactNode } from "react";
import { usePalette } from "@/components/PaletteProvider";
import { InteractiveBackground } from "@/components/InteractiveBackground";

/**
 * Global client shell that needs access to the palette context.
 * Renders the mouse-reactive InteractiveBackground behind all page content
 * so every route gets the same animated, palette-aware canvas.
 */
export function ShellClient({ children }: { children: ReactNode }) {
  const { currentPalette } = usePalette();
  return (
    <>
      <InteractiveBackground palette={currentPalette.id} />
      {children}
    </>
  );
}