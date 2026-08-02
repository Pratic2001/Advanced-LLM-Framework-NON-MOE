// ── Shared helpers for the per-palette WebGL worlds ────────────────────────
//
// These are small utilities, NOT a scene engine: each world component is a
// self-contained scene (like CosmosWebGL.tsx) and only borrows these helpers
// for the two things they all need — probing WebGL and reading the active
// palette's colors. World scenes read `--palette-*` CSS vars live so the
// in-app PaletteEditor retunes them in real time.

import * as THREE from "three";

export interface WorldPalette {
  primary: THREE.Color;
  secondary: THREE.Color;
  tertiary: THREE.Color;
  accent: THREE.Color;
}

// Parse "H S% L%" (the CSS var format) → [h, s, l] in 0..360 / 0..100.
function hslTriple(hsl: string): [number, number, number] | null {
  const [h, s, l] = hsl.trim().split(/\s+/).map((v) => parseFloat(v));
  if (
    h == null ||
    s == null ||
    l == null ||
    Number.isNaN(h) ||
    Number.isNaN(s) ||
    Number.isNaN(l)
  ) {
    return null;
  }
  return [h, s, l];
}

/** Read the four accent colors currently on <html> as THREE.Color. */
export function readPaletteColors(): WorldPalette {
  const root = getComputedStyle(document.documentElement);
  const get = (key: string): THREE.Color => {
    const triple = hslTriple(root.getPropertyValue(`--palette-${key}`));
    return triple
      ? new THREE.Color().setHSL(triple[0] / 360, triple[1] / 100, triple[2] / 100)
      : new THREE.Color(0x000000);
  };
  return {
    primary: get("primary"),
    secondary: get("secondary"),
    tertiary: get("tertiary"),
    accent: get("accent"),
  };
}

/** True when every channel of both palettes is identical (for cheap diffing). */
export function paletteEqual(a: WorldPalette, b: WorldPalette): boolean {
  return (
    a.primary.equals(b.primary) &&
    a.secondary.equals(b.secondary) &&
    a.tertiary.equals(b.tertiary) &&
    a.accent.equals(b.accent)
  );
}

/**
 * Throttled palette watcher for the animate loop. Re-reads the CSS vars at
 * most every `intervalMs` and calls `onChange(next)` only when an accent
 * differs from the last delivered palette. Call the returned function with
 * `performance.now()` each frame. Seed the initial colors with a plain
 * `readPaletteColors()` on mount.
 */
export function makePalettePoller(
  intervalMs: number,
  onChange: (next: WorldPalette) => void,
): (nowMs: number) => void {
  let last: WorldPalette | null = null;
  let lastAt = -Infinity;
  return (nowMs: number) => {
    if (nowMs - lastAt < intervalMs) return;
    lastAt = nowMs;
    const next = readPaletteColors();
    if (!last || !paletteEqual(last, next)) {
      last = next;
      onChange(next);
    }
  };
}
