// ── WebGL capability probe (three-free) ────────────────────────────────────
//
// Kept in its own module WITHOUT importing three so the always-mounted
// InteractiveBackground can gate on GPU availability without pulling Three.js
// into the main bundle. World components (NeonCityWebGL etc.) call this too
// for their internal fallback.

export function supportsWebGL(): boolean {
  if (typeof window === "undefined") return false;
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}
