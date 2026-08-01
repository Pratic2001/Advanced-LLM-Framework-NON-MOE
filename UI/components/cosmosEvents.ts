// ── Cosmography → UI event bus ─────────────────────────────────────────────
//
// A tiny, dependency-free broadcast channel between the WebGL cosmos background
// (CosmosWebGL.tsx, which EMITS) and palette-aware chrome (PaletteProvider,
// which LISTENS). Keeping it in its own module means the palette provider can
// subscribe without pulling Three.js into its bundle.
//
// Events are fire-and-forget pulses: no return value, no acknowledgement, and
// no per-frame chatter. Consumers use the heat/hue to briefly tint the accent
// tokens, then decay back to the palette's base values on their own clock.

export interface CosmosEvent {
  type: "supernova";
  /** 0..1 — how strongly the UI should react to this blast. */
  heat: number;
  /** Target hue (degrees) that the event tints the palette toward. */
  hue: number;
  /** Screen-space position in NDC (-1..1). May sit outside the viewport. */
  x: number;
  y: number;
}

type CosmosListener = (e: CosmosEvent) => void;

const listeners = new Set<CosmosListener>();

/** Subscribe to background events. Returns an unsubscribe function. */
export function onCosmosEvent(listener: CosmosListener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Broadcast a background event to every listener. Never throws. */
export function emitCosmosEvent(e: CosmosEvent) {
  for (const l of listeners) {
    try {
      l(e);
    } catch {
      // A listener must never break the render loop or the emitter.
    }
  }
}
