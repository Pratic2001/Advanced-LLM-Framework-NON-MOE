"use client";

import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import dynamic from "next/dynamic";

// The WebGL3D renderer must never run during SSR (needs a real canvas/GPU).
const CosmosWebGL = dynamic(
  () => import("./CosmosWebGL").then((m) => m.CosmosWebGL),
  { ssr: false, loading: () => null },
);

// ── Palette-aware interactive background ───────────────────────────────────
//
// Each palette id maps to a distinct "feel" rendered into a single canvas.
// Strategies are self-contained: they own their own particle arrays, advance
// their own state, and render to the provided 2D context.

interface RGB {
  r: number;
  g: number;
  b: number;
}

interface RGBA extends RGB {
  a: number;
}

// Parse "H S% L%" (the HSL triple format used by CSS variables) → RGB
function hslTriple(hsl: string): RGB {
  const [h, s, l] = hsl.trim().split(/\s+/).map((v) => parseFloat(v));
  const sn = s / 100;
  const ln = l / 100;
  const c = (1 - Math.abs(2 * ln - 1)) * sn;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = ln - c / 2;
  let r = 0,
    g = 0,
    b = 0;
  if (h < 60) [r, g, b] = [c, x, 0];
  else if (h < 120) [r, g, b] = [x, c, 0];
  else if (h < 180) [r, g, b] = [0, c, x];
  else if (h < 240) [r, g, b] = [0, x, c];
  else if (h < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  return {
    r: Math.round((r + m) * 255),
    g: Math.round((g + m) * 255),
    b: Math.round((b + m) * 255),
  };
}

function rgba(c: RGB, a: number): string {
  return `rgba(${c.r}, ${c.g}, ${c.b}, ${a})`;
}

// ── Neural nodes strategy ─────────────────────────────────────────────────
// Used by neon-cyber, aurora-borealis, cosmic-void.

interface NeuralNode {
  x: number;
  y: number;
  vx: number;
  vy: number;
  baseSize: number;
  phase: number;
}

function createNeuralNodes(count: number, w: number, h: number): NeuralNode[] {
  const nodes: NeuralNode[] = [];
  for (let i = 0; i < count; i++) {
    nodes.push({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      baseSize: 1.2 + Math.random() * 2.3,
      phase: Math.random() * Math.PI * 2,
    });
  }
  return nodes;
}

function renderNeuralNodes(
  ctx: CanvasRenderingContext2D,
  nodes: NeuralNode[],
  palette: RGB[],
  w: number,
  h: number,
  mouse: { x: number; y: number },
  interactive: boolean,
  reduced: boolean,
  time: number
) {
  ctx.clearRect(0, 0, w, h);

  // Update nodes
  for (const n of nodes) {
    if (interactive && !reduced) {
      const dx = mouse.x - n.x;
      const dy = mouse.y - n.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 200 && dist > 0) {
        const force = (1 - dist / 200) * 0.05;
        n.vx += (dx / dist) * force;
        n.vy += (dy / dist) * force;
      }
    }
    n.vx *= 0.985;
    n.vy *= 0.985;
    n.x += n.vx;
    n.y += n.vy;
    if (n.x < -20) n.x = w + 20;
    else if (n.x > w + 20) n.x = -20;
    if (n.y < -20) n.y = h + 20;
    else if (n.y > h + 20) n.y = -20;
    n.phase += 0.012;
  }

  // Connections (only with neighbouring nodes within radius)
  const connectRadius = 130;
  const pulse = (Math.sin(time * 0.001) + 1) * 0.5;
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const dx = nodes[i].x - nodes[j].x;
      const dy = nodes[i].y - nodes[j].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < connectRadius) {
        const opacity = (1 - dist / connectRadius) * 0.18 * (0.6 + pulse * 0.4);
        ctx.strokeStyle = rgba(palette[0], opacity);
        ctx.lineWidth = 0.6;
        ctx.beginPath();
        ctx.moveTo(nodes[i].x, nodes[i].y);
        ctx.lineTo(nodes[j].x, nodes[j].y);
        ctx.stroke();
      }
    }
  }

  // Nodes
  for (const n of nodes) {
    const size = n.baseSize * (0.8 + Math.sin(n.phase) * 0.2);
    const g = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, size * 3);
    g.addColorStop(0, rgba(palette[Math.floor(Math.random() * palette.length)], 0.85));
    g.addColorStop(1, rgba(palette[0], 0));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(n.x, n.y, size * 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ── Embers strategy ───────────────────────────────────────────────────────
// Used by solar-flare, golden-hour, rose-quartz.

interface Ember {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  life: number;
  maxLife: number;
  hue: number;
}

function createEmbers(count: number, w: number, h: number, palette: RGB[]): Ember[] {
  const embers: Ember[] = [];
  for (let i = 0; i < count; i++) {
    embers.push({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.3,
      vy: -(0.3 + Math.random() * 0.9),
      size: 0.6 + Math.random() * 1.8,
      life: Math.random() * 1,
      maxLife: 1,
      hue: Math.floor(Math.random() * palette.length),
    });
  }
  return embers;
}

function renderEmbers(
  ctx: CanvasRenderingContext2D,
  embers: Ember[],
  palette: RGB[],
  w: number,
  h: number,
  _mouse: { x: number; y: number },
  interactive: boolean,
  reduced: boolean,
  time: number
) {
  ctx.clearRect(0, 0, w, h);

  // Slight smoky overlay at the top so embers fade into haze
  const fadeGrad = ctx.createLinearGradient(0, 0, 0, h);
  fadeGrad.addColorStop(0, "rgba(0,0,0,0.55)");
  fadeGrad.addColorStop(0.6, "rgba(0,0,0,0.1)");
  fadeGrad.addColorStop(1, "rgba(0,0,0,0)");

  for (const e of embers) {
    if (!reduced) {
      e.x += e.vx + Math.sin(time * 0.001 + e.y * 0.01) * 0.4;
      e.y += e.vy;
      e.life += 0.0025;
      if (e.life >= e.maxLife || e.y < -10) {
        e.x = Math.random() * w;
        e.y = h + 10;
        e.life = 0;
      }
    }
    const fade = 1 - e.life;
    const c = palette[e.hue];
    const g = ctx.createRadialGradient(e.x, e.y, 0, e.x, e.y, e.size * 4);
    g.addColorStop(0, rgba(c, 0.9 * fade));
    g.addColorStop(0.4, rgba(c, 0.35 * fade));
    g.addColorStop(1, rgba(c, 0));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(e.x, e.y, e.size * 4, 0, Math.PI * 2);
    ctx.fill();

    // Tiny bright core
    ctx.fillStyle = rgba({ r: 255, g: 240, b: 220 }, 0.8 * fade);
    ctx.beginPath();
    ctx.arc(e.x, e.y, e.size * 0.5, 0, Math.PI * 2);
    ctx.fill();
  }

  // Apply fade-to-haze on top
  ctx.fillStyle = fadeGrad;
  ctx.fillRect(0, 0, w, h);
}

// ── Matrix glyphs strategy ────────────────────────────────────────────────
// Used by matrix-green.

const MATRIX_GLYPHS =
  "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン0123456789$#@%&*+=<>";

interface MatrixColumn {
  x: number;
  y: number;
  speed: number;
  length: number;
}

function createMatrixColumns(count: number, w: number, h: number): MatrixColumn[] {
  const cols: MatrixColumn[] = [];
  const colWidth = 18;
  for (let x = 0; x < w; x += colWidth) {
    cols.push({
      x: x + Math.random() * 6,
      y: Math.random() * h,
      speed: 1.5 + Math.random() * 4,
      length: 8 + Math.floor(Math.random() * 20),
    });
  }
  return cols;
}

function renderMatrixGlyphs(
  ctx: CanvasRenderingContext2D,
  cols: MatrixColumn[],
  palette: RGB[],
  w: number,
  h: number,
  _mouse: { x: number; y: number },
  _interactive: boolean,
  reduced: boolean,
  _time: number
) {
  // Soft fade trail so glyphs leave a smear
  ctx.fillStyle = "rgba(0, 0, 0, 0.18)";
  ctx.fillRect(0, 0, w, h);

  const c = palette[0]; // dominant green
  ctx.font = "16px monospace";
  ctx.textBaseline = "top";

  for (const col of cols) {
    if (!reduced) col.y += col.speed;
    if (col.y - col.length * 18 > h) {
      col.y = -col.length * 18 + Math.random() * 60;
    }
    for (let i = 0; i < col.length; i++) {
      const ch = MATRIX_GLYPHS[Math.floor(Math.random() * MATRIX_GLYPHS.length)];
      const y = col.y - i * 18;
      if (y < -20 || y > h + 20) continue;
      const isHead = i === 0;
      const alpha = isHead ? 1 : 0.85 * (1 - i / col.length);
      ctx.fillStyle = isHead
        ? "rgba(220, 255, 220, " + alpha + ")"
        : rgba(c, alpha);
      ctx.fillText(ch, col.x, y);
    }
  }
}

// ── Bubbles strategy ──────────────────────────────────────────────────────
// Used by ocean-depths.

interface Bubble {
  x: number;
  y: number;
  radius: number;
  vy: number;
  wobblePhase: number;
  wobbleAmp: number;
  hue: number;
}

function createBubbles(count: number, w: number, h: number, palette: RGB[]): Bubble[] {
  const bubbles: Bubble[] = [];
  for (let i = 0; i < count; i++) {
    bubbles.push({
      x: Math.random() * w,
      y: h + Math.random() * 200,
      radius: 1.5 + Math.random() * 5,
      vy: 0.4 + Math.random() * 1.2,
      wobblePhase: Math.random() * Math.PI * 2,
      wobbleAmp: 0.4 + Math.random() * 1.2,
      hue: Math.floor(Math.random() * palette.length),
    });
  }
  return bubbles;
}

function renderBubbles(
  ctx: CanvasRenderingContext2D,
  bubbles: Bubble[],
  palette: RGB[],
  w: number,
  h: number,
  _mouse: { x: number; y: number },
  _interactive: boolean,
  reduced: boolean,
  time: number
) {
  ctx.clearRect(0, 0, w, h);

  // Soft underwater caustics via vertical gradient
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, rgba(palette[0], 0.05));
  grad.addColorStop(0.5, rgba(palette[1], 0.03));
  grad.addColorStop(1, rgba(palette[2], 0.06));
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  for (const b of bubbles) {
    if (!reduced) {
      b.wobblePhase += 0.03;
      b.x += Math.sin(b.wobblePhase) * b.wobbleAmp;
      b.y -= b.vy;
      if (b.y < -20) {
        b.y = h + 20;
        b.x = Math.random() * w;
      }
    }
    const c = palette[b.hue];
    const g = ctx.createRadialGradient(
      b.x - b.radius * 0.3,
      b.y - b.radius * 0.3,
      0,
      b.x,
      b.y,
      b.radius
    );
    g.addColorStop(0, rgba({ r: 255, g: 255, b: 255 }, 0.45));
    g.addColorStop(0.4, rgba(c, 0.35));
    g.addColorStop(1, rgba(c, 0.05));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(b.x, b.y, b.radius, 0, Math.PI * 2);
    ctx.fill();

    // Ring outline
    ctx.strokeStyle = rgba(c, 0.5);
    ctx.lineWidth = 0.6;
    ctx.beginPath();
    ctx.arc(b.x, b.y, b.radius, 0, Math.PI * 2);
    ctx.stroke();
  }
}

// ── Cosmos strategy ───────────────────────────────────────────────────────
// Used by deep-space palette. Cinematic 3D scene: light-bending black hole
// with Interstellar-style Doppler-shifted disk, developing colourful hypernova
// whose shockwave grows in real time, glowing stars with parallax depth, and
// mouse-driven interactivity throughout.

interface Star {
  x: number;
  y: number;
  baseBrightness: number;
  phase: number;
  size: number;
  spike: boolean;
  hue: number;
  // 0..1 — drives parallax rate. 0 = background (slow), 1 = foreground (fast).
  depth: number;
  twinkleSpeed: number;
}

interface Galaxy {
  cx: number;
  cy: number;
  armCount: number;
  armSpread: number;
  size: number;
  rotation: number;
  rotationSpeed: number;
  coreColor: RGB;
  armColor: RGB;
  particles: { angle: number; radius: number; drift: number }[];
  depth: number;
}

interface BlackHole {
  x: number;
  y: number;
  // Sway target — the BH eases toward (swayX, swayY) where swayX/Y are driven
  // by the cursor plus a slow Lissajous drift. This is what reads as "alive".
  swayX: number;
  swayY: number;
  radius: number;
  // Accretion disk geometry
  diskInnerRadius: number;
  diskOuterRadius: number;
  // Tilt in radians. -1.05 ≈ -60° matches the Interstellar Gargantua.
  diskTilt: number;
  // Visual palette
  diskHot: RGB;  // inner edge — orange/yellow
  diskCool: RGB; // outer edge — blue/violet
  ringColor: RGB;
  // Rotating particles
  rotation: number;
  rotationSpeed: number;
  armParticles: { angle: number; radius: number; speed: number; size: number }[];
  // Lissajous drift parameters — slow figure-eight so the BH is never static.
  driftAngle: number;
  driftSpeed: number;
  driftRadiusX: number;
  driftRadiusY: number;
  driftPhaseX: number;
  driftPhaseY: number;
  // Cached offscreen disk rendering — rebuilt only when disk geometry/colour
  // changes, so the per-frame draw stays cheap.
  cachedDisk: HTMLCanvasElement | null;
  cachedHot: RGB;
  cachedCool: RGB;
  cachedOuter: number;
  cachedInner: number;
  cachedTilt: number;
}

interface Shockwave {
  x: number;
  y: number;
  age: number;
  maxAge: number;
  color: RGB;
}

interface NebulaCloud {
  x: number;
  y: number;
  radius: number;
  color: RGB;
  phaseX: number;
  phaseY: number;
  speedX: number;
  speedY: number;
  ampX: number;
  ampY: number;
  // Tint bias — mix factor toward warm/cool for variety across the 3 clouds.
  tint: "warm" | "cool" | "violet";
}

// A hypernova is a long-lived stellar lifecycle object — it goes through:
//   1. PROGENITOR    — a single bright star pulses gently, slowly brightening
//   2. COLLAPSE      — over ~1.2s the star contracts, dims, then explodes
//   3. CENTRAL FLASH — bright white-blue burst for ~0.6s
//   4. EJECTA        — filaments racing outward over ~12s as the shockwave
//                      grows; colour evolves white→yellow→orange→red→crimson
//   5. REMNANT       — mottled blue/red nebula built from cached FBM noise
//   6. QUIESCENT     — fades back into the background before being respawned
interface HypernovaFilament {
  angle: number;
  speed: number;          // outward velocity in px/sec
  length: number;         // beam sprite length in px
  hue: number;
  // Phase offset for per-filament noise modulation along the beam sprite —
  // each filament looks different, like tangled spaghetti rather than spokes.
  noisePhase: number;
  // Lateral spread (radians) — small wobble in angle so filaments don't fan
  // out perfectly straight; gives the ejecta an organic, turbulent feel.
  wobbleAmp: number;
  wobblePhase: number;
  wobbleSpeed: number;
}

interface Hypernova {
  cx: number;
  cy: number;
  birth: number;          // ms timestamp it entered PROGENITOR
  phase: number;          // 0..1 progress through current phase
  phaseStart: number;     // ms timestamp the current phase began
  state:
    | "progenitor"
    | "collapse"
    | "flash"
    | "ejecta"
    | "remnant"
    | "quiescent";
  filaments: HypernovaFilament[];
  // Cached radial beam sprite baked during the flash for cheap ejecta draws.
  beamPattern: HTMLCanvasElement | null;
  beamHue: number;
  progenitorSize: number;
  progenitorPulse: number;
  // Current shockwave emitted at the flash moment.
  shockwave: Shockwave | null;
  // Drift so successive cycles don't feel static.
  driftAngle: number;
  driftSpeed: number;
  driftRadius: number;
  driftCx: number;
  driftCy: number;
}

interface CosmosState {
  stars: Star[];
  galaxies: Galaxy[];
  blackHoles: BlackHole[];
  nebulae: NebulaCloud[];
  shockwaves: Shockwave[];
  lastShockwaveAt: number;
  hypernova: Hypernova;
  // ImageData pool used by the per-pixel lensing pass. Reused across frames to
  // avoid the GC pressure of allocating a fresh 200×200 buffer every frame.
  lensBuffer: ImageData | null;
  lensBufferW: number;
  lensBufferH: number;
  // Offscreen canvas for the lensing pass — gets the background drawn into it,
  // then read back as ImageData for in-place warping.
  lensCanvas: HTMLCanvasElement | null;
}

// Hypernova timeline (all in ms from cycle start)
const HN_PROGENITOR_MS = 6000;   // brightening star
const HN_COLLAPSE_MS = 1200;      // quick contract + flash onset
const HN_FLASH_MS = 600;          // blinding central burst
const HN_EJECTA_MS = 12000;       // filaments racing outward (dominant phase)
const HN_REMNANT_MS = 6000;       // cooling mottled nebula
const HN_QUIESCENT_MS = 2200;     // fade
// Total cycle so the user always sees action.
const HN_CYCLE_MS =
  HN_PROGENITOR_MS +
  HN_COLLAPSE_MS +
  HN_FLASH_MS +
  HN_EJECTA_MS +
  HN_REMNANT_MS +
  HN_QUIESCENT_MS;

const HN_FILAMENT_COUNT = 220;
const HN_EJECTA_SPEED = 180; // px/sec at outer radius (slightly slower so the
                             // 12s phase still feels like real expansion)
const HN_REMNANT_EXPAND = 110; // final remnant radius (px)

// Realistic stellar hues — picked from spectroscopic classifications.
const STELLAR_COLORS: RGB[] = [
  { r: 155, g: 195, b: 255 }, // O-type — blue
  { r: 200, g: 220, b: 255 }, // B-type — blue-white
  { r: 245, g: 245, b: 255 }, // A-type — white
  { r: 255, g: 245, b: 220 }, // F-type — yellow-white
  { r: 255, g: 215, b: 175 }, // G-type — sun-like (yellow)
  { r: 255, g: 180, b: 130 }, // K-type — orange
  { r: 255, g: 130, b: 100 }, // M-type — red
  { r: 255, g: 90,  b: 80  }, // M-giant — deep red
];

function pickRandom<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

// ── Mulberry32 seeded PRNG ────────────────────────────────────────────────
// Used to bake deterministic noise textures so the nebula looks identical
// across sessions. Math.random() would shift the texture on every reload.
function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── Module-level caches ───────────────────────────────────────────────────
// These survive `createCosmos()` re-runs (palette change) so we don't rebuild
// 256² noise textures or beam sprites every time the user clicks a palette.
let __nebulaNoiseHi: HTMLCanvasElement | null = null;  // high-freq filaments
let __nebulaNoiseLo: HTMLCanvasElement | null = null;  // low-freq cells
let __beamSpriteCache: HTMLCanvasElement | null = null;
let __nebulaSeed = 0xc0ffee;

function createCosmos(w: number, h: number, palette: RGB[]): {
  stars: Star[];
  galaxies: Galaxy[];
  blackHoles: BlackHole[];
  nebulae: NebulaCloud[];
  shockwaves: Shockwave[];
  lastShockwaveAt: number;
  hypernova: Hypernova;
} {
  // ── Starfield ───────────────────────────────────────────────────────────
  // Many faint background stars, sprinkled with hot blue / cool red giants,
  // plus a sparse set of "diffraction-spike" foreground stars that look like
  // pinpoints in a long-exposure astronomical photo.
  const stars: Star[] = [];
  for (let i = 0; i < 380; i++) {
    const hueIdx = Math.random();
    const color =
      hueIdx < 0.55
        ? 0 // majority white
        : Math.floor(Math.random() * STELLAR_COLORS.length);
    stars.push({
      x: Math.random() * w,
      y: Math.random() * h,
      baseBrightness: 0.25 + Math.random() * 0.7,
      phase: Math.random() * Math.PI * 2,
      size: 0.3 + Math.random() * 1.1,
      spike: false,
      hue: color,
      depth: Math.random(),
      twinkleSpeed: 0.015 + Math.random() * 0.035,
    });
  }
  for (let i = 0; i < 40; i++) {
    stars.push({
      x: Math.random() * w,
      y: Math.random() * h,
      baseBrightness: 0.85 + Math.random() * 0.15,
      phase: Math.random() * Math.PI * 2,
      size: 1.5 + Math.random() * 1.6,
      spike: true,
      hue: pickRandom([1, 2, 4, 5, 6]),
      depth: Math.random(),
      twinkleSpeed: 0.012 + Math.random() * 0.022,
    });
  }
  // A handful of saturated giant stars (blue/orange) — these read as "glowing"
  // because their hue and brightness contrast with the white population.
  for (let i = 0; i < 18; i++) {
    stars.push({
      x: Math.random() * w,
      y: Math.random() * h,
      baseBrightness: 0.7 + Math.random() * 0.3,
      phase: Math.random() * Math.PI * 2,
      size: 1.8 + Math.random() * 1.4,
      spike: true,
      hue: Math.random() < 0.5 ? 0 : 6, // deep blue or red giant
      depth: Math.random(),
      twinkleSpeed: 0.01 + Math.random() * 0.02,
    });
  }

  // ── Galaxies ────────────────────────────────────────────────────────────
  // Two log-spiral galaxies with hot cores and cooler, dustier arms.
  const galaxies: Galaxy[] = [];
  const galaxyCount = 2;
  for (let i = 0; i < galaxyCount; i++) {
    const size = 90 + Math.random() * 90;
    const particleCount = 220;
    const particles: { angle: number; radius: number; drift: number }[] = [];
    for (let j = 0; j < particleCount; j++) {
      const arm = j % 2;
      const t = Math.random();
      const radius = t * size;
      // Logarithmic spiral: tighter near the core, loosening outward.
      const angle = arm * Math.PI + Math.log(radius + 1) * 2.2 + (Math.random() - 0.5) * 0.4;
      particles.push({
        angle,
        radius,
        drift: 0.018 + Math.random() * 0.04,
      });
    }
    galaxies.push({
      cx: Math.random() * w,
      cy: Math.random() * h,
      armCount: 2,
      armSpread: 0.55 + Math.random() * 0.3,
      size,
      rotation: Math.random() * Math.PI * 2,
      rotationSpeed: (Math.random() - 0.5) * 0.0008,
      coreColor: palette[Math.min(1, palette.length - 1)] || { r: 255, g: 220, b: 180 },
      armColor: palette[Math.min(2, palette.length - 1)] || { r: 100, g: 160, b: 255 },
      particles,
      depth: Math.random(),
    });
  }

  // ── Black holes ────────────────────────────────────────────────────────
  // One to two Schwarzschild-rendered black holes with a Doppler-shifted
  // accretion disk. Inner orbits move faster (Kepler-like), the approaching
  // side of the disk is brighter (relativistic beaming).
  const blackHoles: BlackHole[] = [];
  const bhCount = 1 + Math.floor(Math.random() * 2);
  for (let i = 0; i < bhCount; i++) {
    const diskInner = 26;
    const diskOuter = 78;
    const armParticles: { angle: number; radius: number; speed: number; size: number }[] = [];
    for (let j = 0; j < 220; j++) {
      const t = Math.random();
      const radius = diskInner + t * (diskOuter - diskInner);
      armParticles.push({
        angle: Math.random() * Math.PI * 2,
        radius,
        speed: 0.014 + (1 - t) * 0.022,
        size: 0.8 + Math.random() * 1.4,
      });
    }
    // The diskTilt controls how "edge-on" the disk looks — moderate tilts
    // give the strongest gravitational-lensing imagery because we can see
    // both the top and bottom of the ring.
    blackHoles.push({
      x: Math.random() * w * 0.7 + w * 0.15,
      y: Math.random() * h * 0.6 + h * 0.2,
      swayX: Math.random() * w,
      swayY: Math.random() * h,
      radius: 18,
      diskInnerRadius: diskInner,
      diskOuterRadius: diskOuter,
      driftAngle: Math.random() * Math.PI * 2,
      driftSpeed: 0.04 + Math.random() * 0.05,
      driftRadiusX: 14 + Math.random() * 20,
      driftRadiusY: 10 + Math.random() * 14,
      driftPhaseX: Math.random() * Math.PI * 2,
      driftPhaseY: Math.random() * Math.PI * 2,
      rotation: Math.random() * Math.PI * 2,
      rotationSpeed: 0.006,
      // Fall back to physically-motivated colors when palette is sparse.
      diskHot:
        palette[Math.min(2, palette.length - 1)] || { r: 255, g: 180, b: 60 },
      diskCool:
        palette[Math.min(3, palette.length - 1)] || { r: 90, g: 140, b: 255 },
      ringColor:
        palette[Math.min(1, palette.length - 1)] || { r: 255, g: 230, b: 180 },
      armParticles,
      diskTilt: -0.55, // tilt the disk plane so we can see lensing
      cachedDisk: null,
      cachedHot: { r: 0, g: 0, b: 0 },
      cachedCool: { r: 0, g: 0, b: 0 },
      cachedOuter: 0,
      cachedInner: 0,
      cachedTilt: 0,
    });
  }

  // ── Nebula clouds ──────────────────────────────────────────────────────
  // Three large drifting colored blobs in the deep background.
  const nebulae: NebulaCloud[] = [];
  for (let i = 0; i < 3; i++) {
    nebulae.push({
      x: Math.random() * w,
      y: Math.random() * h,
      radius: 220 + Math.random() * 200,
      color: palette[i % palette.length] || { r: 120, g: 100, b: 200 },
      phaseX: Math.random() * Math.PI * 2,
      phaseY: Math.random() * Math.PI * 2,
      speedX: 0.0002 + Math.random() * 0.0003,
      speedY: 0.0002 + Math.random() * 0.0003,
      ampX: 60 + Math.random() * 80,
      ampY: 60 + Math.random() * 80,
      tint: (["warm", "cool", "violet"] as const)[i % 3],
    });
  }

  // ── Hypernova (developing colourful supernova) ─────────────────────────
  // Start somewhere in the empty corner of the canvas — never on top of the
  // hero text. Spawn at a random phase so first-load already feels alive.
  const hx = w * 0.5 + (Math.random() - 0.5) * w * 0.55;
  const hy = h * 0.3 + (Math.random() - 0.5) * h * 0.4;
  const filaments: HypernovaFilament[] = [];
  for (let i = 0; i < HN_FILAMENT_COUNT; i++) {
    filaments.push({
      // Squish slightly along an axis so it reads as a bipolar/jet outflow.
      angle: Math.random() * Math.PI * 2,
      speed: (0.5 + Math.random() * 1.0) * HN_EJECTA_SPEED,
      length: 14 + Math.random() * 36,
      hue: i,
      noisePhase: Math.random() * Math.PI * 2,
      wobbleAmp: 0.1 + Math.random() * 0.3,
      wobblePhase: Math.random() * Math.PI * 2,
      wobbleSpeed: 0.5 + Math.random() * 1.5,
    });
  }
  const hypernova: Hypernova = {
    cx: hx,
    cy: hy,
    birth: performance.now() - Math.random() * HN_CYCLE_MS,
    phase: 0,
    phaseStart: performance.now(),
    state: "progenitor",
    filaments,
    progenitorSize: 1.0,
    progenitorPulse: Math.random() * Math.PI * 2,
    beamPattern: null,
    beamHue: 0,
    shockwave: null,
    driftAngle: Math.random() * Math.PI * 2,
    driftSpeed: 0.04 + Math.random() * 0.05,
    driftRadius: 40 + Math.random() * 40,
    driftCx: hx,
    driftCy: hy,
  };

  return {
    stars,
    galaxies,
    blackHoles,
    nebulae,
    shockwaves: [],
    lastShockwaveAt: 0,
    hypernova,
  };
}

// Build (or rebuild) the cached radial beam sprite used during the ejecta
// phase. Drawing 220 filaments per frame at low cost requires us to bake the
// radial-gradient column into an offscreen once.
function buildBeamPattern(
  hypernova: Hypernova,
  hue: RGB,
  innerR: number
): HTMLCanvasElement {
  const W = 32;
  const H = 160;
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const cx = W / 2;
  const ctx = c.getContext("2d")!;
  // Stretched radial: bright core, fading outward along Y.
  const g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, rgba({ r: 255, g: 255, b: 255 }, 1));
  g.addColorStop(0.05, rgba(hue, 0.95));
  g.addColorStop(0.35, rgba(hue, 0.55));
  g.addColorStop(0.75, rgba(hue, 0.18));
  g.addColorStop(1, rgba(hue, 0));
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.ellipse(cx, H * 0.5, W * 0.45, H * 0.5, 0, 0, Math.PI * 2);
  ctx.fill();
  // Soft inner core
  const core = ctx.createRadialGradient(cx, H * 0.5, 0, cx, H * 0.5, innerR);
  core.addColorStop(0, "rgba(255,255,255,0.9)");
  core.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = core;
  ctx.beginPath();
  ctx.arc(cx, H * 0.5, innerR, 0, Math.PI * 2);
  ctx.fill();
  return c;
}

// Apply a gravitational-lens warp to the canvas: pixels closer to a black
// hole are bent tangentially around it. We do this with a coarse pixel pass
// (every 4th row / column) for performance — the visual effect is forgiving.
function applyLensing(
  ctx: CanvasRenderingContext2D,
  bh: BlackHole,
  w: number,
  h: number,
  mouse: { x: number; y: number }
) {
  const Rs = bh.radius * 2.6;        // Schwarzschild-like influence radius
  const innerCutoff = bh.radius * 1.2;

  // Mouse adds a small directional warp on top of the static well.
  const mdx = mouse.x - bh.x;
  const mdy = mouse.y - bh.y;
  const mdist = Math.sqrt(mdx * mdx + mdy * mdy);
  let mouseWarp = 0;
  let mAngle = 0;
  if (mdist < Rs * 1.4 && mdist > 0) {
    mouseWarp = (1 - mdist / (Rs * 1.4)) * 6;
    mAngle = Math.atan2(mdy, mdx);
  }

  // Sample copy: shift pixels that fall within the well around the BH.
  const step = 2; // trade-off speed <-> smoothness
  let yStart = Math.max(0, Math.floor(bh.y - Rs));
  let yEnd = Math.min(h, Math.ceil(bh.y + Rs) + 1);
  let xStart = Math.max(0, Math.floor(bh.x - Rs));
  let xEnd = Math.min(w, Math.ceil(bh.x + Rs) + 1);
  // Clamp further so we never iterate off the canvas
  if (yEnd < 0) return;
  if (xEnd < 0) return;

  for (let y = yStart; y < yEnd; y += step) {
    for (let x = xStart; x < xEnd; x += step) {
      const dx = x - bh.x;
      const dy = y - bh.y;
      const r = Math.sqrt(dx * dx + dy * dy);
      if (r > Rs || r < 0.001) continue;
      if (r < innerCutoff) continue;

      // Schwarzschild-like deflection: 1/r^2 dominates near the horizon,
      // tapering to zero at Rs.
      const u = (r - innerCutoff) / (Rs - innerCutoff);
      const strength = 0.6 * (1 - u) * (1 - u) * (Rs / r) * (Rs / r) * 6;
      const ang = Math.atan2(dy, dx);

      // Sample a slightly rotated & pulled-in pixel.
      const sampleR = r + strength * 1.5;
      const sampleA = ang + strength * 0.9 + mouseWarp * Math.sin(ang - mAngle) * 0.04;
      const sx = bh.x + Math.cos(sampleA) * sampleR;
      const sy = bh.y + Math.sin(sampleA) * sampleR;
      if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;

      // Pull pixels; srcRect tiny because we sample step-by-step.
      try {
        ctx.drawImage(
          ctx.canvas,
          sx,
          sy,
          step,
          step,
          x,
          y,
          step,
          step
        );
      } catch {
        /* empty (canvas tainted in some browsers) */
      }
    }
  }
}

// Draw a single hypernova sprite given its current state and timing.
function renderHypernova(
  ctx: CanvasRenderingContext2D,
  hn: Hypernova,
  palette: RGB[],
  w: number,
  h: number,
  time: number,
  dt: number
) {
  // Drift the hypernova gently so it doesn't sit dead-center forever.
  hn.driftAngle += hn.driftSpeed * (dt / 1000);
  const cx = hn.driftCx + Math.cos(hn.driftAngle) * hn.driftRadius;
  const cy = hn.driftCy + Math.sin(hn.driftAngle * 1.3) * hn.driftRadius * 0.6;
  hn.cx = cx;
  hn.cy = cy;

  // Decide the active phase from elapsed time since birth.
  const elapsed = (time - hn.birth) % HN_CYCLE_MS;
  let phaseT = 0;
  if (elapsed < HN_PROGENITOR_MS) {
    if (hn.state !== "progenitor") {
      hn.state = "progenitor";
      hn.phaseStart = time;
    }
    phaseT = elapsed / HN_PROGENITOR_MS;
  } else if (elapsed < HN_PROGENITOR_MS + HN_COLLAPSE_MS) {
    if (hn.state !== "collapse") {
      hn.state = "collapse";
      hn.phaseStart = time;
    }
    phaseT = (elapsed - HN_PROGENITOR_MS) / HN_COLLAPSE_MS;
  } else if (elapsed < HN_PROGENITOR_MS + HN_COLLAPSE_MS + HN_FLASH_MS) {
    if (hn.state !== "flash") {
      hn.state = "flash";
      hn.phaseStart = time;
      // Trigger shockwave at the moment of brightest flash.
      hn.shockwave = {
        x: cx,
        y: cy,
        age: 0,
        maxAge: 2800,
        color: { r: 255, g: 240, b: 220 },
      };
      // Build the beam sprite for the upcoming ejecta phase.
      hn.beamHue = 2;
      hn.beamPattern = buildBeamPattern(
        hn,
        { r: 255, g: 220, b: 180 },
        20
      );
    }
    phaseT =
      (elapsed - HN_PROGENITOR_MS - HN_COLLAPSE_MS) / HN_FLASH_MS;
  } else if (
    elapsed <
    HN_PROGENITOR_MS + HN_COLLAPSE_MS + HN_FLASH_MS + HN_EJECTA_MS
  ) {
    if (hn.state !== "ejecta") {
      hn.state = "ejecta";
      hn.phaseStart = time;
    }
    phaseT =
      (elapsed -
        HN_PROGENITOR_MS -
        HN_COLLAPSE_MS -
        HN_FLASH_MS) /
      HN_EJECTA_MS;
  } else if (
    elapsed <
    HN_PROGENITOR_MS +
      HN_COLLAPSE_MS +
      HN_FLASH_MS +
      HN_EJECTA_MS +
      HN_REMNANT_MS
  ) {
    if (hn.state !== "remnant") {
      hn.state = "remnant";
      hn.phaseStart = time;
    }
    phaseT =
      (elapsed -
        HN_PROGENITOR_MS -
        HN_COLLAPSE_MS -
        HN_FLASH_MS -
        HN_EJECTA_MS) /
      HN_REMNANT_MS;
  } else {
    if (hn.state !== "quiescent") {
      hn.state = "quiescent";
      hn.phaseStart = time;
    }
    phaseT =
      (elapsed -
        HN_PROGENITOR_MS -
        HN_COLLAPSE_MS -
        HN_FLASH_MS -
        HN_EJECTA_MS -
        HN_REMNANT_MS) /
      HN_QUIESCENT_MS;
  }
  hn.phase = phaseT;
  void hn.phaseStart;

  // ═══════════════════════════════════════════════════════════════════════
  // Render each phase
  // ═══════════════════════════════════════════════════════════════════════
  ctx.save();
  ctx.globalCompositeOperation = "lighter";

  // PROGENITOR — a single bright star that brightens & gently pulses.
  if (hn.state === "progenitor") {
    const brighten = 0.35 + phaseT * 0.65; // 0.35 -> 1.0
    hn.progenitorPulse += 0.018;
    const pulse = 0.85 + Math.sin(hn.progenitorPulse) * 0.15;
    const size = 2.4 * pulse;
    const color = STELLAR_COLORS[5]; // orange giant
    // Soft halo
    const halo = ctx.createRadialGradient(cx, cy, 0, cx, cy, 50);
    halo.addColorStop(0, rgba(color, 0.65 * brighten));
    halo.addColorStop(0.4, rgba(color, 0.25 * brighten));
    halo.addColorStop(1, rgba(color, 0));
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(cx, cy, 50, 0, Math.PI * 2);
    ctx.fill();
    // Inner core
    ctx.fillStyle = rgba({ r: 255, g: 230, b: 200 }, brighten);
    ctx.beginPath();
    ctx.arc(cx, cy, size, 0, Math.PI * 2);
    ctx.fill();
    // Diffraction spikes for the "bright star" feel
    ctx.strokeStyle = rgba(color, 0.5 * brighten);
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.moveTo(cx - 18, cy);
    ctx.lineTo(cx + 18, cy);
    ctx.moveTo(cx, cy - 18);
    ctx.lineTo(cx, cy + 18);
    ctx.stroke();
  }

  // COLLAPSE — star contracts (bright glow dimming then vanishing).
  if (hn.state === "collapse") {
    const pulse = 0.85 + Math.sin(hn.progenitorPulse) * 0.15;
    const shrink = 1 - phaseT * 0.7;
    const fadeOut = 1 - phaseT;
    const color = STELLAR_COLORS[6];
    ctx.fillStyle = rgba(color, 0.55 * pulse * fadeOut);
    ctx.beginPath();
    ctx.arc(cx, cy, 16 * shrink, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = rgba({ r: 255, g: 200, b: 180 }, fadeOut);
    ctx.beginPath();
    ctx.arc(cx, cy, 3 * shrink, 0, Math.PI * 2);
    ctx.fill();
  }

  // FLASH — blinding burst with bright falloff + dark absorption basin.
  if (hn.state === "flash") {
    // peak in the middle of the phase, dim at the edges
    const peak = Math.sin(phaseT * Math.PI);
    const flashR = 60 + phaseT * 220;
    // Big radial blast
    const flash = ctx.createRadialGradient(cx, cy, 0, cx, cy, flashR);
    flash.addColorStop(0, "rgba(255,255,255," + 0.95 * peak + ")");
    flash.addColorStop(0.2, "rgba(255,240,200," + 0.7 * peak + ")");
    flash.addColorStop(0.6, "rgba(255,140,80," + 0.3 * peak + ")");
    flash.addColorStop(1, "rgba(255,80,40,0)");
    ctx.fillStyle = flash;
    ctx.beginPath();
    ctx.arc(cx, cy, flashR, 0, Math.PI * 2);
    ctx.fill();
    // Bright pinhead
    ctx.fillStyle = "rgba(255,255,255," + peak + ")";
    ctx.beginPath();
    ctx.arc(cx, cy, 6, 0, Math.PI * 2);
    ctx.fill();
    // Two opposing relativistic jets forming — they fade to give way to the
    // proper ejecta beam pattern next phase.
    ctx.save();
    ctx.translate(cx, cy);
    for (const sign of [-1, 1]) {
      ctx.save();
      ctx.rotate(sign * 0.3);
      const jet = ctx.createLinearGradient(0, 0, 0, sign * 160);
      jet.addColorStop(0, "rgba(255,255,255," + 0.9 * peak + ")");
      jet.addColorStop(0.4, "rgba(180,210,255," + 0.5 * peak + ")");
      jet.addColorStop(1, "rgba(120,160,255,0)");
      ctx.fillStyle = jet;
      ctx.beginPath();
      ctx.ellipse(0, sign * 60, 14 * peak, 100 * peak, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
    ctx.restore();
  }

  // EJECTA — filaments racing outward using the cached beam sprite.
  if (hn.state === "ejecta") {
    const expand = phaseT * 1.0; // 0 -> 1
    const beam = hn.beamPattern;
    if (beam) {
      for (const f of hn.filaments) {
        const dist = f.speed * (HN_EJECTA_MS / 1000) * expand;
        const x = cx + Math.cos(f.angle) * dist;
        const y = cy + Math.sin(f.angle) * dist * 0.85;
        const hue =
          // early-phase hot core, mid-phase teal, late-phase violet.
          expand < 0.3
            ? { r: 255, g: 240, b: 200 }
            : expand < 0.65
              ? { r: 130, g: 200, b: 255 }
              : { r: 180, g: 110, b: 255 };
        const fade = 1 - Math.pow(expand, 1.6);
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(f.angle + Math.PI / 2);
        ctx.globalAlpha = fade * 0.85;
        ctx.drawImage(beam, -16, -80);
        ctx.restore();
      }
      // Reset alpha — we used globalAlpha above for the beam draws.
      ctx.globalAlpha = 1;
    }
    // Persistent glow at the centre as the ejecta thins out.
    const coreGlow = ctx.createRadialGradient(cx, cy, 0, cx, cy, 60);
    coreGlow.addColorStop(0, "rgba(255,200,140,0.85)");
    coreGlow.addColorStop(0.5, "rgba(255,120,80,0.35)");
    coreGlow.addColorStop(1, "rgba(255,80,40,0)");
    ctx.fillStyle = coreGlow;
    ctx.beginPath();
    ctx.arc(cx, cy, 60, 0, Math.PI * 2);
    ctx.fill();
  }

  // REMNANT — a slowly expanding, multi-coloured nebula.
  if (hn.state === "remnant") {
    // ease-out so it grows fast then slows
    const easeOut = 1 - Math.pow(1 - phaseT, 2);
    const R = 30 + easeOut * HN_REMNANT_EXPAND;
    // Two overlapping colored blobs: blue & orange for a colourful nebula.
    const c1 = { r: 110, g: 170, b: 255 };
    const c2 = { r: 255, g: 140, b: 100 };
    const a1 = 0.55 * (1 - phaseT * 0.7);
    const a2 = 0.45 * (1 - phaseT * 0.7);
    const g1 = ctx.createRadialGradient(cx - 8, cy - 4, 0, cx - 8, cy - 4, R);
    g1.addColorStop(0, rgba(c1, a1));
    g1.addColorStop(0.6, rgba(c1, a1 * 0.4));
    g1.addColorStop(1, rgba(c1, 0));
    ctx.fillStyle = g1;
    ctx.beginPath();
    ctx.arc(cx - 8, cy - 4, R, 0, Math.PI * 2);
    ctx.fill();
    const g2 = ctx.createRadialGradient(cx + 12, cy + 8, 0, cx + 12, cy + 8, R);
    g2.addColorStop(0, rgba(c2, a2));
    g2.addColorStop(0.6, rgba(c2, a2 * 0.4));
    g2.addColorStop(1, rgba(c2, 0));
    ctx.fillStyle = g2;
    ctx.beginPath();
    ctx.arc(cx + 12, cy + 8, R, 0, Math.PI * 2);
    ctx.fill();
    // Pulsing central core as the neutron star spins down.
    const pulse = 0.55 + Math.sin(time * 0.012) * 0.45;
    ctx.fillStyle = "rgba(200,230,255," + 0.9 * pulse + ")";
    ctx.beginPath();
    ctx.arc(cx, cy, 1.8, 0, Math.PI * 2);
    ctx.fill();
  }

  // QUIESCENT — fades to nothing before the cycle restarts.
  if (hn.state === "quiescent") {
    const fade = 1 - phaseT;
    const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, 70);
    glow.addColorStop(0, "rgba(180,200,255," + 0.35 * fade + ")");
    glow.addColorStop(1, "rgba(180,200,255,0)");
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(cx, cy, 70, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();

  // Update the live shockwave owned by the hypernova — drawn here so it
  // ages in lock-step with the parent state machine.
  if (hn.shockwave) {
    const sh = hn.shockwave;
    sh.age += dt;
    const t = sh.age / sh.maxAge;
    if (t >= 1) {
      hn.shockwave = null;
    } else {
      const radius = t * 420;
      const fade = 1 - t;
      ctx.save();
      ctx.globalCompositeOperation = "lighter";
      const colors = [
        sh.color,
        { r: 180, g: 220, b: 255 },
        { r: 255, g: 255, b: 255 },
      ];
      for (let k = 0; k < 3; k++) {
        ctx.strokeStyle = rgba(colors[k], fade * 0.5);
        ctx.lineWidth = 2.4 - k * 0.6;
        ctx.beginPath();
        ctx.arc(sh.x, sh.y, radius - k * 16, 0, Math.PI * 2);
        ctx.stroke();
      }
      // Inner flash
      const innerFade = Math.max(0, 1 - t * 4);
      if (innerFade > 0) {
        const g = ctx.createRadialGradient(sh.x, sh.y, 0, sh.x, sh.y, 80);
        g.addColorStop(0, rgba(sh.color, innerFade * 0.7));
        g.addColorStop(1, rgba(sh.color, 0));
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(sh.x, sh.y, 80, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }
  }
}

// Render a black hole with a tilted, Doppler-shifted accretion disk and
// gravitational-lensing overlay. The lensing pass is intentionally coarse so
// we can keep it realtime.
function renderBlackHole(
  ctx: CanvasRenderingContext2D,
  bh: BlackHole,
  w: number,
  h: number,
  mouse: { x: number; y: number },
  reduced: boolean
) {
  if (!reduced) bh.rotation += bh.rotationSpeed;

  // 1. Photon ring + outer lensing arc — drawn on top of warped starfield.
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.strokeStyle = rgba(bh.ringColor, 0.95);
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ctx.arc(bh.x, bh.y, bh.radius + 2, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = rgba(bh.ringColor, 0.4);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(bh.x, bh.y, bh.radius + 9, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();

  // 2. Gravitational lensing — pull surrounding pixels around the well.
  if (!reduced) applyLensing(ctx, bh, w, h, mouse);

  // 3. Tilted accretion disk with Doppler brightness asymmetry.
  ctx.save();
  ctx.translate(bh.x, bh.y);
  // Disk tilt compresses Y so the disk looks elliptical when viewed obliquely.
  ctx.scale(1, Math.cos(bh.diskTilt));
  for (const p of bh.armParticles) {
    const angle = p.angle + (timeNow() * p.speed) / 1000;
    const x = Math.cos(angle) * p.radius;
    const y = Math.sin(angle) * p.radius;
    const t = (p.radius - bh.diskInnerRadius) / (bh.diskOuterRadius - bh.diskInnerRadius);
    // Doppler: the side rotating toward the viewer (sin(angle) > 0 in our
    // setup) is brighter & bluer. The receding side dims and reddens.
    const dopplerBoost = 1 + 0.55 * Math.sin(angle);
    // Hot-to-cool gradient based on disk radius.
    let rr = Math.round(bh.diskHot.r * (1 - t) + bh.diskCool.r * t);
    let gg = Math.round(bh.diskHot.g * (1 - t) + bh.diskCool.g * t);
    let bb = Math.round(bh.diskHot.b * (1 - t) + bh.diskCool.b * t);
    rr = Math.min(255, Math.round(rr * dopplerBoost));
    gg = Math.min(255, Math.round(gg * dopplerBoost));
    bb = Math.min(255, Math.round(bb * (0.6 + 0.4 * dopplerBoost)));
    // Keplerian fall-off: brightness drops with r^(-3/4)-ish.
    const fade = Math.pow(1 - t, 0.7) * 0.9;
    ctx.fillStyle = `rgba(${rr}, ${gg}, ${bb}, ${fade})`;
    ctx.beginPath();
    ctx.arc(x, y, p.size, 0, Math.PI * 2);
    ctx.fill();
  }
  // Re-draw the disk's underside: a mirrored, dimmer arc to suggest the
  // disk goes "behind" the event horizon on the far side.
  ctx.scale(1, -1);
  ctx.globalAlpha = 0.35;
  for (const p of bh.armParticles) {
    const angle = p.angle + (timeNow() * p.speed) / 1000 + Math.PI;
    const x = Math.cos(angle) * p.radius;
    const y = Math.sin(angle) * p.radius;
    const t = (p.radius - bh.diskInnerRadius) / (bh.diskOuterRadius - bh.diskInnerRadius);
    const rr = Math.round(bh.diskHot.r * (1 - t) + bh.diskCool.r * t);
    const gg = Math.round(bh.diskHot.g * (1 - t) + bh.diskCool.g * t);
    const bb = Math.round(bh.diskHot.b * (1 - t) + bh.diskCool.b * t);
    const fade = Math.pow(1 - t, 0.7) * 0.4;
    ctx.fillStyle = `rgba(${rr}, ${gg}, ${bb}, ${fade})`;
    ctx.beginPath();
    ctx.arc(x, y, p.size * 0.7, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();

  // 4. Event horizon — pure black sphere on top of everything.
  ctx.fillStyle = "rgba(0, 0, 0, 1)";
  ctx.beginPath();
  ctx.arc(bh.x, bh.y, bh.radius, 0, Math.PI * 2);
  ctx.fill();

  // 5. Subtle inner shadow ring — gives the "swallowing" rim more contrast.
  ctx.save();
  ctx.globalCompositeOperation = "multiply";
  const shadow = ctx.createRadialGradient(
    bh.x,
    bh.y,
    bh.radius * 0.6,
    bh.x,
    bh.y,
    bh.radius * 1.6
  );
  shadow.addColorStop(0, "rgba(0,0,0,1)");
  shadow.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = shadow;
  ctx.beginPath();
  ctx.arc(bh.x, bh.y, bh.radius * 1.6, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// Wrapper so the BH module can read the loop's clock without coupling it
// through 11 extra args — kept as a singleton set by renderCosmos.
let __cosmosClock = 0;
function timeNow() {
  return __cosmosClock;
}

function renderCosmos(
  ctx: CanvasRenderingContext2D,
  state: CosmosState,
  palette: RGB[],
  w: number,
  h: number,
  mouse: { x: number; y: number },
  interactive: boolean,
  reduced: boolean,
  time: number,
  dt: number
) {
  __cosmosClock = time;

  // Slight motion-blur: clear with a low-alpha black instead of full clear,
  // so trails from super-bright events decay naturally over ~5 frames.
  ctx.fillStyle = "rgba(0, 0, 0, 0.92)";
  ctx.fillRect(0, 0, w, h);

  const { stars, galaxies, blackHoles, nebulae, shockwaves, hypernova } = state;

  // ── 1. Nebula atmosphere ─────────────────────────────────────────────
  for (const n of nebulae) {
    if (!reduced) {
      n.phaseX += n.speedX;
      n.phaseY += n.speedY;
    }
    const x = n.x + Math.sin(n.phaseX) * n.ampX;
    const y = n.y + Math.cos(n.phaseY) * n.ampY;
    const g = ctx.createRadialGradient(x, y, 0, x, y, n.radius);
    g.addColorStop(0, rgba(n.color, 0.07));
    g.addColorStop(0.5, rgba(n.color, 0.03));
    g.addColorStop(1, rgba(n.color, 0));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, n.radius, 0, Math.PI * 2);
    ctx.fill();
  }

  // ── 2. Starfield ────────────────────────────────────────────────────
  // Stars at depth=0 are pure background; depth=1 stars are near-camera and
  // shift with the cursor (parallax). Keeps the cosmos feeling layered.
  const parX = interactive && !reduced ? (mouse.x - w / 2) * 0.02 : 0;
  const parY = interactive && !reduced ? (mouse.y - h / 2) * 0.02 : 0;
  for (const s of stars) {
    if (!reduced) s.phase += s.twinkleSpeed;
    const tw = 0.55 + Math.sin(s.phase) * 0.45;
    const a = s.baseBrightness * tw;
    const px = s.x + parX * s.depth;
    const py = s.y + parY * s.depth;
    const color =
      s.hue > 0 && s.hue < STELLAR_COLORS.length
        ? STELLAR_COLORS[s.hue]
        : { r: 255, g: 255, b: 255 };
    if (s.spike) {
      // Bright stars get a 4-point diffraction spike + soft halo.
      ctx.save();
      ctx.globalCompositeOperation = "lighter";
      const halo = ctx.createRadialGradient(px, py, 0, px, py, s.size * 3.2);
      halo.addColorStop(0, rgba(color, a * 0.6));
      halo.addColorStop(1, rgba(color, 0));
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(px, py, s.size * 3.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = rgba(color, a * 0.85);
      ctx.lineWidth = 0.7;
      ctx.beginPath();
      ctx.moveTo(px - s.size * 7, py);
      ctx.lineTo(px + s.size * 7, py);
      ctx.moveTo(px, py - s.size * 7);
      ctx.lineTo(px, py + s.size * 7);
      ctx.stroke();
      ctx.fillStyle = rgba(color, a);
      ctx.beginPath();
      ctx.arc(px, py, s.size, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    } else {
      ctx.fillStyle = rgba(color, a * 0.85);
      ctx.beginPath();
      ctx.arc(px, py, s.size, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // ── 3. Galaxies ─────────────────────────────────────────────────────
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  for (const g of galaxies) {
    if (!reduced) g.rotation += g.rotationSpeed;
    ctx.save();
    // Slight parallax on the galaxy cores too — they're "near-midfield".
    ctx.translate(g.cx + parX * 0.5, g.cy + parY * 0.5);
    ctx.rotate(g.rotation);
    // Bright nucleus
    const coreGrad = ctx.createRadialGradient(0, 0, 0, 0, 0, g.size * 0.22);
    coreGrad.addColorStop(0, rgba(g.coreColor, 0.85));
    coreGrad.addColorStop(1, rgba(g.coreColor, 0));
    ctx.fillStyle = coreGrad;
    ctx.beginPath();
    ctx.arc(0, 0, g.size * 0.22, 0, Math.PI * 2);
    ctx.fill();
    // Spec stars in the core: small inner dot.
    ctx.fillStyle = rgba({ r: 255, g: 245, b: 220 }, 0.95);
    ctx.beginPath();
    ctx.arc(0, 0, 1.4, 0, Math.PI * 2);
    ctx.fill();
    // Arm particles
    for (const p of g.particles) {
      const a = p.angle + p.drift * p.radius * 0.1;
      const x = Math.cos(a) * p.radius;
      const y = Math.sin(a) * p.radius * g.armSpread;
      const fade = Math.pow(1 - p.radius / g.size, 1.2) * 0.7;
      ctx.fillStyle = rgba(g.armColor, fade);
      ctx.beginPath();
      ctx.arc(x, y, 1.5 * fade + 0.4, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }
  ctx.restore();

  // ── 4. Hypernova (developing colourful supernova lifecycle) ─────────
  // Drawn before the black hole so the BH warps it during its peak flash.
  renderHypernova(ctx, hypernova, palette, w, h, time, dt);

  // ── 5. Black holes ──────────────────────────────────────────────────
  // Note: order matters — we draw BH behind the hypernova when the BH is
  // farther from camera (depth sort), but in practice a single BH with a
  // single hypernova reading at fixed position is what we ship; drawing BH
  // here is fine and lets the lensing pull in the just-rendered hypernova
  // pixels (the lensing pass runs after this draw).
  for (const bh of blackHoles) {
    renderBlackHole(ctx, bh, w, h, mouse, reduced);
  }

  // ── 6. Generic shockwaves (legacy radial bursts) ────────────────────
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  for (let i = shockwaves.length - 1; i >= 0; i--) {
    const sh = shockwaves[i];
    if (!reduced) sh.age += 16;
    const t = sh.age / sh.maxAge;
    if (t >= 1) {
      shockwaves.splice(i, 1);
      continue;
    }
    const radius = t * 380;
    const fade = 1 - t;
    const colors = [sh.color, { r: 180, g: 220, b: 255 }, { r: 255, g: 255, b: 255 }];
    for (let k = 0; k < 3; k++) {
      ctx.strokeStyle = rgba(colors[k], fade * 0.6);
      ctx.lineWidth = 2.5 - k * 0.6;
      ctx.beginPath();
      ctx.arc(sh.x, sh.y, radius - k * 14, 0, Math.PI * 2);
      ctx.stroke();
    }
    const innerFade = Math.max(0, 1 - t * 4);
    if (innerFade > 0) {
      const g = ctx.createRadialGradient(sh.x, sh.y, 0, sh.x, sh.y, 60);
      g.addColorStop(0, rgba(sh.color, innerFade * 0.7));
      g.addColorStop(1, rgba(sh.color, 0));
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(sh.x, sh.y, 60, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
}

// ── Strategy registry ─────────────────────────────────────────────────────

type StrategyKey =
  | "neural"
  | "embers"
  | "matrix"
  | "bubbles"
  | "cosmos";

const PALETTE_STRATEGY: Record<string, StrategyKey> = {
  "neon-cyber": "neural",
  "aurora-borealis": "neural",
  "cosmic-void": "neural",
  "solar-flare": "embers",
  "golden-hour": "embers",
  "rose-quartz": "embers",
  "matrix-green": "matrix",
  "ocean-depths": "bubbles",
  "deep-space": "cosmos",
};

// ── The exported component ────────────────────────────────────────────────

export function InteractiveBackground({
  className = "",
  palette = "neon-cyber",
}: {
  className?: string;
  palette?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const mouseRef = useRef<{ x: number; y: number }>({ x: -9999, y: -9999 });
  const stateRef = useRef<unknown>(null);
  const lastTimeRef = useRef(0);
  const [reducedMotion, setReducedMotion] = useState(false);
  const strategyKey: StrategyKey = PALETTE_STRATEGY[palette] || "neural";

  // Palette colors as RGB tuples — convert from HSL strings stored on
  // <html>. The provider writes these vars on the root, so we read them
  // through getComputedStyle to stay in sync.
  const paletteRGB = useMemo<RGB[]>(() => {
    if (typeof window === "undefined") return [];
    const root = getComputedStyle(document.documentElement);
    const keys = [
      "--palette-primary",
      "--palette-secondary",
      "--palette-tertiary",
      "--palette-accent",
    ];
    return keys
      .map((k) => root.getPropertyValue(k).trim())
      .filter(Boolean)
      .map(hslTriple);
  }, [palette]);

  // Subscribe to reduced-motion preference
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReducedMotion(mq.matches);
    const handler = (e: MediaQueryListEvent) => setReducedMotion(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  // Reset state when strategy or palette changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const w = window.innerWidth;
    const h = window.innerHeight;
    canvas.width = w;
    canvas.height = h;

    switch (strategyKey) {
      case "neural":
        stateRef.current = createNeuralNodes(80, w, h);
        break;
      case "embers":
        stateRef.current = createEmbers(140, w, h, paletteRGB);
        break;
      case "matrix":
        stateRef.current = createMatrixColumns(0, w, h);
        break;
      case "bubbles":
        stateRef.current = createBubbles(70, w, h, paletteRGB);
        break;
      case "cosmos":
        // Cosmos strategy is rendered by the CosmosWebGL component instead
        // of the 2D canvas. Don't allocate 2D state for it here.
        stateRef.current = null;
        break;
    }
    lastTimeRef.current = 0;
  }, [strategyKey, paletteRGB]);

  const animate = useCallback(
    (time: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const w = canvas.width;
      const h = canvas.height;
      const dt = lastTimeRef.current === 0 ? 16 : time - lastTimeRef.current;
      lastTimeRef.current = time;

      const state = stateRef.current;
      const mouse = mouseRef.current;
      const reduced = reducedMotion;

      switch (strategyKey) {
        case "neural":
          renderNeuralNodes(
            ctx,
            state as NeuralNode[],
            paletteRGB,
            w,
            h,
            mouse,
            true,
            reduced,
            time
          );
          break;
        case "embers":
          renderEmbers(
            ctx,
            state as Ember[],
            paletteRGB,
            w,
            h,
            mouse,
            true,
            reduced,
            time
          );
          break;
        case "matrix":
          renderMatrixGlyphs(
            ctx,
            state as MatrixColumn[],
            paletteRGB,
            w,
            h,
            mouse,
            true,
            reduced,
            time
          );
          break;
        case "bubbles":
          renderBubbles(
            ctx,
            state as Bubble[],
            paletteRGB,
            w,
            h,
            mouse,
            true,
            reduced,
            time
          );
          break;
        case "cosmos":
          // Cosmos strategy is rendered by the CosmosWebGL component, not
          // the 2D canvas. The canvas is hidden for this strategy so we
          // skip rendering entirely here.
          return;
      }

      animationRef.current = requestAnimationFrame(animate);
    },
    [strategyKey, paletteRGB, reducedMotion]
  );

  useEffect(() => {
    // The cosmos strategy owns its own WebGL RAF loop in CosmosWebGL.
    // Don't run the 2D canvas loop for it — it would burn frames against
    // a hidden canvas.
    if (strategyKey === "cosmos") {
      return;
    }
    animationRef.current = requestAnimationFrame(animate);
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [animate, strategyKey]);

  // Mouse + touch tracking (skipped for cosmos — CosmosWebGL handles its own)
  useEffect(() => {
    if (strategyKey === "cosmos") return;
    const onMove = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY };
    };
    const onTouch = (e: TouchEvent) => {
      const t = e.touches[0];
      if (t) mouseRef.current = { x: t.clientX, y: t.clientY };
    };
    const onResize = () => {
      const c = canvasRef.current;
      if (!c) return;
      c.width = window.innerWidth;
      c.height = window.innerHeight;
    };
    window.addEventListener("mousemove", onMove, { passive: true });
    window.addEventListener("touchmove", onTouch, { passive: true });
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("touchmove", onTouch);
      window.removeEventListener("resize", onResize);
    };
  }, [strategyKey]);

  return (
    <>
      {strategyKey === "cosmos" ? (
        // WebGL2/WebGL1 renderer for the Deep Space palette: realistic
        // accretion disk with gravitational lensing, hypernova lifecycle,
        // and a parallax starfield. Falls back internally to a static
        // starfield if the GPU context cannot be created.
        <CosmosWebGL className={`fixed inset-0 -z-10 pointer-events-none ${className}`} />
      ) : (
        <canvas
          ref={canvasRef}
          className={`fixed inset-0 -z-10 pointer-events-none ${className}`}
          aria-hidden="true"
        />
      )}
    </>
  );
}

// Legacy exports kept so existing imports don't break.
export function ParticleBackground(_: {
  className?: string;
  palette?: string;
  interactive?: boolean;
}) {
  return null;
}

export function GridBackground(_: {
  className?: string;
  palette?: string;
  interactive?: boolean;
}) {
  return null;
}