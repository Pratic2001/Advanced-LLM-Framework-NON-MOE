"use client";

// ── Aurora Borealis — the northern-lights world for "aurora-borealis" ─────
//
// Self-contained WebGL background modeled on CosmosWebGL.tsx's skeleton. Scene:
// high-poly aurora curtains overhead, a dark low-poly mountain ridge, sparse
// twinkling stars, occasional shooting stars, and a click-triggered brightness
// wave that ripples across the curtains.
//
// The curtains are no longer flat painted planes. Each is a genuinely
// high-poly sheet (~8k vertices) whose vertex shader displaces every vertex
// along the sheet normal with real curtain physics (multi-octave fold drapes
// with a field-line lean, a Gaussian-windowed travelling undulation "surf"
// packet, and a click-ripple bulge). The fragment shader then ray-marches a
// thin emitting volume along the line of sight: the march window scales with
// view angle (edge-on rays integrate a long path through the sheet → real
// edge-on brightening and fold self-shadowing), and each sample re-evaluates
// the same displacement field so the folds genuinely occlude each other.
// Colors read live from --palette-* so the PaletteEditor retunes the lights
// in real time.

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { supportsWebGL } from "./webglSupport";
import { readPaletteColors, makePalettePoller } from "./worldUtils";
import { emitCosmosEvent } from "./cosmosEvents";

// ── Curtain physics / ray-march tuning ─────────────────────────────────────
// Kept at the top of scope so they can be tuned in place without touching
// shader strings (same pattern as OceanWebGL's wave constants).
const FOLD_AMP = 13; // geometric fold drape depth (world units)
const FOLD_FREQ = 0.055; // folds per world unit along the curtain length
const LEAN = 0.32; // field-line tilt: horizontal shear per unit of altitude
const SURF_AMP = 4.2; // travelling undulation packet amplitude
const RIPPLE_AMP = 5.0; // click-ripple bulge amplitude
const THICKNESS = 1.25; // emitting sheet half-thickness (world units)
const MARCH_STEPS = 28; // ray-march integration steps per pixel
const CURTAIN_EMIT = 1.0; // radiance scale of the integrated emission
const RAY_FREQ = 0.42; // vertical ray striation frequency
const RAY_TILT = 1.3; // field-aligned lean of the ray striations
const PULSE_COLOR = 0.45; // click-pulse colour boost
const FLASH_AMT = 0.55; // whole-scene flash boost
const LAKE_FRESNEL_POW = 3.0; // frozen-lake fresnel falloff

// ── Frozen-lake planar-mirror vertex + fragment shaders ────────────────────
const LAKE_VERT = `
varying vec3 vWorld;
void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorld = wp.xyz;
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;
const LAKE_FRAG = `
#define LAKE_FRESNEL ${LAKE_FRESNEL_POW}
uniform float uTime;
uniform vec3 uCamPos;
uniform mat4 uProj;
uniform mat4 uMirrorView;
uniform sampler2D uLakeTex;
uniform vec3 uColorA;
varying vec3 vWorld;
float hashL(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noiseL(vec2 x){
  vec2 i = floor(x); vec2 f = fract(x);
  f = f*f*(3.0-2.0*f);
  return mix(mix(hashL(i), hashL(i+vec2(1,0)), f.x),
             mix(hashL(i+vec2(0,1)), hashL(i+vec2(1,1)), f.x), f.y);
}
void main() {
  vec4 clip = uProj * (uMirrorView * vec4(vWorld, 1.0));
  vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
  bool inMirror = clip.w > 0.01 && uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;

  // Ice surface: faint ripples/frost displace the sample so it's not a hard mirror.
  vec2 rip = vec2(noiseL(vWorld.xz * 0.8 + vec2(uTime * 0.02, 0.0)),
                  noiseL(vWorld.xz * 0.8 + vec2(0.0, uTime * 0.02)));
  uv += (rip - 0.5) * 0.012;

  vec3 refl = vec3(0.0);
  if (inMirror) {
    refl = texture2D(uLakeTex, clamp(uv, 0.004, 0.996)).rgb;
    // Fresnel: grazing horizons reflect the aurora most; near-field is flat ice.
    vec3 V = normalize(uCamPos - vWorld);
    float fres = 0.04 + 0.96 * pow(1.0 - max(dot(V, vec3(0.0, 1.0, 0.0)), 0.0), LAKE_FRESNEL);
    refl *= fres;
  }
  // Icy body colour + frost-crack veins.
  vec3 ice = vec3(0.02, 0.028, 0.05);
  float crack = smoothstep(0.30, 0.0, noiseL(vWorld.xz * 0.05)) * 0.35
              + smoothstep(0.35, 0.0, noiseL(vWorld.xz * 0.19 + 11.0)) * 0.3;
  vec3 col = ice + refl + uColorA * crack * 0.18;
  // Horizon melt to the scene clear colour (0x030610) so the ice meets the sky
  // with no visible seam once the mountains are absent from the horizon.
  float horizon = 1.0 - smoothstep(-72.0, -40.0, vWorld.z);
  col = mix(col, vec3(0.012, 0.024, 0.063), horizon);
  gl_FragColor = vec4(col, 1.0);
}
`;

// Shared noise / fbm helpers (the project's established GLSL idiom).
const NOISE_GLSL = `
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 x){
  vec2 i = floor(x); vec2 f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float fbm(vec2 p){
  float v = 0.0; float a = 0.5;
  for (int i = 0; i < 5; i++) { v += a * noise(p); p *= 2.1; a *= 0.5; }
  return v;
}
`;

// The sheet's displacement field, evaluated at a local (s, v, u) where
// s = length along the curtain (world units), v = mesh-local altitude
// (world units, -halfH..halfH), u = normalized altitude (0 bottom, 1 top).
// Returns the three components (fold, surf, ripple); the actual sheet offset
// is (fold-0.5)*FOLD_AMP + surf*SURF_AMP + ripple*RIPPLE_AMP. Both shaders
// evaluate the exact same function so the fragment march integrates through
// the same geometry the vertex shader draws.
const CURTAIN_FIELD = `
uniform float uTime;
uniform float uPulse;
uniform vec3 uPulseCenter;
uniform vec3 uOrigin;
uniform vec3 uAxisX;
uniform vec3 uAxisY;
uniform float uHalfH;

vec3 sheetComps(float s, float v, float u){
  // Multi-octave fold drapes, sheared off-vertical by the magnetic dip so the
  // curtain leans like a real field-aligned auroral arc.
  float fold = fbm(vec2((s - v * LEAN) * FOLD_FREQ + uTime * 0.12, u * 1.2 + uTime * 0.05));
  // Travelling undulation: a Gaussian-windowed sine packet gliding along the
  // curtain — the "waves of light running along the aurora" motion.
  float w = (s * 0.13 - uTime * 1.1) * 0.16;
  float pack = exp(-w * w);
  float surf = sin(s * 0.42 - uTime * 2.8) * pack;
  // Click-ripple bulge: a radial displacement wave through the sheet.
  vec2 pc = vec2(dot(uPulseCenter - uOrigin, uAxisX),
                 dot(uPulseCenter - uOrigin, uAxisY) / (2.0 * uHalfH) + 0.5);
  vec2 dpc = vec2(s - pc.x, (u - pc.y) * 2.0 * uHalfH);
  float ripple = exp(-dot(dpc, dpc) * 0.02) * uPulse;
  return vec3(fold, surf, ripple);
}
float sheetZ(vec3 comps){
  return (comps.x - 0.5) * FOLD_AMP + comps.y * SURF_AMP + comps.z * RIPPLE_AMP;
}
`;

// ── Vertex shader = real curtain physics ──────────────────────────────────
// Displace every vertex along the sheet's own z, compute the perturbed normal
// via finite differences of the same field (the deep-space disk's pattern),
// and hand the fragment its world position, world normal, and local (s, u).
const CURTAIN_VERT = `
#define FOLD_AMP ${FOLD_AMP.toFixed(2)}
#define FOLD_FREQ ${FOLD_FREQ.toFixed(4)}
#define SURF_AMP ${SURF_AMP.toFixed(2)}
#define RIPPLE_AMP ${RIPPLE_AMP.toFixed(2)}
#define LEAN ${LEAN.toFixed(3)}
${NOISE_GLSL}
${CURTAIN_FIELD}
varying vec3 vWorld;
varying vec3 vNormal;
varying float vS;
varying float vU;
void main() {
  float s = position.x;
  float v = position.y;
  float u = uv.y;
  float z = sheetZ(sheetComps(s, v, u));
  // Finite-difference normal of the same field (out-of-plane z only).
  float eps = 0.4;
  float zx = sheetZ(sheetComps(s + eps, v, u));
  float zy = sheetZ(sheetComps(s, v + eps, u));
  vec3 nLocal = normalize(vec3(-(zx - z) / eps, -(zy - z) / eps, 1.0));
  vec3 worldPos = (modelMatrix * vec4(position.x, position.y, z, 1.0)).xyz;
  vWorld = worldPos;
  vNormal = normalize(mat3(modelMatrix) * nLocal);
  vS = s;
  vU = u;
  gl_Position = projectionMatrix * viewMatrix * vec4(worldPos, 1.0);
}
`;

// ── Fragment shader = ray trace the emitting sheet ────────────────────────
// A bounded, jittered line-of-sight march through the thin curtain volume.
// The march window scales with view angle: edge-on rays integrate a long path
// through the sheet (real edge-on brightening); face-on rays cross it quickly.
// Each sample re-evaluates the sheet's displaced z so the folds genuinely
// shadow each other along the integration path.
const CURTAIN_FRAG = `
#define FOLD_AMP ${FOLD_AMP.toFixed(2)}
#define FOLD_FREQ ${FOLD_FREQ.toFixed(4)}
#define SURF_AMP ${SURF_AMP.toFixed(2)}
#define RIPPLE_AMP ${RIPPLE_AMP.toFixed(2)}
#define LEAN ${LEAN.toFixed(3)}
#define THICKNESS ${THICKNESS.toFixed(2)}
#define MARCH_STEPS ${MARCH_STEPS}
#define CURTAIN_EMIT ${CURTAIN_EMIT.toFixed(2)}
#define RAY_FREQ ${RAY_FREQ.toFixed(3)}
#define RAY_TILT ${RAY_TILT.toFixed(2)}
#define PULSE_COLOR ${PULSE_COLOR.toFixed(2)}
#define FLASH_AMT ${FLASH_AMT.toFixed(2)}
${NOISE_GLSL}
${CURTAIN_FIELD}
uniform vec3 uColorA;
uniform vec3 uColorB;
uniform vec3 uColorC;
uniform vec3 uAccent;
uniform float uFlash;
uniform vec3 uAxisZ;
varying vec3 vWorld;
varying vec3 vNormal;
varying float vS;
varying float vU;

// Real-aurora colour ladder: bright green base -> teal -> cyan -> violet,
// with a rose/pink nitrogen fringe at the top of tall curtains.
vec3 auroraColor(float band) {
  vec3 col = uAccent;                                    // green base
  col = mix(col, uColorA, smoothstep(0.04, 0.42, band)); // teal
  col = mix(col, uColorB, smoothstep(0.35, 0.62, band)); // cyan
  col = mix(col, uColorC, smoothstep(0.6, 0.85, band));  // violet
  vec3 rose = mix(uColorC, vec3(0.9, 0.42, 0.62), 0.85);
  col = mix(col, rose, smoothstep(0.82, 1.0, band));     // pink fringe
  return col;
}

void main() {
  vec3 rd = normalize(vWorld - cameraPosition);
  // March window scales with view angle: edge-on (dot -> 0) integrates a long
  // path through the sheet; face-on crosses it quickly.
  float cosA = max(abs(dot(rd, normalize(vNormal))), 0.08);
  float halfW = clamp(THICKNESS * 4.0 / cosA, 2.0, 40.0);

  // The fragment's own sheet reference: local (s, u) and displaced z.
  float v0 = (vU - 0.5) * 2.0 * uHalfH;
  vec3 comps0 = sheetComps(vS, v0, vU);
  float refZ = sheetZ(comps0);

  float dl = 2.0 * halfW / float(MARCH_STEPS);
  float jit = hash(gl_FragCoord.xy); // jitter kills banding along the march
  vec3 base = vWorld - rd * halfW;
  vec3 acc = vec3(0.0);

  for (int i = 0; i < MARCH_STEPS; i++) {
    float t = (float(i) + jit) * dl;
    vec3 P = base + rd * t;
    vec3 dP = P - uOrigin;
    float sL = dot(dP, uAxisX);
    float uL = dot(dP, uAxisY) / (2.0 * uHalfH) + 0.5;
    float zL = dot(dP, uAxisZ);
    if (uL < -0.05 || uL > 1.05) continue; // outside the curtain altitude
    float vL = (uL - 0.5) * 2.0 * uHalfH;

    // Sheet density at this depth — measured against the sheet's own z at the
    // sample's (s, u), so folds modulate the integration (self-shadowing).
    vec3 comps = sheetComps(sL, vL, uL);
    float zSheet = sheetZ(comps);
    float q = (zL - zSheet) / THICKNESS;
    float d = exp(-q * q);
    if (d < 0.02) continue;

    // Vertical envelope: sharp lower border, fuzzy upper.
    float env = smoothstep(0.0, 0.05, uL) * (1.0 - smoothstep(0.86, 1.0, uL));
    if (env <= 0.0) continue;

    // Vertical field-aligned rays + fold brightness.
    float rays = noise(vec2(sL * RAY_FREQ + uL * RAY_TILT - uTime * 0.06, uL * 3.0 + uTime * 0.05));
    rays *= rays;
    float rayEnv = 0.4 + 0.6 * comps.x;
    float bright = (0.4 + 1.1 * comps.x) * (0.45 + 1.1 * rays * rayEnv);

    float band = clamp(uL * 0.95 + (comps.x - 0.5) * 0.35 + 0.05, 0.0, 1.0);
    acc += auroraColor(band) * bright * env * d * dl * CURTAIN_EMIT;
  }

  // Click ripple + flash (kept outside the march, scaled by the fragment's
  // own vertical envelope so it respects the curtain shape).
  vec2 dw = vWorld.xz - uPulseCenter.xz;
  float wave = exp(-dot(dw, dw) * 0.004);
  float env0 = smoothstep(0.0, 0.05, vU) * (1.0 - smoothstep(0.86, 1.0, vU));
  acc += uColorB * uPulse * wave * env0 * PULSE_COLOR;
  acc += uAccent * uFlash * FLASH_AMT * env0;

  // Additive output — the RGB *is* the radiance (alpha 1 adds it exactly).
  gl_FragColor = vec4(acc, 1.0);
}
`;

// Tapered streak texture for shooting stars.
function makeStreakTexture(): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = 256;
  c.height = 16;
  const ctx = c.getContext("2d")!;
  const g = ctx.createLinearGradient(0, 0, 256, 0);
  g.addColorStop(0, "rgba(255,255,255,0)");
  g.addColorStop(0.55, "rgba(255,255,255,0.85)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 256, 16);
  return c;
}

interface Streak {
  sprite: THREE.Sprite;
  life: number;
  maxLife: number;
  start: THREE.Vector3;
  end: THREE.Vector3;
}

export function AuroraWebGL({ className = "" }: { className?: string }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [webglOk] = useState(() => supportsWebGL());
  const [broken, setBroken] = useState(false);

  useEffect(() => {
    const host = hostRef.current;
    if (!host || !webglOk) return;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        powerPreference: "high-performance",
      });
    } catch {
      setBroken(true);
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x030610, 1);
    host.appendChild(renderer.domElement);
    renderer.domElement.style.position = "fixed";
    renderer.domElement.style.inset = "0";
    renderer.domElement.style.display = "block";

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      70,
      window.innerWidth / window.innerHeight,
      0.1,
      400,
    );
    const clock = new THREE.Clock();
    const disposables: THREE.Texture[] = [];
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    // ── Palette → live uniforms ───────────────────────────────────────────
    const colA = { value: new THREE.Color(0x2dd4bf) }; // primary (teal)
    const colB = { value: new THREE.Color(0x22d3ee) }; // secondary (cyan)
    const colC = { value: new THREE.Color(0x7c3aed) }; // tertiary (violet)
    const colD = { value: new THREE.Color(0x34d399) }; // accent (green)
    const applyColors = (p: ReturnType<typeof readPaletteColors>) => {
      colA.value.copy(p.primary);
      colB.value.copy(p.secondary);
      colC.value.copy(p.tertiary);
      colD.value.copy(p.accent);
    };
    applyColors(readPaletteColors());
    const poll = makePalettePoller(400, applyColors);

    // ── Aurora curtains ───────────────────────────────────────────────────
    let pulse = 0;
    const pulseCenter = new THREE.Vector3(0, 8, -20);
    // Shared uniform values (same objects across every curtain material, so
    // updating one updates all).
    const sharedU = {
      uTime: { value: 0 },
      uColorA: colA,
      uColorB: colB,
      uColorC: colC,
      uAccent: colD,
      uPulse: { value: 0 },
      uPulseCenter: { value: pulseCenter },
      uFlash: { value: 0 },
    };
    // Four wide high-poly curtains (PlaneGeometry w×h, 192×40 segments ≈ 8k
    // vertices each) — near/mid/far bands spread across the sky for depth.
    // Each has its own material carrying its local frame (origin + basis axes
    // + half-height) so the fragment march can map world points back to the
    // curtain's (s, u, z) frame.
    const curtainCfg = [
      { x: -44, y: 15, z: -52, ry: 0.5, w: 92, h: 34 },
      { x: -8, y: 12, z: -28, ry: 0.06, w: 76, h: 27 },
      { x: 24, y: 16, z: -48, ry: -0.32, w: 88, h: 33 },
      { x: 54, y: 14, z: -38, ry: 0.42, w: 80, h: 30 },
    ];
    const curtainMats: THREE.ShaderMaterial[] = [];
    for (const cfg of curtainCfg) {
      const ry = cfg.ry;
      const mat = new THREE.ShaderMaterial({
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
        uniforms: {
          ...sharedU,
          uOrigin: { value: new THREE.Vector3(cfg.x, cfg.y, cfg.z) },
          uAxisX: { value: new THREE.Vector3(Math.cos(ry), 0, -Math.sin(ry)) },
          uAxisY: { value: new THREE.Vector3(0, 1, 0) },
          uAxisZ: { value: new THREE.Vector3(Math.sin(ry), 0, Math.cos(ry)) },
          uHalfH: { value: cfg.h / 2 },
        },
        vertexShader: CURTAIN_VERT,
        fragmentShader: CURTAIN_FRAG,
      });
      curtainMats.push(mat);
      const m = new THREE.Mesh(new THREE.PlaneGeometry(cfg.w, cfg.h, 192, 40), mat);
      m.position.set(cfg.x, cfg.y, cfg.z);
      m.rotation.y = ry;
      m.renderOrder = 2;
      scene.add(m);
    }

    // ── Frozen lake (planar mirror of the aurora / mountains) ────────────
    // A sheet of ice that reflects the curtains + ranges from a mirrored
    // camera, with fresnel falloff and faint frost-crack veins and ripples.
    const LAKE_RT_SCALE = 0.6;
    const lakeRT = new THREE.WebGLRenderTarget(
      Math.max(2, Math.floor(window.innerWidth * LAKE_RT_SCALE)),
      Math.max(2, Math.floor(window.innerHeight * LAKE_RT_SCALE)),
    );
    disposables.push(lakeRT.texture);
    const lakeGeo = new THREE.PlaneGeometry(320, 90, 1, 1);
    lakeGeo.rotateX(-Math.PI / 2);
    const lakeMat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uCamPos: { value: camera.position },
        uProj: { value: new THREE.Matrix4() },
        uMirrorView: { value: new THREE.Matrix4() },
        uLakeTex: { value: lakeRT.texture },
        uColorA: colA,
      },
      vertexShader: LAKE_VERT,
      fragmentShader: LAKE_FRAG,
      side: THREE.DoubleSide, // needs both sides for a near-y plane
    });
    const lake = new THREE.Mesh(lakeGeo, lakeMat);
    lake.position.set(0, -0.4, 0);
    lake.renderOrder = 2;
    scene.add(lake);

    // Mirror camera: reflects the scene across the ice plane. auto-update is
    // off — we repoint matrixWorld manually every frame (like the neon mirror).
    const mirrorCam = new THREE.PerspectiveCamera(70, camera.aspect, 0.1, 200);
    mirrorCam.matrixWorldAutoUpdate = false;
    const reflMatrix = new THREE.Matrix4().makeScale(1, -1, 1);

    // ── Stars ─────────────────────────────────────────────────────────────
    const STARS = 700;
    const starPos = new Float32Array(STARS * 3);
    for (let i = 0; i < STARS; i++) {
      const x = (Math.random() - 0.5) * 220;
      const y = 2 + Math.random() * 80;
      const z = -20 - Math.random() * 120;
      starPos[i * 3] = x;
      starPos[i * 3 + 1] = y;
      starPos[i * 3 + 2] = z;
    }
    const starGeo = new THREE.BufferGeometry();
    starGeo.setAttribute("position", new THREE.BufferAttribute(starPos, 3));
    const starMat = new THREE.PointsMaterial({
      color: 0xbfe6ff,
      size: 0.3,
      transparent: true,
      opacity: 0.75,
      sizeAttenuation: true,
    });
    const stars = new THREE.Points(starGeo, starMat);
    scene.add(stars);

    // ── Shooting stars ────────────────────────────────────────────────────
    const streakTex = new THREE.CanvasTexture(makeStreakTexture());
    disposables.push(streakTex);
    const streaks: Streak[] = [];
    function spawnStreak() {
      const sx = (Math.random() - 0.5) * 80;
      const sy = 20 + Math.random() * 34;
      const sz = -30 - Math.random() * 40;
      const len = 20 + Math.random() * 24;
      const mat = new THREE.SpriteMaterial({
        map: streakTex,
        color: colD.value,
        transparent: true,
        opacity: 0.9,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      const sprite = new THREE.Sprite(mat);
      sprite.scale.set(len, 1.6, 1);
      const start = new THREE.Vector3(sx, sy, sz);
      const end = new THREE.Vector3(sx + 16, sy - 10, sz + 12);
      sprite.position.copy(start);
      scene.add(sprite);
      streaks.push({ sprite, life: 0, maxLife: 1.1, start, end });
    }

    // ── Click pulse rings + global flash ──────────────────────────────────
    const ringGeo = new THREE.RingGeometry(0.5, 0.66, 64);
    const clickRings: { mesh: THREE.Mesh; life: number; maxLife: number }[] = [];
    let flash = 0;
    function spawnRing(at: THREE.Vector3, strength: number) {
      const m = new THREE.Mesh(
        ringGeo,
        new THREE.MeshBasicMaterial({
          color: 0xd6efff,
          transparent: true,
          opacity: 0.85 * strength,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
          side: THREE.DoubleSide,
        }),
      );
      m.rotation.x = -Math.PI / 2;
      m.position.copy(at);
      m.renderOrder = 3;
      scene.add(m);
      clickRings.push({ mesh: m, life: 0, maxLife: 1.2 });
    }

    // ── Interaction state ─────────────────────────────────────────────────
    const orbit = {
      azimuth: 0,
      polar: 1.34,
      radius: 24,
      target: new THREE.Vector3(0, 7, -8),
    };
    const mouse = { x: 0, y: 0 };
    let dragging = false;
    let lastPX = 0;
    let lastPY = 0;
    let pressStart = 0;
    let downX = 0;
    let downY = 0;

    const onPointerDown = (e: PointerEvent) => {
      if (e.button === 2) {
        dragging = true;
        lastPX = e.clientX;
        lastPY = e.clientY;
      } else if (e.button === 0) {
        pressStart = performance.now();
        downX = e.clientX;
        downY = e.clientY;
      }
    };
    const onPointerMove = (e: PointerEvent) => {
      mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouse.y = (e.clientY / window.innerHeight) * 2 - 1;
      if (!dragging) return;
      orbit.azimuth = Math.max(
        -0.9,
        Math.min(0.9, orbit.azimuth - (e.clientX - lastPX) * 0.004),
      );
      orbit.polar = Math.max(
        1.0,
        Math.min(1.55, orbit.polar - (e.clientY - lastPY) * 0.004),
      );
      lastPX = e.clientX;
      lastPY = e.clientY;
    };
    const onPointerUp = (e: PointerEvent) => {
      if (e.button === 2) {
        dragging = false;
        return;
      }
      if (e.button !== 0) return;
      const dx = e.clientX - downX;
      const dy = e.clientY - downY;
      const dur = performance.now() - pressStart;
      if (Math.hypot(dx, dy) > 6 || dur > 400) return;
      if (
        e.target instanceof Element &&
        e.target.closest(
          "a,button,input,textarea,select,[role='button'],[contenteditable]",
        )
      ) {
        return;
      }
      const ndcX = (e.clientX / window.innerWidth) * 2 - 1;
      const ndcY = -(e.clientY / window.innerHeight) * 2 + 1;
      emitCosmosEvent({ type: "aurora-pulse", heat: 0.5, hue: 173, x: ndcX, y: ndcY });
      // World-space ripple centre: ray against the y=8 curtain plane.
      const ndc = new THREE.Vector3(ndcX, ndcY, 0.5).unproject(camera);
      const dir = ndc.sub(camera.position).normalize();
      const tHit = (8 - camera.position.y) / dir.y;
      if (tHit > 0) {
        pulseCenter.copy(camera.position).addScaledVector(dir, tHit);
      }
      pulse = 1;
      flash = 1;
      spawnRing(pulseCenter.clone(), 1);
    };
    const onWheel = (e: WheelEvent) => {
      if (!e.shiftKey) return;
      e.preventDefault();
      orbit.radius = Math.max(10, Math.min(40, orbit.radius * (1 + e.deltaY * 0.001)));
    };
    const onContextMenu = (e: Event) => e.preventDefault();
    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("wheel", onWheel, { passive: false });
    window.addEventListener("contextmenu", onContextMenu);
    window.addEventListener("resize", onResize);

    // ── Ambient ripples ───────────────────────────────────────────────────
    let ambientTimer = 10 + Math.random() * 6;
    function updateAmbient(dt: number) {
      ambientTimer -= dt;
      if (ambientTimer > 0) return;
      ambientTimer = 11 + Math.random() * 7;
      pulse = Math.max(pulse, 0.5);
      flash = Math.max(flash, 0.4);
      pulseCenter.set(
        (Math.random() - 0.5) * 50,
        8,
        -20 - Math.random() * 30,
      );
      spawnRing(pulseCenter.clone(), 0.5);
      emitCosmosEvent({ type: "aurora-pulse", heat: 0.22, hue: 173, x: 0, y: 0 });
    }

    // ── Animate loop ──────────────────────────────────────────────────────
    let raf = 0;
    let sceneTime = 0;
    let streakTimer = 6 + Math.random() * 6;
    function animate() {
      raf = requestAnimationFrame(animate);
      const dt = Math.min(clock.getDelta(), 0.05);
      sceneTime += reduced ? 0 : dt;
      const t = sceneTime;
      poll(performance.now());

      if (!reduced) {
        orbit.azimuth += Math.sin(t * 0.06) * dt * 0.02;
        pulse = Math.max(0, pulse - dt * 0.7);
        flash = Math.max(0, flash - dt * 1.2);
      }

      const pa = orbit.polar;
      const aa = orbit.azimuth;
      const r = orbit.radius;
      camera.position.set(
        orbit.target.x + r * Math.sin(pa) * Math.sin(aa) + mouse.x * 1.1,
        orbit.target.y + r * Math.cos(pa) + -mouse.y * 0.5,
        orbit.target.z + r * Math.sin(pa) * Math.cos(aa),
      );
      camera.lookAt(orbit.target);

      for (const mat of curtainMats) {
        mat.uniforms.uTime.value = t;
        mat.uniforms.uPulse.value = pulse;
        mat.uniforms.uFlash.value = flash;
      }
      starMat.opacity = 0.6 + 0.2 * Math.sin(t * 0.7);

      // Shooting stars.
      if (!reduced) streakTimer -= dt;
      if (streakTimer <= 0) {
        streakTimer = 7 + Math.random() * 8;
        if (!reduced) spawnStreak();
      }
      for (let i = streaks.length - 1; i >= 0; i--) {
        const s = streaks[i];
        s.life += dt;
        const p = Math.min(1, s.life / s.maxLife);
        s.sprite.position.lerpVectors(s.start, s.end, p);
        (s.sprite.material as THREE.SpriteMaterial).opacity = (1 - p) * 0.9;
        if (p >= 1) {
          scene.remove(s.sprite);
          (s.sprite.material as THREE.Material).dispose();
          streaks.splice(i, 1);
        }
      }

      // Click pulse rings expand across the curtain plane.
      for (let i = clickRings.length - 1; i >= 0; i--) {
        const r = clickRings[i];
        r.life += dt;
        const p = Math.min(1, r.life / r.maxLife);
        r.mesh.scale.setScalar(1 + p * 36);
        (r.mesh.material as THREE.MeshBasicMaterial).opacity = (1 - p) * 0.85;
        if (p >= 1) {
          scene.remove(r.mesh);
          (r.mesh.material as THREE.Material).dispose();
          clickRings.splice(i, 1);
        }
      }

      if (!reduced) updateAmbient(dt);

      // Frozen-lake planar reflection: render the scene (curtains, stars,
      // ranges) from a camera mirrored across the ice, then draw the lake
      // sampling that buffer. The lake hides itself during the mirrored pass
      // so it reflects sky, not its own ice.
      lake.visible = false;
      const reflM = reflMatrix.clone().multiply(camera.matrixWorld);
      mirrorCam.matrixWorld.copy(reflM);
      mirrorCam.matrixWorldInverse.copy(reflM).invert();
      mirrorCam.projectionMatrix.copy(camera.projectionMatrix);
      mirrorCam.aspect = camera.aspect;
      renderer.setRenderTarget(lakeRT);
      renderer.clear();
      renderer.render(scene, mirrorCam);
      lake.visible = true;
      lakeMat.uniforms.uTime.value = t;
      lakeMat.uniforms.uProj.value.copy(camera.projectionMatrix);
      lakeMat.uniforms.uMirrorView.value.copy(mirrorCam.matrixWorldInverse);

      renderer.setRenderTarget(null);
      renderer.render(scene, camera);
    }
    animate();

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("wheel", onWheel);
      window.removeEventListener("contextmenu", onContextMenu);
      window.removeEventListener("resize", onResize);
      streaks.forEach((s) => {
        (s.sprite.material as THREE.Material).dispose();
      });
      clickRings.forEach((r) => {
        (r.mesh.material as THREE.Material).dispose();
      });
      ringGeo.dispose();
      scene.traverse((obj) => {
        const mesh = obj as THREE.Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
        const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else if (mat) mat.dispose();
      });
      disposables.forEach((t2) => t2.dispose());
      renderer.forceContextLoss();
      renderer.dispose();
      if (renderer.domElement.parentNode === host) {
        host.removeChild(renderer.domElement);
      }
    };
  }, [webglOk]);

  if (!webglOk || broken) {
    return (
      <div
        aria-hidden
        className={`fixed inset-0 -z-10 ${className}`}
        style={{
          background:
            "radial-gradient(ellipse at 50% 0%, hsl(var(--palette-primary) / 0.32), hsl(var(--palette-tertiary) / 0.1) 55%, transparent 75%)",
        }}
      />
    );
  }

  return (
    <div
      ref={hostRef}
      aria-hidden
      className={`fixed inset-0 -z-10 pointer-events-none ${className}`}
    />
  );
}
