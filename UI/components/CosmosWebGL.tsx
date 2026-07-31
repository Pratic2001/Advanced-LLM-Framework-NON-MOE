"use client";

/**
 * CosmosWebGL — cinematic DeepSpace background for the LLMForge UI.
 *
 * Real WebGL/3D renderer built on React Three Fiber. Deliberately does NOT
 * use the postprocessing EffectComposer: its final screen pass writes alpha-0
 * into a transparent drawing buffer, which on many GPUs makes the whole scene
 * invisible (only the canvas' CSS background shows). Instead everything is
 * drawn directly to the canvas with alpha 1, so the scene is opaque and visible
 * on any GPU that supports WebGL at all.
 *
 * The scene is a small explicit multi-pass pipeline, owned in SceneContent's
 * useFrame. The R3F scene only holds the final composite quad + the additive
 * hypernova quad; the black hole's real gravitational lensing happens between
 * two off-screen render targets:
 *
 *   1. Background pass → RT_bg (≈0.8× res): the cool deep-navy nebula, spectral
 *      starfield and spiral galaxies, rendered with the orbiting camera.
 *   2. Black-hole pass → RT_bh (≈0.5× res): a full-screen quad that, per pixel,
 *      integrates the exact Schwarzschild null-geodesic equation
 *
 *        ẍ = −(3/2)·Rs·(r²·|ẋ|² − (x·ẋ)²)·x / r⁵
 *
 *      (equivalent to the Binet equation d²u/dφ² + u = (3/2)·Rs·u²). Escaped
 *      rays sample RT_bg — so background stars are genuinely lensed into arcs
 *      and the supernova's light wave bends around the orb — disk hits get real
 *      physics shading (Keplerian T ∝ r^−3/4, Doppler beaming, gravitational
 *      redshift), and captured rays are pure black: the event-horizon shadow.
 *      The photon ring, far-side disk arcs and Einstein ring all emerge from
 *      the integration rather than being faked.
 *   3. R3F composite: a full-screen quad displaying RT_bh (with a gentle
 *      vignette) plus the hypernova quad drawn additively on top at full res.
 *
 * Because the supernova's illumination wavefront lives in the nebula (pass 1),
 * it automatically passes through the lensing pass — the light-speed brightening
 * wave visibly bends around the black hole.
 *
 *   · A black hole with a real event-horizon shadow, a Keplerian accretion disk
 *     (white-hot inner rim → golden mid → deep-orange outer, with a bright,
 *     blue-shifted approaching side), a thin photon ring and a warm halo.
 *   · A periodic supernova in a far corner: on flash it floods the frame with a
 *     light-speed illumination wave + radial god-ray shafts, then a Sedov–Taylor
 *     blast wave (R ∝ t^0.42) expands while its colour cools.
 *   · Mouse drives the camera orbit (the whole scene parallaxes, the disk
 *     inclination tilts) and the hypernova sway. `prefers-reduced-motion` damps
 *     every movement. If WebGL is unavailable (or renders nothing), an animated
 *     CSS starfield fallback keeps the palette from ever reading as a flat colour.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

/* ─────────────────────────── Shared GLSL ─────────────────────────────── */

const GLSL_COMMON = /* glsl */ `
float hash21(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = hash21(i);
  float b = hash21(i + vec2(1.0, 0.0));
  float c = hash21(i + vec2(0.0, 1.0));
  float d = hash21(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbm(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  mat2 R = mat2(0.8, 0.6, -0.6, 0.8);
  for (int i = 0; i < 5; i++) { v += a * vnoise(p); p = R * p * 2.03; a *= 0.5; }
  return v;
}
// ACES filmic tone-map (Narkowicz approximation) + sRGB-ish gamma.
vec3 acesGamma(vec3 x) {
  vec3 t = clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
  return pow(t, vec3(0.4545));
}
`;

// Vertex shader for the screen-aligned quads (fill the viewport).
const SCREEN_QUAD_VS = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

/* ─────────────────────────── Nebula shaders ─────────────────────────── */

const NEBULA_VS = /* glsl */ `
varying vec3 vDir;
void main() {
  vDir = normalize(position);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const NEBULA_FS = /* glsl */ `
uniform float uTime;
uniform vec2 uMouse;
uniform vec3 uFlashDir;  // sky direction of the supernova flash
uniform vec3 uBhDir;     // sky direction of the black hole (camera→BH line of sight)
uniform float uFlashR;   // light travelled since the flash (scene units, grows at c)
uniform float uFlashE;   // residual flash energy (decays over ~2s)
varying vec3 vDir;

${GLSL_COMMON}

void main() {
  vec3 dir = normalize(vDir);

  // Mouse parallax: gently re-sample the cloud as the pointer moves.
  vec2 mo = uMouse - 0.5;
  dir = normalize(dir + vec3(-mo.x * 0.5, -mo.y * 0.35, 0.0) * 0.45);

  float t = uTime * 0.03;
  vec2 p = vec2(atan(dir.z, dir.x), asin(clamp(dir.y, -1.0, 1.0)));
  p = p * vec2(0.8, 0.55) + vec2(t * 0.7, t * 0.4);

  float n1 = fbm(p * 1.7 + 3.0);
  float n2 = fbm(p * 2.4 + 11.0);
  float n3 = fbm(p * 1.1 + 23.0);

  // Dust lanes carve the cloud into filaments.
  float dust = smoothstep(0.26, 0.74, n3);
  float lanes = 1.0 - smoothstep(0.30, 0.66, fbm(p * 2.1 + 7.0)) * 0.7;

  // COOL palette: deep-navy base, teal O-III, violet dust, faint warm H-alpha
  // accents — so the golden disk + supernova stand out as the heroes.
  vec3 ha = vec3(1.0, 0.28, 0.18) * pow(n1, 2.0) * 0.60;    // faint warm H-alpha
  vec3 o3 = vec3(0.28, 0.56, 0.95) * pow(n2, 1.8) * 1.05;   // O-III teal
  vec3 vio = vec3(0.55, 0.32, 0.82) * pow(n3, 1.4) * 0.60;  // violet dust
  vec3 col = (ha + o3 + vio) * dust * lanes * 1.9;
  // Space ambient bright enough that the sky is a deep navy-blue, never a
  // black void, even in the calmest phase of the hypernova cycle.
  col += vec3(0.030, 0.046, 0.082);

  // Supernova light-echo wavefront: a warm illumination shell expanding outward
  // from the flash at the speed of light. Real supernova light echoes work
  // exactly this way — each cloud element brightens as the light reaches it.
  float nebR = 420.0;
  float ang = acos(clamp(dot(dir, normalize(uFlashDir)), -1.0, 1.0));
  float dist = ang * nebR;                       // great-circle distance on the sky
  float wave = exp(-pow((dist - uFlashR) / 70.0, 2.0));  // bright ring at the front
  // Gentle fill behind the front (guarded so a zero-radius flash can't produce
  // an undefined smoothstep edge).
  float rA = max(uFlashR * 0.05, 0.001);
  float rB = max(uFlashR, 0.001);
  float lit = smoothstep(rA, rB, dist);
  col += vec3(1.0, 0.72, 0.40) * uFlashE * (wave * 5.2 + lit * 0.65);

  // Warm golden halo seated on the black hole's line of sight. It lives in the
  // background pass so the geodesic lensing below bends it into the bright
  // Einstein halo wrapping the event-horizon shadow — the "fiery golden light
  // bending around the orb". When the supernova's light wave reaches the BH
  // region it flares up (× uFlashE), so the propagating light visibly passes
  // the orb.
  float bhAng = acos(clamp(dot(dir, normalize(uBhDir)), -1.0, 1.0));
  float bhGlow = exp(-pow(bhAng / 0.16, 2.0));
  col += vec3(1.0, 0.66, 0.36) * bhGlow * (0.55 + uFlashE * 1.15);

  gl_FragColor = vec4(acesGamma(col), 1.0);
}
`;

/* ────────────────────────── Starfield shaders ───────────────────────── */

const STAR_VS = /* glsl */ `
attribute float aSize;
attribute vec3 aColor;
attribute float aPhase;
attribute float aTw;
varying vec3 vColor;
varying float vPhase;
varying float vTw;
void main() {
  vColor = aColor;
  vPhase = aPhase;
  vTw = aTw;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = aSize * (400.0 / max(-mv.z, 1.0));
  gl_Position = projectionMatrix * mv;
}
`;

const STAR_FS = /* glsl */ `
uniform float uTime;
varying vec3 vColor;
varying float vPhase;
varying float vTw;
void main() {
  float d = length(gl_PointCoord - 0.5);
  float a = smoothstep(0.5, 0.02, d);
  float tw = 0.55 + 0.45 * sin(uTime * vTw + vPhase);
  float b = a * tw * 1.7;
  gl_FragColor = vec4(vColor * b, b);
}
`;

/* ─────────────────────────── Composite shader ───────────────────────── */

// The final screen quad: displays the lensed RT_bh with a gentle vignette and
// a whisper of film grain so the frame reads like a graded astrophoto.
const COMPOSITE_FS = /* glsl */ `
uniform sampler2D uTex;
uniform float uTime;
varying vec2 vUv;

${GLSL_COMMON}

void main() {
  vec3 col = texture2D(uTex, vUv).rgb;
  vec2 q = vUv * 2.0 - 1.0;
  float vig = 1.0 - dot(q, q) * 0.34;
  col *= 0.84 + 0.16 * vig;
  float grain = hash21(vUv * 743.0 + uTime * 0.4) - 0.5;
  col += grain * 0.016;
  gl_FragColor = vec4(col, 1.0);
}
`;

/* ──────────────── Real geodesic black-hole shader ───────────────────── */

// Per-pixel Schwarzschild null-geodesic ray tracing. Every frame each pixel
// casts one photon, integrates ẍ = −(3/2)·Rs·(r²|ẋ|² − (x·ẋ)²)·x/r⁵ with an
// adaptive step (fine near the horizon so near-critical rays resolve the photon
// ring) and terminates on: horizon crossing (→ black shadow), disk-plane
// crossing inside [ISCO, r_out] (→ physics disk shading), or escape (→ sample
// the lensed background RT). The BH sits at the origin; the disk is the y=0
// plane. Rs is one scene unit.
const GEODESIC_BH_FS = /* glsl */ `
uniform float uTime;
uniform vec2 uMouse;
uniform vec2 uResolution;    // full drawing-buffer size (for aspect / px scale)
uniform vec3 uCamPos;        // orbiting camera position (world)
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uCamFwd;        // camera basis vectors (world)
uniform mat4 uViewProj;      // bgCamera projection × view inverse (world→clip)
uniform vec2 uBhOffset;      // screen position (NDC) of the BH shadow centre
uniform float uCamDist;      // camera distance from the BH (in Rs)
uniform float uFov;          // camera vertical fov (radians)
uniform sampler2D uBgTex;    // lensed background (nebula + stars + galaxies)
uniform float uDiscInner;    // ISCO (Rs)
uniform float uDiscOuter;    // disk outer edge (Rs)
uniform float uDirectB;      // above this impact parameter → flat-space fast path
uniform float uMaxSteps;     // geodesic step budget
uniform float uStepBase;     // base affine step
varying vec2 vUv;

${GLSL_COMMON}

float hash11(float p) {
  p = fract(p * 0.1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}

// Tanner Helland blackbody colour approximation (T in Kelvin → RGB 0..1).
vec3 blackbody(float t) {
  t = t / 100.0;
  vec3 c;
  c.r = t < 66.0 ? 255.0 : 329.698727446 * pow(max(t - 60.0, 0.001), -0.1332047592);
  c.g = t < 66.0
    ? (99.4708025861 * log(max(t, 0.001)) - 161.1195681661)
    : 288.1221695283 * pow(max(t - 60.0, 0.001), -0.0755148492);
  c.b = t >= 66.0 ? 255.0 : (t <= 19.0 ? 0.0 : 138.5177312231 * log(max(t - 10.0, 0.001)) - 305.0447927307);
  return clamp(c / 255.0, 0.0, 1.0);
}

// Exact Schwarzschild null-geodesic acceleration (Rs = 1).
vec3 gAcc(vec3 p, vec3 v) {
  float r2 = dot(p, p);
  float r5 = r2 * r2 * sqrt(r2);
  // |p×v|² = r²v² − (p·v)²; always ≥ 0, clamped against fp error.
  float L2 = max(r2 * dot(v, v) - dot(p, v) * dot(p, v), 0.0);
  return -1.5 * L2 * p / r5;
}

// Keplerian accretion-disk shading at a hit point P (world), photon dir kDir.
vec3 shadeDisk(vec3 P, vec3 kDir) {
  float r = length(P.xz);
  float M = 0.5;                                     // Rs = 2M = 1
  vec3 tangent = normalize(vec3(-P.z, 0.0, P.x));    // CCW orbital direction
  // Local Keplerian speed and Lorentz factor (β = √(3M/r), RS units).
  float beta = sqrt(3.0 * M / max(r, 0.05));
  float gamma = 1.0 / sqrt(max(1.0 - beta * beta, 1e-4));
  float kt = dot(tangent, kDir);                     // photon along the flow?
  // Doppler factor: >1 approaching (bright, blue), <1 receding (dim, red).
  float doppler = 1.0 / (gamma * max(1.0 - beta * kt, 1e-4));
  // Gravitational redshift + time dilation at the emission radius.
  float gred = sqrt(max(1.0 - 1.0 / max(r, 0.05), 0.0));
  // Keplerian temperature profile T ∝ r^−3/4, Doppler-shifted. Tin is set just
  // below white-hot so the disk reads warm-golden: white-gold inner rim →
  // golden mid → deep orange outer edge.
  float Tin = 7000.0;
  float T = Tin * pow(uDiscInner / max(r, 1e-3), 0.75) * max(doppler, 0.12);
  vec3 base = blackbody(T);
  // Thermal intensity ∝ T³ + Doppler beaming ∝ δ³ (bolometric δ⁴, projected).
  float I = pow(T / Tin, 2.5) * pow(max(doppler, 1e-3), 3.0) * 3.0;
  vec3 col = base * I * (gred * gred);
  // Turbulent magnetic filaments wound around the disk.
  float nf = fbm(P.xz * 0.9 + uTime * 0.05);
  col *= 0.72 + 0.56 * nf;
  col *= smoothstep(uDiscInner, uDiscInner * 1.25, r);  // fade ISCO edge
  col *= smoothstep(uDiscOuter, uDiscOuter * 0.82, r);  // fade outer edge
  return acesGamma(max(col, 0.0));
}

// Sample the lensed background along a world direction.
vec3 sampleBg(vec3 dir) {
  vec4 clip = uViewProj * vec4(dir, 0.0);
  vec2 uv = clip.xy / max(clip.w, 1e-6) * 0.5 + 0.5;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) return vec3(0.008, 0.014, 0.030);
  return texture2D(uBgTex, uv).rgb;
}

void main() {
  float aspect = uResolution.x / uResolution.y;
  vec2 ndc = vUv * 2.0 - 1.0;
  vec2 off = ndc - uBhOffset;                        // NDC offset from BH centre
  float tanA = tan(uFov * 0.5);
  vec3 rd = normalize(uCamFwd + uCamRight * off.x * tanA * aspect + uCamUp * off.y * tanA);
  vec3 ro = uCamPos;
  float b0 = length(cross(ro, rd));                  // impact parameter

  vec3 col;
  if (b0 > uDirectB) {
    // Fast path — negligible bending at this distance from the BH. Do a plain
    // disk-plane intersection, else sample the background at the naive ray.
    if (abs(rd.y) > 1e-5) {
      float tp = -ro.y / rd.y;
      if (tp > 0.0) {
        vec3 P = ro + rd * tp;
        float rr = length(P.xz);
        if (rr > uDiscInner && rr < uDiscOuter) col = shadeDisk(P, rd);
        else col = sampleBg(rd);
      } else col = sampleBg(rd);
    } else col = sampleBg(rd);
  } else {
    // Full geodesic integration (leapfrog / velocity-Verlet, adaptive step).
    vec3 pos = ro;
    vec3 vel = rd;
    vec3 prevPos = ro;
    float prevY = ro.y;
    bool captured = false;
    bool hitDisk = false;
    vec3 hitP = vec3(0.0);
    vec3 hitK = vec3(0.0);
    int maxi = int(uMaxSteps);
    for (int i = 0; i < 240; i++) {
      if (i >= maxi) break;
      float r = length(pos);
      // Step shrinks as the ray nears the horizon so the photon ring resolves.
      float dt = clamp(uStepBase * max(r - 1.0, 0.06) / 2.5, 0.004, 0.6);
      vec3 a1 = gAcc(pos, vel);
      vec3 vh = vel + a1 * (dt * 0.5);
      vec3 np = pos + vh * dt;
      vec3 a2 = gAcc(np, vh);
      vec3 nv = vh + a2 * (dt * 0.5);
      pos = np;
      vel = nv;
      r = length(pos);
      if (r < 1.03) { captured = true; break; }      // fell past the horizon
      if (r > 24.0) break;                           // escaped the lens
      if (prevY * pos.y <= 0.0) {                    // crossed the disk plane
        float tt = prevY / max(prevY - pos.y, 1e-6);
        vec3 hp = mix(prevPos, pos, clamp(tt, 0.0, 1.0));
        float rr = length(hp.xz);
        if (rr > uDiscInner && rr < uDiscOuter) { hitDisk = true; hitP = hp; hitK = normalize(nv); break; }
      }
      prevPos = pos;
      prevY = pos.y;
    }
    if (captured) col = vec3(0.0);                   // the event-horizon shadow
    else if (hitDisk) col = shadeDisk(hitP, hitK);
    else col = sampleBg(normalize(vel));             // lensed background
  }

  // Photon ring + warm halo, in screen space, tied to the shadow edge. The
  // geodesics already put real (dim) light here; this keeps the ring crisp at
  // half resolution. Shadow edge sits at impact parameter b_c = (3√3/2)·Rs.
  float shadowAng = atan(2.598 / max(uCamDist, 0.001));  // rad off-axis
  float angOff = atan(tanA * length(vec2(off.x * aspect, off.y)));
  float pxAng = (2.0 * tanA) / max(uResolution.y, 1.0);
  float ringW = 1.7 * pxAng;
  float ringD = abs(angOff - shadowAng);
  float ring = exp(-(ringD * ringD) / (ringW * ringW));
  // The photon ring is hotter on the side whose disk gas co-rotates toward the
  // camera (relativistic beaming) — a subtle brightness asymmetry around the rim.
  float ringAng = atan(off.y * aspect, off.x);
  float ringSide = 0.72 + 0.28 * cos(ringAng);
  col += vec3(1.0, 0.84, 0.52) * ring * 3.0 * ringSide;
  float haloD = max(angOff - shadowAng, 0.0);
  col += vec3(1.0, 0.60, 0.30) * exp(-(haloD * haloD) / (shadowAng * shadowAng * 0.30)) * 0.85;

  gl_FragColor = vec4(col, 1.0);
}
`;

/* ─────────────────────────── Hypernova shader ───────────────────────── */

// Full supernova lifecycle on a screen-aligned additive quad. The blast wave
// grows like a real Sedov–Taylor shock (R ∝ t^0.42) and its colour temperature
// cools as it expands: white → gold → orange → red, exactly as a real
// shockfront does. The flash phase also throws radial god-ray shafts and floods
// the frame with light; a matching wavefront is sent into the nebula (via
// uniforms driven from JS) so the illumination is seen travelling through space
// and being lensed by the black hole.
const HYPERNOVA_FS = /* glsl */ `
uniform float uTime;
uniform vec2 uPos;       // blast centre, UV 0..1
uniform vec2 uMouse;     // pointer, UV 0..1
uniform float uMaxR;     // max shock radius as fraction of min(resolution)
uniform vec2 uResolution;
varying vec2 vUv;

${GLSL_COMMON}

float hash11(float p) {
  p = fract(p * 0.1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}

// Cooling sequence of a real blast wave: the shock starts white-hot and cools
// through gold, orange and red as it expands. Stops are pushed warm so the
// whole blast stays "fiery golden" for as long as possible.
vec3 cooling(float x) {
  x = clamp(x, 0.0, 1.0);
  vec3 c0 = vec3(1.00, 0.95, 0.90);
  vec3 c1 = vec3(1.00, 0.92, 0.76);
  vec3 c2 = vec3(1.00, 0.70, 0.34);
  vec3 c3 = vec3(0.98, 0.44, 0.20);
  vec3 c4 = vec3(0.76, 0.22, 0.16);
  vec3 c5 = vec3(0.52, 0.14, 0.10);
  float x5 = x * 5.0;
  if (x5 < 1.0) return mix(c0, c1, x5);
  else if (x5 < 2.0) return mix(c1, c2, x5 - 1.0);
  else if (x5 < 3.0) return mix(c2, c3, x5 - 2.0);
  else if (x5 < 4.0) return mix(c3, c4, x5 - 3.0);
  else return mix(c4, c5, x5 - 4.0);
}

void main() {
  // Hypernova lifecycle (seconds), driven by uTime.
  float PROG    = 7.0;   // blue supergiant, brightening
  float COLL    = 1.2;   // core collapse
  float FLASH   = 0.7;   // shock breakout flash
  float EJECTA  = 13.0;  // blast wave expands + cools
  float REMNANT = 7.5;   // remnant nebula + pulsar wind
  float QUIESC  = 2.5;   // calm
  float CYCLE   = PROG + COLL + FLASH + EJECTA + REMNANT + QUIESC;

  float age = mod(uTime, CYCLE);
  float minDim = min(uResolution.x, uResolution.y);
  float maxR = uMaxR * minDim;

  vec2 px = vUv * uResolution;
  vec2 pc = uPos * uResolution;
  vec2 d = px - pc;
  float r = length(d);
  float ang = atan(d.y, d.x);

  // Cheap early-out for the huge empty region of the frame.
  if (r > maxR * 1.08) {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  vec3 add = vec3(0.0);
  float mo = uMouse.x + uMouse.y;

  if (age < PROG) {
    // Blue supergiant breathing in the deep field — bright enough that the
    // scene is visibly alive from the very first frame, then it builds.
    float p = age / PROG;
    float bright = 0.55 + 0.45 * p;
    float pulse = 0.82 + 0.18 * sin(uTime * 3.2);
    float haloR = 40.0 + 18.0 * p;
    add += vec3(0.70, 0.82, 1.0) * exp(-(r * r) / (haloR * haloR)) * (1.8 * bright * pulse);
    add += vec3(1.0, 0.96, 0.90) * exp(-(r * r) / 2.6) * bright * pulse * 1.25;
    // Faint diffraction spikes.
    float spk = exp(-abs(d.x) * 0.05) * exp(-abs(d.y) * 0.05);
    add += vec3(0.65, 0.80, 1.0) * spk * 0.15 * bright * pulse;
  } else if (age < PROG + COLL) {
    // The core collapses and the star dims, shrinking.
    float p = (age - PROG) / COLL;
    float shrink = 1.0 - 0.82 * p;
    float fade = 1.0 - p * p;
    float hr = 18.0 * shrink + 3.0;
    add += vec3(0.95, 0.60, 0.42) * exp(-(r * r) / (hr * hr)) * fade;
    add += vec3(1.0, 0.92, 0.85) * exp(-(r * r) / 3.0) * fade;
  } else if (age < PROG + COLL + FLASH) {
    // Shock breakout: an intense, fast burst of bluish-white light that floods
    // the whole frame, plus radial god-ray shafts.
    float p = (age - PROG - COLL) / FLASH;
    float peak = sin(p * 3.14159265);
    float flashR = maxR * (0.12 + 1.15 * p);
    vec3 fc = mix(vec3(1.0, 0.98, 0.92), vec3(1.0, 0.90, 0.72), p);
    add += fc * exp(-(r * r) / (flashR * flashR)) * (3.6 * peak);
    add += vec3(1.0) * exp(-(r * r) / 4.0) * peak;
    // Screen-wide warm glow so the flash reads as a full event.
    add += vec3(1.0, 0.62, 0.34) * exp(-(r * r) / (maxR * maxR * 1.6)) * peak * 0.32;
    // Radial god rays radiating from the blast centre.
    for (int i = 0; i < 28; i++) {
      float fi = float(i);
      float a0 = hash11(fi * 1.7 + 13.0) * 6.2831853;
      float spread = 0.055 + hash11(fi * 3.1 + 5.0) * 0.08;
      float ad = mod(ang - a0 + 3.14159265, 6.2831853) - 3.14159265;
      float shaft = exp(-(ad * ad) / (spread * spread));
      float rr = r / max(1.0, flashR);
      shaft *= exp(-rr * 2.2);
      vec3 sc = mix(vec3(1.0, 0.88, 0.62), vec3(1.0, 0.78, 0.50), fi * 0.028);
      add += sc * shaft * peak * 1.35;
    }
  } else if (age < PROG + COLL + FLASH + EJECTA) {
    // The main event: a Sedov–Taylor blast wave that GROWS while cooling.
    float p = (age - PROG - COLL - FLASH) / EJECTA;
    float R = maxR * (0.08 + 0.92 * pow(p, 0.42));   // R ∝ t^0.42
    float w = 5.0 + p * 30.0;                        // front thickens as it slows
    float front = exp(-((r - R) * (r - R)) / (w * w));
    vec3 cool = cooling(p);                          // colour evolves as it grows
    float bright = pow(0.16 / max(0.16, R / maxR), 1.6) * 1.7;
    add += cool * front * bright;

    // Hot leading rim riding ahead of the cooled ejecta.
    float rimW = w * 0.25;
    float rim = exp(-((r - R * 1.02) * (r - R * 1.02)) / (rimW * rimW));
    add += mix(vec3(1.0, 0.98, 0.92), vec3(0.65, 0.80, 1.0), 0.4) * rim * bright * 0.55;

    // Fading god-ray shafts linger briefly after the flash.
    float rays = smoothstep(0.25, 0.0, p);
    for (int i = 0; i < 20; i++) {
      float fi = float(i);
      float a0 = hash11(fi * 1.7 + 13.0) * 6.2831853;
      float spread = 0.07 + hash11(fi * 3.1 + 5.0) * 0.10;
      float ad = mod(ang - a0 + 3.14159265, 6.2831853) - 3.14159265;
      float shaft = exp(-(ad * ad) / (spread * spread));
      float rr = r / max(1.0, R);
      shaft *= exp(-rr * 2.0);
      add += vec3(1.0, 0.82, 0.55) * shaft * (0.5 * rays) * (1.0 - rr);
    }

    // Thinning ejecta glow behind the shock.
    float rr = r / max(1.0, R);
    float inner = exp(-rr * rr * 2.2) * (1.0 - smoothstep(R * 0.45, R, r)) * 0.30 * (1.0 - p * 0.5);
    add += cool * inner;

    // Tangled Rayleigh–Taylor filaments inside the expanding shell.
    for (int i = 0; i < 26; i++) {
      float fi = float(i);
      float a0 = hash11(fi * 1.13 + 3.7) * 6.2831853;
      float curve = sin(r * 0.012 + fi * 3.1 + uTime * 0.8) * 0.24;
      float ad = mod(ang - (a0 + curve + mo * 0.02) + 3.14159265, 6.2831853) - 3.14159265;
      float wfil = 0.05 + hash11(fi * 0.91) * 0.06;
      float fall = exp(-(ad * ad) / (wfil * wfil));
      float r0 = R * 0.55;
      float r1 = R * 1.03;
      if (r > r0 && r < r1) {
        float rt = (r - r0) / (r1 - r0);
        float turb = 0.5 + 0.5 * sin(uTime * 2.1 + fi * 7.0 + r * 0.02 + mo);
        add += cool * fall * (0.32 * turb) * (1.0 - rt * 0.6) * (1.0 - p * 0.2);
      }
    }
  } else if (age < PROG + COLL + FLASH + EJECTA + REMNANT) {
    // Expanding supernova remnant: H-alpha rim, O-III mottled interior.
    float p = (age - PROG - COLL - FLASH - EJECTA) / REMNANT;
    float R = maxR * (0.98 + 0.02 * p);
    float w = 7.0 + p * 22.0;
    float front = exp(-((r - R) * (r - R)) / (w * w));
    add += vec3(0.85, 0.24, 0.16) * front * (0.45 * (1.0 - p * 0.5));

    vec2 nq = (d / max(1.0, R)) * 7.0 + vec2(0.0, uTime * 0.03);
    float m1 = fbm(nq + 3.0);
    float m2 = fbm(nq * 1.6 - 5.0);
    float shell = 1.0 - smoothstep(R * 0.08, R, r);
    float mott = smoothstep(0.30, 0.70, m1 * m2);
    vec3 blue = vec3(0.30, 0.62, 1.0);   // O-III
    vec3 red = vec3(1.0, 0.30, 0.22);    // H-alpha
    vec3 neb = mix(red, blue, smoothstep(0.40, 0.75, m1));
    add += neb * mott * shell * 0.55 * (1.0 - p * 0.6);

    // Pulsar wind nebula at the centre.
    float pulse = 0.5 + 0.5 * sin(uTime * 2.6);
    add += vec3(0.55, 0.85, 1.0) * exp(-(r * r) / (15.0 * 15.0)) * (0.85 + 0.45 * pulse);

    // Opposing jet lobes.
    float jett = exp(-pow((r - 26.0) * 0.05, 2.0)) * (0.5 + 0.5 * sin(ang * 1.0 + uTime * 0.4));
    add += vec3(0.40, 0.60, 1.0) * jett * 0.22 * (1.0 - p * 0.5);
  } else {
    // Quiet gap between cycles.
    float p = (age - PROG - COLL - FLASH - EJECTA - REMNANT) / QUIESC;
    float fade = 1.0 - p;
    add += vec3(0.40, 0.50, 0.80) * exp(-(r * r) / 7.0) * fade * 0.6;
  }

  gl_FragColor = vec4(acesGamma(add), 1.0);
}
`;

/* ──────────────────────── Scene construction ────────────────────────── */

// Composition + physics constants. Tweak these to move the black hole on the
// screen, rescale the disk, or change how fast the supernova light wave crosses
// the nebula.
const NEBULA_R = 420;        // radius of the sky sphere (scene units)
const CAM_DIST = 12.0;       // camera distance from the BH (in Rs)
const BH_FOV_DEG = 72;       // shared vertical fov (both cameras)
const BH_FOV = (BH_FOV_DEG * Math.PI) / 180;
const DISC_IN = 3.1;         // ISCO (Rs)
const DISC_OUT = 10.0;       // disk outer edge (Rs)
const DIRECT_B = 13.0;       // impact-parameter fast-path cutoff (Rs)
const INCL_BASE = 0.34;      // camera elevation above the disk plane (radians)
const BH_NDC = { x: -0.30, y: 0.0 };  // BH shadow centre, NDC (UV 0.35 / 0.50) — large orb, left of centre
const HN_UV = { x: 0.76, y: 0.66 };   // supernova centre, UV (upper-right, far field)
const LIGHT_SPEED = 430;     // supernova light-echo speed (scene units / s)

// Hypernova lifecycle (must mirror the constants inside HYPERNOVA_FS).
const HN_PROG = 7.0;
const HN_COLL = 1.2;
const HN_FLASH = 0.7;
const HN_EJ = 13.0;
const HN_REM = 7.5;
const HN_Q = 2.5;
const HN_CYCLE = HN_PROG + HN_COLL + HN_FLASH + HN_EJ + HN_REM + HN_Q;

// Distance from the fixed R3F camera (z=5) to the effect quads (z=2).
const QUAD_DIST = 3.0;
const FOV_RAD = BH_FOV;

// The visible half-height of the frustum at the quad's plane; the quads are
// scaled to (2*halfH*aspect, 2*halfH) so UV [0,1] maps onto the viewport.
const QUAD_HALF_H = QUAD_DIST * Math.tan(FOV_RAD / 2);

const STELLAR_COLORS = [
  new THREE.Color(0.6, 0.75, 1.0),  // O — blue
  new THREE.Color(0.75, 0.82, 1.0), // B — blue-white
  new THREE.Color(0.95, 0.95, 1.0), // A — white
  new THREE.Color(1.0, 0.98, 0.88), // F — yellow-white
  new THREE.Color(1.0, 0.92, 0.75), // G — yellow
  new THREE.Color(1.0, 0.78, 0.55), // K — orange
  new THREE.Color(1.0, 0.55, 0.42), // M — red
  new THREE.Color(1.0, 0.4, 0.38),  // M giant — deep red
];

/**
 * Builds the starfield point cloud, plus a small cluster of bright stars seated
 * directly behind the black hole (along the camera→BH line). Those are the ones
 * the geodesic pass lenses into arcs around the shadow — the Einstein ring.
 */
function buildStarGeometry() {
  const N = 2600;
  const CLUSTER = 32;
  const pos = new Float32Array((N + CLUSTER) * 3);
  const size = new Float32Array(N + CLUSTER);
  const col = new Float32Array((N + CLUSTER) * 3);
  const ph = new Float32Array(N + CLUSTER);
  const tw = new Float32Array(N + CLUSTER);

  for (let i = 0; i < N; i++) {
    const r = 150 + Math.random() * 110;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    pos[i * 3 + 0] = r * Math.sin(phi) * Math.cos(theta);
    pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta) * 0.82;
    pos[i * 3 + 2] = r * Math.cos(phi);

    const roll = Math.random();
    const stellar = STELLAR_COLORS[Math.floor(Math.random() * STELLAR_COLORS.length)];
    if (roll < 0.55) {
      col[i * 3 + 0] = 0.95 + Math.random() * 0.05;
      col[i * 3 + 1] = 0.93 + Math.random() * 0.07;
      col[i * 3 + 2] = 0.92 + Math.random() * 0.08;
    } else {
      col[i * 3 + 0] = stellar.r;
      col[i * 3 + 1] = stellar.g;
      col[i * 3 + 2] = stellar.b;
    }

    // A few rare, large bright stars punctuate the field.
    const bright = Math.random();
    size[i] = bright < 0.006 ? 3.5 + Math.random() * 4.0 : 0.5 + Math.pow(Math.random(), 3) * 2.4;
    ph[i] = Math.random() * Math.PI * 2;
    tw[i] = 0.4 + Math.random() * 2.4;
  }

  // Lensable cluster: bright warm stars clustered within ~3.5° of the camera→BH
  // line of sight, so they land right in the shadow annulus and get bent into a
  // visible Einstein arc / partial ring of golden pinpoints around the orb.
  {
    const camX = CAM_DIST * Math.cos(INCL_BASE);
    const camY = CAM_DIST * Math.sin(INCL_BASE);
    const bhDir = new THREE.Vector3(-camX, -camY, 0).normalize();
    const up = new THREE.Vector3(0, 1, 0);
    const perp1 = new THREE.Vector3().crossVectors(bhDir, up).normalize();
    const perp2 = new THREE.Vector3().crossVectors(bhDir, perp1).normalize();
    const p = new THREE.Vector3();
    for (let k = 0; k < CLUSTER; k++) {
      const i = N + k;
      const d = 150 + Math.random() * 110;
      const ang = Math.random() * Math.PI * 2;
      const off = 0.004 + Math.random() * 0.06;   // ~0.2°..3.4° from the LOS
      const la = Math.cos(ang) * d * off;
      const lb = Math.sin(ang) * d * off;
      p.copy(bhDir).multiplyScalar(d);
      p.x += perp1.x * la + perp2.x * lb;
      p.y += perp1.y * la + perp2.y * lb;
      p.z += perp1.z * la + perp2.z * lb;
      pos[i * 3 + 0] = p.x;
      pos[i * 3 + 1] = p.y;
      pos[i * 3 + 2] = p.z;
      col[i * 3 + 0] = 1.0;
      col[i * 3 + 1] = 0.78 + Math.random() * 0.14;
      col[i * 3 + 2] = 0.48 + Math.random() * 0.14;
      size[i] = 3.0 + Math.random() * 2.4;
      ph[i] = Math.random() * Math.PI * 2;
      tw[i] = 0.3 + Math.random() * 1.2;
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute("aSize", new THREE.Float32BufferAttribute(size, 1));
  geo.setAttribute("aColor", new THREE.Float32BufferAttribute(col, 3));
  geo.setAttribute("aPhase", new THREE.Float32BufferAttribute(ph, 1));
  geo.setAttribute("aTw", new THREE.Float32BufferAttribute(tw, 1));
  return geo;
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeGalaxyTexture(seed: number, warm = false): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, 256, 256);

  // Bright galactic core — a warm golden variant is used for the galaxy that
  // sits behind the black hole so its lensed light reads as fiery gold.
  const g = ctx.createRadialGradient(128, 128, 0, 128, 128, 44);
  if (warm) {
    g.addColorStop(0, "rgba(255, 240, 196, 0.98)");
    g.addColorStop(0.4, "rgba(255, 196, 120, 0.55)");
    g.addColorStop(1, "rgba(255, 176, 96, 0)");
  } else {
    g.addColorStop(0, "rgba(255, 240, 210, 0.95)");
    g.addColorStop(0.4, "rgba(255, 210, 160, 0.35)");
    g.addColorStop(1, "rgba(255, 200, 150, 0)");
  }
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 256, 256);

  // Logarithmic spiral arms of blue + old-red star clusters.
  const rng = mulberry32(seed);
  for (let i = 0; i < 900; i++) {
    const r = Math.pow(rng(), 0.7) * 112;
    const arm = rng() < 0.5 ? 0 : Math.PI;
    const angle = arm + Math.log(1 + r) * 2.3 + (rng() - 0.5) * 0.35;
    const x = 128 + Math.cos(angle) * r;
    const y = 128 + Math.sin(angle) * r;
    const a = 0.5 * (1 - r / 112);
    ctx.fillStyle =
      rng() < 0.6
        ? warm
          ? `rgba(255, 196, 132, ${a})`
          : `rgba(150, 185, 255, ${a})`
        : `rgba(255, 195, 150, ${a * 0.8})`;
    ctx.fillRect(x, y, 1.4, 1.4);
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// Builds the background scene (nebula + starfield + galaxies) imperatively so
// it can be rendered to RT_bg by the orbiting camera, outside the R3F tree.
function buildBackgroundScene(): {
  scene: THREE.Scene;
  nebMat: THREE.ShaderMaterial;
  starMat: THREE.ShaderMaterial;
  disposables: Array<{ dispose(): void }>;
} {
  const scene = new THREE.Scene();
  const disposables: Array<{ dispose(): void }> = [];

  const nebMat = new THREE.ShaderMaterial({
    vertexShader: NEBULA_VS,
    fragmentShader: NEBULA_FS,
    side: THREE.BackSide,
    depthWrite: false,
    depthTest: false,
    uniforms: {
      uTime: { value: 0 },
      uMouse: { value: new THREE.Vector2(0.5, 0.5) },
      uFlashDir: { value: new THREE.Vector3(0, 0, -1) },
      uBhDir: { value: new THREE.Vector3(0, 0, -1) },
      uFlashR: { value: -1 },
      uFlashE: { value: 0 },
    },
  });
  const nebula = new THREE.Mesh(new THREE.SphereGeometry(NEBULA_R, 48, 32), nebMat);
  nebula.renderOrder = 0;
  nebula.frustumCulled = false;
  scene.add(nebula);
  disposables.push(nebula.geometry, nebMat);

  const starGeo = buildStarGeometry();
  const starMat = new THREE.ShaderMaterial({
    vertexShader: STAR_VS,
    fragmentShader: STAR_FS,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: { uTime: { value: 0 } },
  });
  const stars = new THREE.Points(starGeo, starMat);
  stars.renderOrder = 1;
  stars.frustumCulled = false;
  scene.add(stars);
  disposables.push(starGeo, starMat);

  // The warm golden galaxy seated directly on the black hole's line of sight
  // (base camera→BH direction). The geodesic pass bends its light into the
  // bright Einstein arcs that wrap the shadow — the "light bending around the
  // orb" that reads as real gravitational lensing.
  const baseCamX = CAM_DIST * Math.cos(INCL_BASE);
  const baseCamY = CAM_DIST * Math.sin(INCL_BASE);
  const baseDist = Math.hypot(baseCamX, baseCamY);

  const GALAXIES: Array<{
    pos: [number, number, number];
    scale: number;
    seed: number;
    warm: boolean;
    faceCam: boolean;
  }> = [
    { pos: [-150, 95, -270], scale: 95, seed: 7, warm: false, faceCam: false },
    { pos: [185, -125, -310], scale: 72, seed: 13, warm: false, faceCam: false },
    { pos: [45, 175, -330], scale: 58, seed: 29, warm: false, faceCam: false },
    {
      pos: [(-baseCamX / baseDist) * 260, (-baseCamY / baseDist) * 260, 0],
      scale: 85,
      seed: 41,
      warm: true,
      faceCam: true,
    },
  ];
  for (const g of GALAXIES) {
    const tex = makeGalaxyTexture(g.seed, g.warm);
    const mat = new THREE.MeshBasicMaterial({
      map: tex,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
    });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat);
    mesh.position.set(...g.pos);
    mesh.scale.setScalar(g.scale);
    if (g.faceCam) mesh.lookAt(baseCamX, baseCamY, 0);
    mesh.renderOrder = 2;
    mesh.frustumCulled = false;
    scene.add(mesh);
    disposables.push(tex, mat, mesh.geometry);
  }

  return { scene, nebMat, starMat, disposables };
}

// Recreate (or resize) a render target on demand. RTs carry no depth; the
// passes are all screen-space 2D.
function ensureRT(
  gl: THREE.WebGLRenderer,
  ref: { current: THREE.WebGLRenderTarget | null },
  w: number,
  h: number,
): THREE.WebGLRenderTarget {
  const rt = ref.current;
  if (rt && rt.width === w && rt.height === h) return rt;
  if (rt) rt.dispose();
  const next = new THREE.WebGLRenderTarget(Math.max(2, w), Math.max(2, h), {
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    depthBuffer: false,
    stencilBuffer: false,
  });
  ref.current = next;
  return next;
}

type QuadProps = {
  material: THREE.ShaderMaterial;
  renderOrder: number;
  meshRef?: React.RefObject<THREE.Mesh | null>;
};

function ScreenQuad({ material, renderOrder, meshRef }: QuadProps) {
  return (
    <mesh ref={meshRef} position={[0, 0, 2]} renderOrder={renderOrder}>
      <planeGeometry args={[1, 1]} />
      <primitive object={material} attach="material" />
    </mesh>
  );
}

/* ─────────────────────────── Scene rig ──────────────────────────────── */

function SceneContent() {
  const compositeRef = useRef<THREE.Mesh>(null);
  const hnRef = useRef<THREE.Mesh>(null);
  const mouse = useRef({ x: 0.5, y: 0.5 });
  const reducedRef = useRef(false);
  const timeRef = useRef(0);
  const lastFlashRef = useRef(false);
  const flashStateRef = useRef<{ active: boolean; start: number; dir: THREE.Vector3 }>({
    active: false,
    start: 0,
    dir: new THREE.Vector3(0, 0, -1),
  });
  const perfRef = useRef({ bhScale: 0.5, maxSteps: 240, smooth: 0 });
  const bgRTRef = useRef<THREE.WebGLRenderTarget | null>(null);
  const bhRTRef = useRef<THREE.WebGLRenderTarget | null>(null);

  // Scratch vectors — no per-frame allocation.
  const scratch = useMemo(
    () => ({
      right: new THREE.Vector3(),
      up: new THREE.Vector3(),
      fwd: new THREE.Vector3(),
      ndc: new THREE.Vector3(),
      dir: new THREE.Vector3(),
    }),
    [],
  );

  // Final-screen materials (R3F scene): composite of RT_bh + additive hypernova.
  const compositeMat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: SCREEN_QUAD_VS,
        fragmentShader: COMPOSITE_FS,
        depthTest: false,
        depthWrite: false,
        uniforms: { uTex: { value: null }, uTime: { value: 0 } },
      }),
    [],
  );
  const hnMat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: SCREEN_QUAD_VS,
        fragmentShader: HYPERNOVA_FS,
        transparent: true,
        depthTest: false,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        uniforms: {
          uTime: { value: 0 },
          uPos: { value: new THREE.Vector2(HN_UV.x, HN_UV.y) },
          uMouse: { value: new THREE.Vector2(0.5, 0.5) },
          uMaxR: { value: 0.4 },
          uResolution: { value: new THREE.Vector2(1920, 1080) },
        },
      }),
    [],
  );

  // Imperative lensing rig: background scene + BH quad scene + orbiting camera.
  const rig = useMemo(() => {
    const bg = buildBackgroundScene();
    const bgCamera = new THREE.PerspectiveCamera(BH_FOV_DEG, 1, 0.1, 1200);

    const bhMat = new THREE.ShaderMaterial({
      vertexShader: SCREEN_QUAD_VS,
      fragmentShader: GEODESIC_BH_FS,
      depthTest: false,
      depthWrite: false,
      uniforms: {
        uTime: { value: 0 },
        uMouse: { value: new THREE.Vector2(0.5, 0.5) },
        uResolution: { value: new THREE.Vector2(1920, 1080) },
        uCamPos: { value: new THREE.Vector3(0, 0, 0) },
        uCamRight: { value: new THREE.Vector3(1, 0, 0) },
        uCamUp: { value: new THREE.Vector3(0, 1, 0) },
        uCamFwd: { value: new THREE.Vector3(0, 0, -1) },
        uViewProj: { value: new THREE.Matrix4() },
        uBhOffset: { value: new THREE.Vector2(BH_NDC.x, BH_NDC.y) },
        uCamDist: { value: CAM_DIST },
        uFov: { value: BH_FOV },
        uBgTex: { value: null },
        uDiscInner: { value: DISC_IN },
        uDiscOuter: { value: DISC_OUT },
        uDirectB: { value: DIRECT_B },
        uMaxSteps: { value: 240 },
        uStepBase: { value: 0.12 },
      },
    });
    const bhScene = new THREE.Scene();
    const bhQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), bhMat);
    bhScene.add(bhQuad);
    const bhCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const viewProj = new THREE.Matrix4();

    const disposables = [
      ...bg.disposables,
      bhQuad.geometry,
      bhMat,
    ];
    return { bg, bhMat, bhScene, bhCam, bgCamera, viewProj, disposables };
  }, []);

  useEffect(
    () => () => {
      compositeMat.dispose();
      hnMat.dispose();
      for (const d of rig.disposables) d.dispose();
      bgRTRef.current?.dispose();
      bhRTRef.current?.dispose();
    },
    [compositeMat, hnMat, rig],
  );

  // Track the pointer from the window (the canvas itself is pointer-transparent).
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      mouse.current = {
        x: e.clientX / Math.max(window.innerWidth, 1),
        y: e.clientY / Math.max(window.innerHeight, 1),
      };
    };
    const onTouch = (e: TouchEvent) => {
      const t = e.touches[0];
      if (t) {
        mouse.current = {
          x: t.clientX / Math.max(window.innerWidth, 1),
          y: t.clientY / Math.max(window.innerHeight, 1),
        };
      }
    };
    window.addEventListener("mousemove", onMove, { passive: true });
    window.addEventListener("touchmove", onTouch, { passive: true });

    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    reducedRef.current = mq.matches;
    const onMq = (e: MediaQueryListEvent) => (reducedRef.current = e.matches);
    mq.addEventListener("change", onMq);

    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("touchmove", onTouch);
      mq.removeEventListener("change", onMq);
    };
  }, []);

  useFrame((state, delta) => {
    const gl = state.gl;
    const w = Math.max(gl.domElement.width, 1);
    const h = Math.max(gl.domElement.height, 1);
    const aspect = w / h;
    const speed = reducedRef.current ? 0.15 : 1;
    timeRef.current += Math.min(delta, 0.1) * speed;
    const t = timeRef.current;

    const m = mouse.current;
    const nx = m.x - 0.5;
    const ny = m.y - 0.5;
    const parScale = reducedRef.current ? 0.25 : 1;

    // Frame-time governor: if sustained frames run long, back the BH pass off
    // (lower RT resolution + step budget); restore when fast again.
    perfRef.current.smooth = perfRef.current.smooth * 0.92 + Math.min(delta, 0.1) * 0.08;
    const perf = perfRef.current;
    if (perf.smooth > 0.042) {
      perf.bhScale = Math.max(0.3, perf.bhScale - 0.04);
      perf.maxSteps = Math.max(120, perf.maxSteps - 40);
    } else if (perf.smooth < 0.02 && perf.bhScale < 0.5) {
      perf.bhScale = Math.min(0.5, perf.bhScale + 0.03);
      perf.maxSteps = Math.min(240, perf.maxSteps + 30);
    }

    // Orbiting camera (mouse parallax + slow Lissajous drift). The camera looks
    // at the BH at the origin; its orbit is the "whole scene reacts to pointer".
    const cam = rig.bgCamera;
    cam.fov = BH_FOV_DEG;
    cam.aspect = aspect;
    cam.updateProjectionMatrix();
    const incl = INCL_BASE + ny * 0.09 * parScale;
    const azim = Math.sin(t * 0.05) * 0.05 + nx * 0.16 * parScale;
    cam.position.set(
      CAM_DIST * Math.cos(incl) * Math.cos(azim),
      CAM_DIST * Math.sin(incl),
      CAM_DIST * Math.cos(incl) * Math.sin(azim),
    );
    cam.lookAt(0, 0, 0);
    cam.updateMatrixWorld();
    rig.viewProj.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    const e = cam.matrixWorld.elements;
    scratch.right.set(e[0], e[1], e[2]);
    scratch.up.set(e[4], e[5], e[6]);
    scratch.fwd.set(-e[8], -e[9], -e[10]);

    // Hypernova screen position (drifting + pointer sway).
    const hnU = HN_UV.x + Math.sin(t * 0.05) * 0.03 + nx * 0.06 * parScale;
    const hnV = HN_UV.y + Math.cos(t * 0.07) * 0.02 + ny * 0.05 * parScale;

    // Flash → nebula light-echo wavefront. When the hypernova enters its flash
    // phase we record the sky direction of the blast centre and start a
    // light-speed shell expanding across the nebula.
    const age = t % HN_CYCLE;
    const flashLo = HN_PROG + HN_COLL;
    const flashHi = flashLo + HN_FLASH;
    const inFlash = age >= flashLo && age < flashHi;
    const nebMat = rig.bg.nebMat;
    if (inFlash && !lastFlashRef.current) {
      scratch.ndc.set(hnU * 2 - 1, -(hnV * 2 - 1), 0.5);
      scratch.ndc.unproject(cam);
      scratch.dir.copy(scratch.ndc).sub(cam.position).normalize();
      flashStateRef.current = { active: true, start: t, dir: scratch.dir.clone() };
    }
    lastFlashRef.current = inFlash;
    const fs = flashStateRef.current;
    const flT = fs.active ? t - fs.start : 1e9;
    // The light wave lingers ~4s so its travel across the sky is actually
    // watchable, and it decays slowly enough to keep reading as propagation.
    if (fs.active && flT < 4.0) {
      nebMat.uniforms.uFlashDir.value.copy(fs.dir);
      nebMat.uniforms.uFlashR.value = flT * LIGHT_SPEED;
      nebMat.uniforms.uFlashE.value = Math.exp(-flT * 1.0);
    } else {
      nebMat.uniforms.uFlashE.value = 0;
      if (fs.active && flT >= 4.0) fs.active = false;
    }
    nebMat.uniforms.uTime.value = t;
    (nebMat.uniforms.uMouse.value as THREE.Vector2).set(m.x, m.y);
    // Sky direction of the black hole → the warm halo in the nebula shader is
    // seated on the orb and gets lensed into the golden arcs around it.
    scratch.dir.copy(cam.position).negate().normalize();
    (nebMat.uniforms.uBhDir.value as THREE.Vector3).copy(scratch.dir);
    (rig.bg.starMat.uniforms.uTime as { value: number }).value = t;

    // Pass 1 — background → RT_bg.
    const bgRT = ensureRT(gl, bgRTRef, Math.round(w * 0.8), Math.round(h * 0.8));
    gl.setRenderTarget(bgRT);
    gl.clear(true, true, false);
    gl.render(rig.bg.scene, cam);

    // Pass 2 — geodesic lensing → RT_bh.
    const bhRT = ensureRT(gl, bhRTRef, Math.round(w * perf.bhScale), Math.round(h * perf.bhScale));
    const bhMat = rig.bhMat;
    bhMat.uniforms.uTime.value = t;
    (bhMat.uniforms.uMouse.value as THREE.Vector2).set(m.x, m.y);
    (bhMat.uniforms.uResolution.value as THREE.Vector2).set(w, h);
    (bhMat.uniforms.uCamPos.value as THREE.Vector3).copy(cam.position);
    (bhMat.uniforms.uCamRight.value as THREE.Vector3).copy(scratch.right);
    (bhMat.uniforms.uCamUp.value as THREE.Vector3).copy(scratch.up);
    (bhMat.uniforms.uCamFwd.value as THREE.Vector3).copy(scratch.fwd);
    (bhMat.uniforms.uViewProj.value as THREE.Matrix4).copy(rig.viewProj);
    (bhMat.uniforms.uBhOffset.value as THREE.Vector2).set(
      BH_NDC.x + Math.sin(t * 0.05) * 0.014 + nx * 0.028 * parScale,
      BH_NDC.y + Math.cos(t * 0.07) * 0.01 + ny * 0.028 * parScale,
    );
    bhMat.uniforms.uMaxSteps.value = perf.maxSteps;
    bhMat.uniforms.uBgTex.value = bgRT.texture;
    gl.setRenderTarget(bhRT);
    gl.clear(true, true, false);
    gl.render(rig.bhScene, rig.bhCam);
    gl.setRenderTarget(null);

    // Final — R3F renders the composite quad + hypernova quad to the screen.
    compositeMat.uniforms.uTex.value = bhRT.texture;
    compositeMat.uniforms.uTime.value = t;
    hnMat.uniforms.uTime.value = t;
    (hnMat.uniforms.uPos.value as THREE.Vector2).set(hnU, hnV);
    (hnMat.uniforms.uMouse.value as THREE.Vector2).set(m.x, m.y);
    (hnMat.uniforms.uResolution.value as THREE.Vector2).set(w, h);

    // Scale the full-screen quads to exactly cover the viewport.
    const qw = QUAD_HALF_H * 2 * aspect;
    const qh = QUAD_HALF_H * 2;
    if (compositeRef.current) compositeRef.current.scale.set(qw, qh, 1);
    if (hnRef.current) hnRef.current.scale.set(qw, qh, 1);
  });

  return (
    <>
      {/* Final composite of the lensed black-hole pass. */}
      <ScreenQuad material={compositeMat} renderOrder={2} meshRef={compositeRef} />
      {/* Hypernova: additive full lifecycle on top. */}
      <ScreenQuad material={hnMat} renderOrder={3} meshRef={hnRef} />
    </>
  );
}

/* ──────────────── Runtime opacity probe (WebGL watchdog) ─────────────── */

function Probe({ onDark }: { onDark: () => void }) {
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const camera = useThree((s) => s.camera);
  const framesRef = useRef(0);
  const doneRef = useRef(false);

  // Runs inside the R3F frame loop (not a setTimeout) so the one-off render
  // to a tiny target can't corrupt GL state on software renderers.
  useFrame(() => {
    framesRef.current += 1;
    if (doneRef.current || framesRef.current < 8) return;
    doneRef.current = true;
    try {
      const rt = new THREE.WebGLRenderTarget(8, 8);
      gl.setRenderTarget(rt);
      gl.render(scene, camera);
      const glc = gl.getContext();
      const px = new Uint8Array(4);
      glc.readPixels(4, 4, 1, 1, glc.RGBA, glc.UNSIGNED_BYTE, px);
      gl.setRenderTarget(null);
      rt.dispose();
      // Fully transparent + fully black → the scene never drew; swap to the
      // animated CSS fallback so the palette still reads as a design.
      if (px[3] < 32 && px[0] < 10 && px[1] < 10 && px[2] < 10) {
        onDark();
      }
    } catch {
      // Ignore — a failed probe is not a reason to tear down the canvas.
    }
  });

  return null;
}

/* ──────────────── Animated CSS fallback (no WebGL) ──────────────────── */

const FALLBACK_CSS = `
@keyframes cfTw { 0%,100% { opacity: 0.12; } 50% { opacity: 1; } }
@keyframes cfDriftA { from { transform: translate(0,0) scale(1); } to { transform: translate(6vmin,-3vmin) scale(1.18); } }
@keyframes cfDriftB { from { transform: translate(0,0) scale(1); } to { transform: translate(-5vmin,4vmin) scale(1.22); } }
@keyframes cfBHPulse { 0%,100% { box-shadow: 0 0 40px 6px rgba(255,160,80,0.22); } 50% { box-shadow: 0 0 72px 16px rgba(255,160,80,0.5); } }
@keyframes cfSn { 0%,100% { opacity: 0.45; transform: translate(-50%,-50%) scale(0.92); }
  8% { opacity: 1; transform: translate(-50%,-50%) scale(1.3); }
  40% { opacity: 0.75; transform: translate(-50%,-50%) scale(1.55); }
  100% { opacity: 0.4; transform: translate(-50%,-50%) scale(1.75); } }
`;

function CosmosFallback({ className = "" }: { className?: string }) {
  const stars = useMemo(() => {
    const rng = mulberry32(42);
    return Array.from({ length: 150 }, () => ({
      left: rng() * 100,
      top: rng() * 100,
      size: 0.8 + rng() * 2.2,
      o: 0.25 + rng() * 0.7,
      dur: 2 + rng() * 4,
      delay: rng() * 4,
      warm: rng() < 0.22,
      big: rng() < 0.06,
    }));
  }, []);

  return (
    <div
      className={className}
      aria-hidden="true"
      style={{
        position: "fixed",
        inset: 0,
        overflow: "hidden",
        background:
          "radial-gradient(140% 130% at 30% 38%, #0c1a33 0%, #070d22 42%, #03040e 72%, #010208 100%)",
      }}
    >
      <style>{FALLBACK_CSS}</style>

      {/* Nebula blobs */}
      <div
        style={{
          position: "absolute", left: "10%", top: "26%", width: "62vmin", height: "62vmin",
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(255,96,60,0.20) 0%, rgba(255,60,40,0.07) 42%, transparent 72%)",
          filter: "blur(10px)", animation: "cfDriftA 26s ease-in-out infinite alternate",
        }}
      />
      <div
        style={{
          position: "absolute", left: "55%", top: "48%", width: "70vmin", height: "70vmin",
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(70,150,255,0.16) 0%, rgba(50,110,235,0.06) 45%, transparent 72%)",
          filter: "blur(12px)", animation: "cfDriftB 31s ease-in-out infinite alternate",
        }}
      />
      <div
        style={{
          position: "absolute", left: "28%", top: "58%", width: "50vmin", height: "50vmin",
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(180,110,255,0.14) 0%, rgba(150,80,220,0.05) 45%, transparent 72%)",
          filter: "blur(10px)", animation: "cfDriftA 37s ease-in-out infinite alternate-reverse",
        }}
      />

      {/* Stars */}
      {stars.map((s, i) => (
        <span
          key={i}
          style={{
            position: "absolute",
            left: s.left + "%",
            top: s.top + "%",
            width: s.size + "px",
            height: s.size + "px",
            borderRadius: "50%",
            background: s.warm ? "rgba(255,240,210,1)" : "rgba(220,235,255,1)",
            boxShadow: s.big
              ? "0 0 6px 2px rgba(255,255,255,0.8)"
              : `0 0 ${s.size * 2}px ${s.size * 0.5}px rgba(255,255,255,0.35)`,
            opacity: s.o,
            animation: `cfTw ${s.dur}s ease-in-out ${s.delay}s infinite`,
          }}
        />
      ))}

      {/* Black hole — large orb, left of centre (matches the WebGL composition) */}
      <div
        style={{
          position: "absolute", left: "30%", top: "47%", width: "56vmin", height: "56vmin",
          borderRadius: "50%", transform: "translate(-50%,-50%) rotateX(60deg)",
          background:
            "radial-gradient(circle, rgba(255,200,120,0.9) 0%, rgba(255,120,40,0.55) 22%, rgba(40,20,10,0.65) 45%, rgba(0,0,0,0.95) 62%, transparent 72%)",
          animation: "cfBHPulse 5s ease-in-out infinite",
        }}
      />

      {/* Hypernova — far field, upper right */}
      <div
        style={{
          position: "absolute", left: "76%", top: "20%", width: "38vmin", height: "38vmin",
          borderRadius: "50%",
          background:
            "radial-gradient(circle, rgba(255,255,255,0.95) 0%, rgba(165,200,255,0.6) 20%, rgba(120,160,255,0.25) 45%, transparent 70%)",
          animation: "cfSn 9s ease-in-out infinite",
        }}
      />
    </div>
  );
}

/* ─────────────────────────── Public component ───────────────────────── */

function supportsWebGL(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

export function CosmosWebGL({ className = "" }: { className?: string }) {
  // If WebGL can't be created, show the animated CSS starfield instead of a
  // black void so the palette never reads as a flat colour.
  const [webglOk] = useState(() => supportsWebGL());
  const [broken, setBroken] = useState(false);

  if (!webglOk || broken) {
    return <CosmosFallback className={className} />;
  }

  return (
    <Canvas
      className={className}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: -10,
        pointerEvents: "none",
        background:
          "radial-gradient(140% 130% at 30% 38%, #0c1a33 0%, #070d22 42%, #03040e 72%, #010208 100%)",
      }}
      dpr={[1, 1.8]}
      gl={{
        antialias: false,
        alpha: false,
        premultipliedAlpha: false,
        powerPreference: "high-performance",
      }}
      camera={{ fov: BH_FOV_DEG, near: 0.1, far: 900, position: [0, 0, 5] }}
      fallback={<CosmosFallback className={className} />}
    >
      <SceneContent />
      <Probe onDark={() => setBroken(true)} />
    </Canvas>
  );
}
