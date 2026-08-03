"use client";

// ── Solar Flare — HELIOS raymarched stellar plasma, composed in two passes ─
//
// The HELIOS single-pass raymarcher (helios-sun-simulation.html) is ported as a
// DOUBLE-pass composition, matching the original solar-flare scene's architecture
// while keeping the new HELIOS design (SDF sphere-traced photosphere, 5 transient
// coronal flare loops, CME ejecta, corona halo + streamers):
//
//   Pass 1 — DISK (opaque, into an offscreen render target)
//     The full HELIOS surface march: 110-step sphere-trace over a granulated fbm
//     photosphere with sunspots, limb darkening, the smin-blended flare loops ON
//     the disk, and a hot rim. Miss rays are discarded (the corona pass paints the
//     sky). Detail is pushed beyond the reference: two scales of granulation in the
//     surface SDF, a fine granulation shimmer across the face, limb faculae, and
//     tighter march/normal epsilons so the limb reads as texture, not a smooth ball.
//
//   Pass 2 — CORONA (additive, over the render target)
//     Background + twinkling stars, the exponential corona halo with fbm streamers,
//     a flare-proximity glow (dFlareMinHelper) and the CME ejecta — now layered
//     OVER the sun instead of only outside its silhouette. An analytic photosphere
//     chord attenuates what sits behind the disk, so the corona glows as a hot ring
//     around/at the limb but never washes out the surface detail on the disk face.
//
//   Pass 3 — POST (vignette + ACES + gamma on the composed image)
//     Reads the render target and applies HELIOS's filmic grade to the combined
//     result (the reference graded the whole frame once; so do we).
//
// Adaptations (project conventions, per the deep-space precedent):
//   · Fixed plasma ramp → live palette uniforms (uColorA..uColorD), mixed toward
//     warm-white so the sun retunes live with the PaletteEditor.
//   · HELIOS's uFlareIntensity/uTurbulence sliders → constants FLARE_ACTIVITY=1.0 /
//     TURBULENCE=1.0 plus a transient uFlareBoost (click / ambient eruptions).
//   · uRes/uTanFov → the project's uTanHalfFovY/uAspect camera-basis uniforms.
//   · The reference's FRAG.replace() patch for dFlareMinHelper is inlined.
//   · prefers-reduced-motion freezes uTime.
//
// Controls (as requested, matches the other ray-traced worlds):
//   · left click → solar eruption (flare boost + ejecta burst + chrome event)
//   · right mouse + drag → orbit
//   · shift + scroll → dolly zoom
//   · plain mouse move → gentle parallax
// Interactive page elements are left untouched.

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { supportsWebGL } from "./webglSupport";
import { readPaletteColors, makePalettePoller } from "./worldUtils";
import { emitCosmosEvent } from "./cosmosEvents";

// Fullscreen-quad vertex shader: bypasses matrices, NDC = plane position.
const QUAD_VERT = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

// ── Shared GLSL: hashing / noise / smooth-min ──────────────────────────────
const NOISE_GLSL = `
float hash1(vec3 p){
  p = fract(p*0.3183099 + vec3(0.71,0.113,0.419));
  p *= 17.0;
  return fract(p.x*p.y*p.z*(p.x+p.y+p.z));
}

float vnoise(vec3 x){
  vec3 i = floor(x);
  vec3 f = fract(x);
  f = f*f*(3.0-2.0*f);
  float n000 = hash1(i+vec3(0.0,0.0,0.0));
  float n100 = hash1(i+vec3(1.0,0.0,0.0));
  float n010 = hash1(i+vec3(0.0,1.0,0.0));
  float n110 = hash1(i+vec3(1.0,1.0,0.0));
  float n001 = hash1(i+vec3(0.0,0.0,1.0));
  float n101 = hash1(i+vec3(1.0,0.0,1.0));
  float n011 = hash1(i+vec3(0.0,1.0,1.0));
  float n111 = hash1(i+vec3(1.0,1.0,1.0));
  return mix(
    mix(mix(n000,n100,f.x), mix(n010,n110,f.x), f.y),
    mix(mix(n001,n101,f.x), mix(n011,n111,f.x), f.y),
    f.z
  );
}

float fbm(vec3 p, int oct){
  float v = 0.0;
  float a = 0.5;
  for(int i=0;i<7;i++){
    if(i>=oct) break;
    v += a*vnoise(p);
    p = p*2.03 + vec3(1.7,9.2,4.1);
    a *= 0.5;
  }
  return v;
}

float smin(float a, float b, float k){
  float h = clamp(0.5 + 0.5*(b-a)/k, 0.0, 1.0);
  return mix(b,a,h) - k*h*(1.0-h);
}
`;

// ── Shared GLSL: flare loops, CME ejecta, activity knobs ───────────────────
const FLARE_GLSL = `
#define SUN_R 1.0
#define NUM_FLARES 5
#define PI 3.14159265359

// Activity knobs — HELIOS exposes these as UI sliders; the port pins them to
// the reference defaults and layers a transient uFlareBoost on top.
#define FLARE_ACTIVITY 1.0
#define TURBULENCE 1.0

vec3 hotAccent(){
  return mix(uColorD, vec3(1.0,0.9,0.7), 0.45); // rim / ejecta highlight
}

// Effective flare strength = reference default + click/ambient eruption boost.
float flareActivity(){
  return FLARE_ACTIVITY * (1.0 + 1.6*uFlareBoost);
}

// shared per-flare parameters so the loop geometry and the plasma-ejection
// event (below) stay perfectly in sync.
void flareParams(float idx, out vec3 R, out vec3 T, out float thetaMax, out float H, out float phase, out float cyc){
  vec3 seed = vec3(idx*13.37, idx*7.19, idx*3.71);

  R = normalize(vec3(
    hash1(seed+vec3(1.0,0.0,0.0))-0.5,
    (hash1(seed+vec3(0.0,1.0,0.0))-0.5)*0.7,
    hash1(seed+vec3(0.0,0.0,1.0))-0.5
  ));
  vec3 upr = normalize(vec3(
    hash1(seed+vec3(3.1,0.0,0.0))-0.5,
    hash1(seed+vec3(0.0,4.7,0.0))-0.5,
    hash1(seed+vec3(0.0,0.0,5.3))-0.5
  ));
  T = normalize(cross(R, upr));

  thetaMax = mix(0.15, 0.32, hash1(seed+vec3(9.0,1.0,2.0)));
  H = mix(0.24, 0.62, hash1(seed+vec3(2.0,9.0,4.0)));
  phase = hash1(seed+vec3(5.0,5.0,1.0)) * 23.0;
  cyc = mix(4.0, 8.5, hash1(seed+vec3(6.0,2.0,8.0)));
}

// returns the loop's brightness/flux envelope (0..1) for a given point in
// its eruption cycle: quiet -> rapid rise -> flash peak -> decay -> quiet.
float flareEnvelope(float tcy, float cyc){
  float riseEnd = cyc*0.08;
  float peakEnd = cyc*0.12;
  float decayEnd = cyc*0.40;
  float env;
  if(tcy < riseEnd){
    env = smoothstep(0.0, riseEnd, tcy);
  } else if(tcy < peakEnd){
    env = 1.0;
  } else if(tcy < decayEnd){
    env = mix(1.0, 0.05, smoothstep(peakEnd, decayEnd, tcy));
  } else {
    env = 0.05;
  }
  return env;
}

// analytic projection onto the flare's arc plane; returns signed distance
// to a kinked, tapered, transient plasma filament.
float flareDist(vec3 p, float idx){
  vec3 R, T; float thetaMax, H, phase, cyc;
  flareParams(idx, R, T, thetaMax, H, phase, cyc);

  float tcy = mod(uTime + phase*6.2, cyc);
  float env = flareEnvelope(tcy, cyc) * flareActivity();

  float x = dot(p, R);
  float y = dot(p, T);
  float z = length(p - x*R - y*T);

  float ang = atan(y, x);
  float angC = clamp(ang, -thetaMax, thetaMax);

  float turb = fbm(vec3(angC*5.0, idx*3.0, uTime*0.4), 3) - 0.5;
  float bulge = H * cos(1.5707963*angC/thetaMax) * (1.0 + turb*0.55*TURBULENCE);
  float rC = SUN_R + max(bulge, -0.02);

  vec2 tangent2D = vec2(-sin(angC), cos(angC));
  float wobble = (fbm(vec3(angC*7.0+idx*9.0, idx*2.0+1.0, uTime*0.22), 3) - 0.5) * 0.14 * H * TURBULENCE;
  vec2 curvePt = vec2(cos(angC), sin(angC)) * rC + tangent2D*wobble;

  float zWobble = (fbm(vec3(angC*6.0+idx*5.0, idx*4.0+2.0, uTime*0.28), 3) - 0.5) * 0.16 * H * TURBULENCE;
  float zAdj = abs(z - zWobble);

  float dPlane = length(vec2(x,y) - curvePt);
  float d2 = length(vec2(dPlane, zAdj));

  float edgeFall = 1.0 - smoothstep(thetaMax*0.5, thetaMax, abs(angC));
  float tubeR = mix(0.005, 0.026, edgeFall) * env;

  return d2 - tubeR;
}

// glowing plasma ejected outward from a flare's apex once it flashes —
// a real-time coronal-mass-ejection style event, not a static shape.
vec3 ejectaGlow(vec3 ro, vec3 rd, float tHit, bool hitFlag){
  vec3 total = vec3(0.0);
  for(int i=0;i<NUM_FLARES;i++){
    vec3 R, T; float thetaMax, H, phase, cyc;
    flareParams(float(i), R, T, thetaMax, H, phase, cyc);

    float tcy = mod(uTime + phase*6.2, cyc);
    float peakEnd = cyc*0.12;
    float age = tcy - peakEnd;
    if(age < 0.0) continue;

    float travel = age*0.62;
    float opacity = smoothstep(0.0,0.18,age) * (1.0 - smoothstep(0.0,1.7,travel));
    if(opacity < 0.003) continue;

    vec3 apexDir = R;
    float apexR = SUN_R + H;
    vec3 pos = apexDir*(apexR + travel);

    vec3 toP = pos - ro;
    float tProj = clamp(dot(toP, rd), 0.0, 24.0);
    vec3 cp = ro + rd*tProj;
    float dd = length(cp - pos);
    float size = mix(0.05, 0.30, travel/1.7);
    float glowAmt = exp(-(dd*dd)/(size*size));

    float occl = (hitFlag && tProj > tHit + 0.01) ? 0.0 : 1.0;
    total += hotAccent() * glowAmt * opacity * occl * flareActivity();
  }
  return total;
}

// Corona pass samples flare proximity at the ray's closest-approach point,
// outside the main march loop — the reference patches this helper in with a
// string replace; here it is declared directly.
float dFlareMinHelper(vec3 q){
  float d = 1e5;
  for(int i=0;i<NUM_FLARES;i++){ d = min(d, flareDist(q, float(i))); }
  return d;
}
`;

// ── Shared GLSL: live palette ramps (HELIOS's fixed plasma ramp, retuned) ──
const PALETTE_GLSL = `
vec3 plasmaRamp(float t){
  vec3 c1 = uColorB * 0.30;                          // deep ember
  vec3 c2 = mix(uColorB, uColorA, 0.25);             // dark red-orange
  vec3 c3 = mix(uColorA, uColorC, 0.45);             // bright orange
  vec3 c4 = mix(uColorC, vec3(1.0,0.9,0.7), 0.4);    // gold -> warm-white
  vec3 surf = mix(c1,c2, smoothstep(0.0,0.35,t));
  surf = mix(surf,c3, smoothstep(0.28,0.62,t));
  surf = mix(surf,c4, smoothstep(0.58,0.95,t));
  return surf;
}

vec3 flareRamp(float t){
  vec3 f1 = uColorB * 0.55;                          // dark filament
  vec3 f2 = mix(uColorA, uColorC, 0.30);             // bright orange
  vec3 f3 = mix(uColorC, vec3(1.0), 0.40);           // white-gold core
  vec3 fcol = mix(f1,f2, smoothstep(0.18,0.58,t));
  fcol = mix(fcol,f3, smoothstep(0.55,0.90,t));
  return fcol;
}
`;

const RAY_SPHERE_GLSL = `
bool raySphere(vec3 ro, vec3 rd, float R, out float t0, out float t1) {
  float b = dot(ro, rd);
  float c = dot(ro, ro) - R * R;
  float h = b * b - c;
  if (h < 0.0) return false;
  h = sqrt(h);
  t0 = -b - h;
  t1 = -b + h;
  return true;
}
`;

const ACES_GLSL = `
vec3 acesFilm(vec3 x){
  float a=2.51, b=0.03, c=2.43, d=0.59, e=0.14;
  return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}
`;

// ── Pass 1: opaque raymarched photosphere + flare loops ────────────────────
const DISK_FRAG = `
varying vec2 vUv;
uniform float uTime;
uniform vec3 uCamPos;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uCamForward;
uniform float uTanHalfFovY;
uniform float uAspect;
uniform float uFlareBoost;
uniform vec3 uColorA;
uniform vec3 uColorB;
uniform vec3 uColorC;
uniform vec3 uColorD;
${NOISE_GLSL}
${FLARE_GLSL}
${PALETTE_GLSL}

float dFlareMinGlobal;

// Differential rotation: the photosphere spins fastest at the equator and
// crawls at the poles (analytic approximation of the real shear law), so both
// granulation scales genuinely advect across the disk instead of drifting
// uniformly. Sampling the surface point under the rotation keeps the pattern
// attached to the material.
vec3 rotatePole(vec3 p, float t){
  float lat = clamp(abs(p.y)/max(length(p), 1e-4), 0.0, 1.0);
  float ang = t*0.30*(1.0 - 0.78*lat*lat);
  float c = cos(ang), s = sin(ang);
  return vec3(p.x*c - p.z*s, p.y, p.x*s + p.z*c);
}

float map(vec3 p){
  // Photosphere with two scales of granulation — broad convective cells plus a
  // finer overlying layer, so the limb reads as texture rather than a smooth ball.
  // The pattern is sampled from the differential-rotated surface coordinate so
  // the convective cells advect along the rotation field.
  vec3 sp = rotatePole(p, uTime);
  float gran = fbm(sp*4.2 + vec3(0.0,0.0,uTime*0.035), 5) - 0.5;
  float granFine = fbm(sp*11.0 + vec3(uTime*0.05, uTime*0.03, 0.0), 4) - 0.5;
  float coreR = SUN_R + gran*0.016*TURBULENCE + granFine*0.006;
  float dCore = length(p) - coreR;

  float dFlare = 1e5;
  for(int i=0;i<NUM_FLARES;i++){
    dFlare = min(dFlare, flareDist(p, float(i)));
  }
  dFlareMinGlobal = dFlare;

  return smin(dCore, dFlare, 0.05);
}

vec3 calcNormal(vec3 p){
  vec2 e = vec2(0.0012, 0.0);
  return normalize(vec3(
    map(p+e.xyy) - map(p-e.xyy),
    map(p+e.yxy) - map(p-e.yxy),
    map(p+e.yyx) - map(p-e.yyx)
  ));
}

void main(){
  // three.js fullscreen quad uvs are in [0,1]; HELIOS's fullscreen triangle
  // used [-1,1], so map to NDC before building the ray.
  vec2 uv = vUv * 2.0 - 1.0;
  uv.x *= uAspect;
  vec3 rd = normalize(uCamForward + uv.x*uTanHalfFovY*uCamRight + uv.y*uTanHalfFovY*uCamUp);
  vec3 ro = uCamPos;

  float t = 0.0;
  bool hit = false;
  vec3 p;
  const int STEPS = 110;
  for(int i=0;i<STEPS;i++){
    p = ro + rd*t;
    float d = map(p);
    if(d < 0.0006){ hit = true; break; }
    t += d*0.82;
    if(t > 9.0) break;
  }

  // The corona pass paints the sky; here only the opaque photosphere + loops.
  if(!hit) { discard; }

  map(p);
  float dFlareAtHit = dFlareMinGlobal;
  vec3 n = calcNormal(p);
  vec3 v = -rd;
  float mu = clamp(dot(n,v), 0.0, 1.0);
  bool isFlare = dFlareAtHit < (length(p)-SUN_R+0.02);

  // Advect surface features (granulation, faculae, spots) with the sheared
  // rotation field rather than a uniform time-sweep.
  vec3 spt = rotatePole(p, uTime);
  float n1 = fbm(spt*6.0 + vec3(0.0,0.0,uTime*0.06), 7);
  float n2 = fbm(spt*1.4 + vec3(0.0,0.0,uTime*0.01), 4);
  float spot = smoothstep(0.32,0.14,n2);
  float tp = clamp(n1*0.65+0.38 - spot*0.55, 0.0, 1.0);

  vec3 surf = plasmaRamp(tp);
  float limb = 1.0 - 0.62*(1.0-mu);
  surf *= limb;

  if(isFlare){
    float fn = fbm(p*9.0 + vec3(0.0,0.0,uTime*0.7), 5);
    vec3 fcol = flareRamp(fn);
    float rim = pow(1.0-mu, 2.2);
    fcol += rim*hotAccent()*0.6;
    surf = mix(surf, fcol, 0.92);
  }

  float rimGlow = pow(1.0-mu, 3.2)*0.32;
  surf += rimGlow*hotAccent();

  // Fine granulation shimmer across the face (also advected).
  float fine = fbm(spt*22.0 + vec3(uTime*0.12, uTime*0.07, 0.0), 4);
  surf *= 0.9 + 0.22*fine;

  // Faculae: bright plage patches, most visible toward the limb.
  float facNoise = fbm(spt*3.0 + vec3(0.0,0.0,uTime*0.02), 4);
  float fac = smoothstep(0.62, 0.86, facNoise) * pow(mu, 2.2);
  surf *= 1.0 + 0.5*fac;

  gl_FragColor = vec4(surf * 0.8, 1.0);
}
`;

// ── Pass 2: additive corona / stars / ejecta layered over the sun ──────────
const CORONA_FRAG = `
varying vec2 vUv;
uniform float uTime;
uniform vec3 uCamPos;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uCamForward;
uniform float uTanHalfFovY;
uniform float uAspect;
uniform float uFlareBoost;
uniform vec3 uColorA;
uniform vec3 uColorB;
uniform vec3 uColorC;
uniform vec3 uColorD;
${NOISE_GLSL}
${FLARE_GLSL}
${PALETTE_GLSL}
${RAY_SPHERE_GLSL}

void main(){
  vec2 uv = vUv * 2.0 - 1.0;
  uv.x *= uAspect;
  vec3 rd = normalize(uCamForward + uv.x*uTanHalfFovY*uCamRight + uv.y*uTanHalfFovY*uCamUp);
  vec3 ro = uCamPos;

  // Analytic photosphere chord: the disk occludes what sits behind it. atten
  // goes 0 across the disk face -> 1 just off the limb, so the corona glows as
  // a hot ring around (and right at) the edge instead of washing out the detail.
  float th0, th1;
  bool diskHit = raySphere(ro, rd, SUN_R, th0, th1);
  float chord = diskHit ? max(th1 - max(th0, 0.0), 0.0) : 0.0;
  float atten = 1.0 - smoothstep(0.0, 2.0, chord);

  // Background + twinkling stars (hidden behind the disk).
  vec3 col = vec3(0.010,0.007,0.014) * (0.6+0.4*rd.y);
  vec3 sg = floor(rd*420.0);
  float sh = hash1(sg);
  float star = smoothstep(0.9935,1.0,sh);
  float tw = 0.6+0.4*sin(uTime*2.0 + sh*80.0);
  col += vec3(star)*tw*0.75;
  col *= atten;

  // Corona halo + fbm streamers, tinted by altitude, hottest at the limb.
  float b = dot(-ro, rd);
  if(b > 0.0){
    vec3 closest = ro + rd*b;
    float distC = length(closest);
    float halo = exp(-max(distC-SUN_R,0.0)*3.2);
    float ang = atan(closest.z, closest.x);
    vec3 sP = vec3(cos(ang)*3.1, sin(ang)*3.1, distC*2.2 - uTime*0.06);
    // Broad streamer field plus a finer, faster-rolling filament octave so the
    // corona resolves into sharp radial filaments over the smooth halo.
    float streamer = fbm(sP, 5);
    float filament = fbm(sP*3.2 + vec3(0.0,0.0,uTime*0.11), 4) - 0.5;
    float glow = halo * mix(0.4, 1.45, streamer) * (1.0 + 0.35*filament);

    vec3 coronaInner = mix(uColorA, vec3(1.0,0.9,0.75), 0.35);
    vec3 coronaOuter = mix(uColorC, vec3(1.0,0.85,0.55), 0.45);
    vec3 corona = mix(coronaInner, coronaOuter, smoothstep(SUN_R, SUN_R+0.45, distC));

    col += corona * glow * (0.4+0.3*flareActivity()) * atten;

    // Flare-proximity glow around the active loops, sampled at closest approach.
    float flareHalo = exp(-max(dFlareMinHelper(closest)-0.0,0.0)*10.0) * halo;
    col += hotAccent() * flareHalo * 0.28 * atten;
  }

  // CME ejecta — glowing plasma in front of the disk is added, occluded behind it.
  float e0, e1;
  bool eHit = raySphere(ro, rd, SUN_R, e0, e1);
  col += ejectaGlow(ro, rd, eHit ? max(e0, 0.0) : 0.0, eHit);

  gl_FragColor = vec4(col, 1.0);
}
`;

// ── Pass 3: filmic grade on the composed double-pass image ─────────────────
const POST_FRAG = `
varying vec2 vUv;
uniform sampler2D uTex;
uniform float uAspect;
${ACES_GLSL}
void main(){
  vec2 uv = vUv * 2.0 - 1.0;
  uv.x *= uAspect;
  vec3 col = texture2D(uTex, vUv).rgb;

  float vig = smoothstep(1.5, 0.35, length(uv));
  col *= mix(0.72, 1.0, vig);

  col = acesFilm(col*0.6);
  col = pow(col, vec3(1.0/2.2));

  gl_FragColor = vec4(col, 1.0);
}
`;

export function SolarFlareWebGL({ className = "" }: { className?: string }) {
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
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.autoClear = false; // manual pass-by-pass clearing
    host.appendChild(renderer.domElement);
    renderer.domElement.style.position = "fixed";
    renderer.domElement.style.inset = "0";
    renderer.domElement.style.display = "block";

    const camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      0.1,
      300,
    );
    const clock = new THREE.Clock();
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    // ── Palette → live uniforms ───────────────────────────────────────────
    const colA = { value: new THREE.Color(0xf97316) }; // primary (orange)
    const colB = { value: new THREE.Color(0xef4444) }; // secondary (red)
    const colC = { value: new THREE.Color(0xfacc15) }; // tertiary (gold)
    const colD = { value: new THREE.Color(0xfb923c) }; // accent
    const applyColors = (p: ReturnType<typeof readPaletteColors>) => {
      colA.value.copy(p.primary);
      colB.value.copy(p.secondary);
      colC.value.copy(p.tertiary);
      colD.value.copy(p.accent);
    };
    applyColors(readPaletteColors());
    const poll = makePalettePoller(400, applyColors);

    // ── Camera basis uniforms shared by the ray-trace passes ──────────────
    const camU = {
      uCamPos: { value: new THREE.Vector3() },
      uCamRight: { value: new THREE.Vector3() },
      uCamUp: { value: new THREE.Vector3() },
      uCamForward: { value: new THREE.Vector3() },
      uTanHalfFovY: { value: Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) },
      uAspect: { value: camera.aspect },
    };
    const boost = { value: 0 }; // 0..1 transient eruption burst

    const diskUniforms = {
      uTime: { value: 0 },
      uCamPos: camU.uCamPos,
      uCamRight: camU.uCamRight,
      uCamUp: camU.uCamUp,
      uCamForward: camU.uCamForward,
      uTanHalfFovY: camU.uTanHalfFovY,
      uAspect: camU.uAspect,
      uFlareBoost: boost,
      uColorA: colA,
      uColorB: colB,
      uColorC: colC,
      uColorD: colD,
    };
    const coronaUniforms = {
      uTime: { value: 0 },
      uCamPos: camU.uCamPos,
      uCamRight: camU.uCamRight,
      uCamUp: camU.uCamUp,
      uCamForward: camU.uCamForward,
      uTanHalfFovY: camU.uTanHalfFovY,
      uAspect: camU.uAspect,
      uFlareBoost: boost,
      uColorA: colA,
      uColorB: colB,
      uColorC: colC,
      uColorD: colD,
    };

    // ── Pass 1 scene: opaque photosphere + loops ──────────────────────────
    const diskMat = new THREE.ShaderMaterial({
      uniforms: diskUniforms,
      vertexShader: QUAD_VERT,
      fragmentShader: DISK_FRAG,
      side: THREE.DoubleSide,
    });
    const diskQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), diskMat);
    diskQuad.frustumCulled = false;
    const diskScene = new THREE.Scene();
    diskScene.add(diskQuad);

    // ── Pass 2 scene: additive corona / stars / ejecta ────────────────────
    const coronaMat = new THREE.ShaderMaterial({
      uniforms: coronaUniforms,
      vertexShader: QUAD_VERT,
      fragmentShader: CORONA_FRAG,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: false,
      side: THREE.DoubleSide,
    });
    const coronaQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), coronaMat);
    coronaQuad.frustumCulled = false;
    const coronaScene = new THREE.Scene();
    coronaScene.add(coronaQuad);

    // ── Composition target: disk + additive corona combined here ──────────
    // Half-float so the HDR sum (opaque disk + additive corona/ejecta) survives
    // to the filmic grade instead of clipping to white in an 8-bit buffer.
    const dpr = renderer.getPixelRatio();
    const composeRT = new THREE.WebGLRenderTarget(
      Math.floor(window.innerWidth * dpr),
      Math.floor(window.innerHeight * dpr),
      {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat,
        type: THREE.HalfFloatType,
        // MSAA the composite: renderer antialias only covers the default
        // framebuffer, so the blitted solar scene would otherwise stay jagged.
        samples: renderer.capabilities.isWebGL2 ? 4 : 0,
      },
    );

    // ── Pass 3 scene: filmic grade on the composed image ──────────────────
    const postMat = new THREE.ShaderMaterial({
      uniforms: {
        uTex: { value: composeRT.texture },
        uAspect: { value: camera.aspect },
      },
      vertexShader: QUAD_VERT,
      fragmentShader: POST_FRAG,
    });
    const postQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), postMat);
    postQuad.frustumCulled = false;
    const postScene = new THREE.Scene();
    postScene.add(postQuad);

    const postCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    // ── Interaction state (matches the reference camera scale: SUN_R = 1) ─
    const orbit = {
      azimuth: 0.6,
      polar: 0.22,
      radius: 3.4,
      target: new THREE.Vector3(0, 0, 0),
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
      orbit.azimuth -= (e.clientX - lastPX) * 0.0045;
      orbit.polar = Math.max(
        -1.3,
        Math.min(1.3, orbit.polar - (e.clientY - lastPY) * 0.0045),
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
      boost.value = 1;
      emitCosmosEvent({ type: "solar-eruption", heat: 0.65, hue: 24, x: ndcX, y: ndcY });
    };
    const onWheel = (e: WheelEvent) => {
      if (!e.shiftKey) return;
      e.preventDefault();
      orbit.radius = Math.max(1.7, Math.min(7.0, orbit.radius + e.deltaY * 0.0022));
    };
    const onContextMenu = (e: Event) => e.preventDefault();
    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      camU.uAspect.value = camera.aspect;
      postMat.uniforms.uAspect.value = camera.aspect;
      const p = renderer.getPixelRatio();
      composeRT.setSize(
        Math.floor(window.innerWidth * p),
        Math.floor(window.innerHeight * p),
      );
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("wheel", onWheel, { passive: false });
    window.addEventListener("contextmenu", onContextMenu);
    window.addEventListener("resize", onResize);

    // ── Ambient eruptions (softer flare bursts on a timer) ────────────────
    let ambientTimer = 9 + Math.random() * 6;
    function updateAmbient(dt: number) {
      ambientTimer -= dt;
      if (ambientTimer > 0) return;
      ambientTimer = 10 + Math.random() * 6;
      boost.value = Math.max(boost.value, 0.55);
      emitCosmosEvent({
        type: "solar-eruption",
        heat: 0.25,
        hue: 32,
        x: 0,
        y: 0,
      });
    }

    // ── Camera basis temp storage ─────────────────────────────────────────
    const _right = new THREE.Vector3();
    const _up = new THREE.Vector3();
    const _fwd = new THREE.Vector3();

    // ── Animate loop ──────────────────────────────────────────────────────
    let raf = 0;
    let sceneTime = 0;
    function animate() {
      raf = requestAnimationFrame(animate);
      const dt = Math.min(clock.getDelta(), 0.05);
      sceneTime += reduced ? 0 : dt;
      const t = sceneTime;
      poll(performance.now());

      if (!reduced) orbit.azimuth += dt * 0.025;
      boost.value = Math.max(0, boost.value - dt * 0.5);
      if (!reduced) updateAmbient(dt);

      const pa = orbit.polar;
      const aa = orbit.azimuth;
      const r = orbit.radius;
      // Gentle mouse parallax, proportional to orbit distance.
      const px = mouse.x * r * 0.085;
      const py = -mouse.y * r * 0.043;
      camera.position.set(
        orbit.target.x + r * Math.cos(pa) * Math.sin(aa) + px,
        orbit.target.y + r * Math.sin(pa) + py,
        orbit.target.z + r * Math.cos(pa) * Math.cos(aa),
      );
      camera.lookAt(orbit.target);
      camera.updateMatrixWorld(true);
      camera.matrixWorldInverse.copy(camera.matrixWorld).invert();

      // Rebuild the camera ray basis for the ray-trace passes.
      camera.matrixWorld.extractBasis(_right, _up, _fwd);
      _fwd.negate();
      camU.uCamPos.value.copy(camera.position);
      camU.uCamRight.value.copy(_right);
      camU.uCamUp.value.copy(_up);
      camU.uCamForward.value.copy(_fwd);

      diskMat.uniforms.uTime.value = t;
      coronaMat.uniforms.uTime.value = t;

      // Double pass into the composition target: opaque photosphere first, then
      // the additive corona/ejecta layered over it (hot at the limb, hidden on
      // the face) — then grade the composed image to screen.
      renderer.setRenderTarget(composeRT);
      renderer.setClearColor(0x000000, 1);
      renderer.clear();
      renderer.render(diskScene, postCamera);
      renderer.render(coronaScene, postCamera);
      renderer.setRenderTarget(null);
      renderer.clear();
      renderer.render(postScene, postCamera);
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
      diskQuad.geometry.dispose();
      coronaQuad.geometry.dispose();
      postQuad.geometry.dispose();
      diskMat.dispose();
      coronaMat.dispose();
      postMat.dispose();
      composeRT.dispose();
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
            "radial-gradient(ellipse at 50% 60%, hsl(var(--palette-primary) / 0.4), hsl(var(--palette-tertiary) / 0.12) 55%, transparent 75%)",
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
