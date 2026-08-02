"use client";

// ── Solar Flare — the living-sun world for the "solar-flare" palette ──────
//
// Self-contained WebGL background modeled on CosmosWebGL.tsx's skeleton, but
// with the sun upgraded to a screen-space ray-traced volume. Two fullscreen
// passes replace the old sphere mesh + corona billboards:
//
//   1. Disk pass   — per-pixel ray-sphere intersection against the photosphere
//      (R = 4), granulated fbm surface with sunspots / flicker / limb
//      darkening. Opaque and depth-written via gl_FragDepth, so the additive
//      prominence / wind / jet particles depth-test against the real sun shape.
//   2. Corona pass — additive volumetric ray-march through the fbm density
//      shell surrounding the photosphere (white→gold→orange→red altitude
//      tint), with analytic loop-prominence arcs and self-occlusion via
//      transmittance.
//
// Stars render first, then disk, corona, and finally particles (near-side
// prominences/jets over the disk, far-side culled). Colors read live from
// --palette-* so the PaletteEditor retunes the star in real time.

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { supportsWebGL } from "./webglSupport";
import { readPaletteColors, makePalettePoller } from "./worldUtils";
import { emitCosmosEvent } from "./cosmosEvents";

// ── Shared GLSL helpers for both ray-trace passes ──────────────────────────
const NOISE_GLSL = `
float hash(vec3 p){ p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float noise(vec3 x){
  vec3 i = floor(x); vec3 f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash(i), hash(i + vec3(1,0,0)), f.x),
                 mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
                 mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z);
}
float fbm(vec3 p){
  float v = 0.0; float a = 0.5;
  for (int i = 0; i < 4; i++) { v += a * noise(p); p *= 2.03; a *= 0.5; }
  return v;
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

// Fullscreen-quad vertex shader: bypasses matrices, NDC = plane position.
const QUAD_VERT = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

// ── Disk pass: ray-traced photosphere (opaque, writes gl_FragDepth) ───────
const DISK_FRAG = `
uniform float uTime;
uniform vec3 uCamPos;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uCamForward;
uniform float uTanHalfFovY;
uniform float uAspect;
uniform mat4 uProj;
uniform mat4 uView;
uniform vec3 uColorA;
uniform vec3 uColorB;
uniform vec3 uColorC;
varying vec2 vUv;
${NOISE_GLSL}
${RAY_SPHERE_GLSL}
void main() {
  vec2 ndc = vUv * 2.0 - 1.0;
  vec3 ro = uCamPos;
  vec3 rd = normalize(uCamForward + uCamRight * ndc.x * uTanHalfFovY * uAspect + uCamUp * ndc.y * uTanHalfFovY);
  const float R = 4.0;
  float t0, t1;
  if (!raySphere(ro, rd, R, t0, t1)) { discard; }
  float t = max(t0, 0.0);
  vec3 p = ro + rd * t;
  vec3 n = normalize(p);
  float nf = fbm(p * 1.7 + uTime * 0.12);
  // Bright granulation, softer sunspots where the broad noise dips. The ramp is
  // shallow and capped at 55% so the disk reads as organic cells, never the
  // hard-edged blocky blotches of a low-poly / voxel sun.
  vec3 col = mix(uColorA, uColorB, nf * 0.8 + 0.2);
  col *= 1.0 - smoothstep(0.45, 0.12, nf) * 0.55;
  // Fine cellular granulation on top.
  float fine = fbm(p * 6.5 + vec3(uTime * 0.35, uTime * 0.18, 0.0));
  col *= 0.86 + 0.28 * fine;
  // Gentle living flicker.
  col *= 0.94 + 0.06 * sin(uTime * 3.0 + nf * 24.0);
  // Limb darkening toward the edge.
  vec3 V = normalize(uCamPos - p);
  float ndv = abs(dot(n, V));
  col *= mix(0.5, 1.0, smoothstep(0.0, 0.9, ndv));
  // Hot edge glow.
  col += uColorB * smoothstep(0.9, 1.0, ndv) * 0.25;
  // Real sphere depth so additive particles cull against the disk shape.
  vec4 clip = uProj * (uView * vec4(p, 1.0));
  gl_FragDepth = clip.z / clip.w * 0.5 + 0.5;
  gl_FragColor = vec4(col, 1.0);
}
`;

// ── Corona pass: volumetric ray-marched fbm shell (additive) ──────────────
const CORONA_FRAG = `
uniform float uTime;
uniform vec3 uCamPos;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uCamForward;
uniform float uTanHalfFovY;
uniform float uAspect;
uniform vec3 uColorA;
uniform vec3 uColorB;
uniform vec3 uColorC;
uniform vec3 uColorD;
varying vec2 vUv;
${NOISE_GLSL}
${RAY_SPHERE_GLSL}
// Analytic loop-prominence arcs: three rotating tori that read as coronal loops.
float loopDensity(vec3 p) {
  float d = 0.0;
  vec3 n;
  float ca, sa;
  vec3 ppl;
  float rp;
  float dist;
  // Loop 1 — equatorial arcade.
  n = vec3(0.0, 1.0, 0.0);
  ca = cos(uTime * 0.10);
  sa = sin(uTime * 0.10);
  n = vec3(n.x * ca + n.z * sa, n.y, -n.x * sa + n.z * ca);
  ppl = p - n * dot(p, n);
  rp = length(ppl) + 1e-4;
  dist = abs(rp - 6.8);
  d += exp(-dist * dist * 3.0) * 0.9;
  // Loop 2 — tilted arcade.
  n = vec3(0.5, 0.86, 0.1);
  ca = cos(uTime * 0.07 + 2.1);
  sa = sin(uTime * 0.07 + 2.1);
  n = vec3(n.x * ca + n.z * sa, n.y, -n.x * sa + n.z * ca);
  ppl = p - n * dot(p, n);
  rp = length(ppl) + 1e-4;
  dist = abs(rp - 8.6);
  d += exp(-dist * dist * 1.8) * 0.7;
  // Loop 3 — high-latitude arcade.
  n = vec3(-0.35, 0.9, -0.25);
  ca = cos(uTime * 0.05 + 4.0);
  sa = sin(uTime * 0.05 + 4.0);
  n = vec3(n.x * ca + n.z * sa, n.y, -n.x * sa + n.z * ca);
  ppl = p - n * dot(p, n);
  rp = length(ppl) + 1e-4;
  dist = abs(rp - 7.2);
  d += exp(-dist * dist * 2.6) * 0.8;
  return d;
}
void main() {
  vec2 ndc = vUv * 2.0 - 1.0;
  vec3 ro = uCamPos;
  vec3 rd = normalize(uCamForward + uCamRight * ndc.x * uTanHalfFovY * uAspect + uCamUp * ndc.y * uTanHalfFovY);
  const float R = 4.0;
  const float ROUTER = 12.0; // shell radius: ~3x the photosphere
  float ts0, ts1;
  // Rays that never enter the outer shell see no corona.
  if (!raySphere(ro, rd, ROUTER, ts0, ts1)) { gl_FragColor = vec4(0.0); return; }
  float t = max(ts0, 0.0);
  // Inside the disk silhouette the opaque photosphere hides the corona, except
  // right at the limb where the chord is short (that is the glowing edge).
  float atten = 1.0;
  float d0, d1;
  if (raySphere(ro, rd, R, d0, d1)) {
    atten = 1.0 - smoothstep(0.0, 4.0, d1 - d0);
    if (atten < 0.02) { gl_FragColor = vec4(0.0); return; }
  }
  const int STEPS = 48;
  float step = (ROUTER * 2.0) / float(STEPS);
  vec3 scatter = vec3(0.0);
  float transmittance = 1.0;
  for (int i = 0; i < STEPS; i++) {
    vec3 p = ro + rd * (t + (float(i) + 0.5) * step);
    float r = length(p);
    if (r > ROUTER + 0.05) break;
    float alt = r - R;
    if (r < R || alt > 6.0) continue; // inside the disk, or past the visible shell
    float base = fbm(p * 0.42 + uTime * 0.05);
    float shell = (0.35 + 0.65 * base) * exp(-alt * 0.55);
    float loop = loopDensity(p) * 0.6;
    float dens = shell + loop;
    if (dens > 0.004) {
      // White-hot near the surface → gold → orange → red far out.
      vec3 tint = mix(vec3(1.0, 0.9, 0.75), uColorB, smoothstep(0.0, 1.0, alt));
      tint = mix(tint, uColorA, smoothstep(1.2, 3.0, alt));
      tint = mix(tint, uColorC, smoothstep(3.0, 8.0, alt));
      vec3 loopTint = mix(uColorD, uColorC, smoothstep(1.5, 5.0, alt));
      scatter += (shell * tint + loop * loopTint) * step * transmittance;
      transmittance *= exp(-dens * step * 1.4);
      if (transmittance < 0.02) break;
    }
  }
  gl_FragColor = vec4(scatter * atten, 1.0);
}
`;

// Flash texture for eruptions.
function makeFlashTexture(): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const ctx = c.getContext("2d")!;
  const g = ctx.createRadialGradient(128, 128, 4, 128, 128, 124);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.35, "rgba(255,255,255,0.6)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 256, 256);
  return c;
}

const SUN_R = 4;

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
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x0a0503, 1);
    renderer.autoClear = false; // manual pass-by-pass clearing below
    host.appendChild(renderer.domElement);
    renderer.domElement.style.position = "fixed";
    renderer.domElement.style.inset = "0";
    renderer.domElement.style.display = "block";

    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x0a0503, 40, 120);
    const particleScene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      0.1,
      300,
    );
    const clock = new THREE.Clock();
    const disposables: THREE.Texture[] = [];
    // Round soft-glow sprite for every particle system — without a map the
    // points render as hard-edged squares, which reads as "Minecraft sun".
    const glowTex = new THREE.CanvasTexture(makeFlashTexture());
    disposables.push(glowTex);
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

    // ── Camera basis uniforms shared by both ray-trace passes ─────────────
    const camU = {
      uCamPos: { value: new THREE.Vector3() },
      uCamRight: { value: new THREE.Vector3() },
      uCamUp: { value: new THREE.Vector3() },
      uCamForward: { value: new THREE.Vector3() },
      uTanHalfFovY: { value: Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) },
      uAspect: { value: camera.aspect },
    };

    // ── Disk pass scene (opaque photosphere) ──────────────────────────────
    const diskMat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uCamPos: camU.uCamPos,
        uCamRight: camU.uCamRight,
        uCamUp: camU.uCamUp,
        uCamForward: camU.uCamForward,
        uTanHalfFovY: camU.uTanHalfFovY,
        uAspect: camU.uAspect,
        uProj: { value: camera.projectionMatrix },
        uView: { value: camera.matrixWorldInverse },
        uColorA: colA,
        uColorB: colC,
        uColorC: colB,
      },
      vertexShader: QUAD_VERT,
      fragmentShader: DISK_FRAG,
      side: THREE.DoubleSide,
    });
    const diskQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), diskMat);
    diskQuad.frustumCulled = false;
    const diskScene = new THREE.Scene();
    diskScene.add(diskQuad);

    // ── Corona pass scene (additive volumetric shell) ─────────────────────
    const coronaMat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uCamPos: camU.uCamPos,
        uCamRight: camU.uCamRight,
        uCamUp: camU.uCamUp,
        uCamForward: camU.uCamForward,
        uTanHalfFovY: camU.uTanHalfFovY,
        uAspect: camU.uAspect,
        uColorA: colA,
        uColorB: colC,
        uColorC: colB,
        uColorD: colD,
      },
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

    const postCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    // ── Background stars ──────────────────────────────────────────────────
    const STARS = 900;
    const starPos = new Float32Array(STARS * 3);
    for (let i = 0; i < STARS; i++) {
      const dir = new THREE.Vector3(
        Math.random() * 2 - 1,
        Math.random() * 2 - 1,
        Math.random() * 2 - 1,
      ).normalize();
      const r = 70 + Math.random() * 90;
      starPos[i * 3] = dir.x * r;
      starPos[i * 3 + 1] = dir.y * r;
      starPos[i * 3 + 2] = dir.z * r;
    }
    const starGeo = new THREE.BufferGeometry();
    starGeo.setAttribute("position", new THREE.BufferAttribute(starPos, 3));
    const starMat = new THREE.PointsMaterial({
      color: 0xffd9b8,
      size: 0.25,
      transparent: true,
      opacity: 0.6,
      sizeAttenuation: true,
      map: glowTex,
      alphaTest: 0.05,
    });
    scene.add(new THREE.Points(starGeo, starMat));

    // ── Prominence arcs (curved helical particle streamers) ───────────────
    const PROM = 520;
    const promDir = new Float32Array(PROM * 3);
    const promPhase = new Float32Array(PROM);
    const promSpeed = new Float32Array(PROM);
    const promTwist = new Float32Array(PROM);
    const promReach = new Float32Array(PROM);
    const promCol = new Float32Array(PROM * 3);
    for (let i = 0; i < PROM; i++) {
      resetProm(i);
    }
    function resetProm(i: number) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      promDir[i * 3] = Math.sin(phi) * Math.cos(theta);
      promDir[i * 3 + 1] = Math.cos(phi);
      promDir[i * 3 + 2] = Math.sin(phi) * Math.sin(theta);
      promPhase[i] = Math.random(); // scattered start
      promSpeed[i] = 0.05 + Math.random() * 0.1;
      promTwist[i] = (Math.random() - 0.5) * 2.4;
      promReach[i] = 1.5 + Math.random() * 4.5;
    }
    function colorProm(i: number, out: THREE.Color, a: THREE.Color, b: THREE.Color) {
      const t = Math.min(1, promPhase[i] * 1.4);
      out.copy(a).lerp(b, t);
    }
    const promPositions = new Float32Array(PROM * 3);
    const promColors = new Float32Array(PROM * 3);
    const promGeo = new THREE.BufferGeometry();
    promGeo.setAttribute("position", new THREE.BufferAttribute(promPositions, 3));
    promGeo.setAttribute("color", new THREE.BufferAttribute(promColors, 3));
    const promMat = new THREE.PointsMaterial({
      size: 0.22,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
      map: glowTex,
      alphaTest: 0.05,
    });
    const proms = new THREE.Points(promGeo, promMat);
    particleScene.add(proms);
    const _tmp = new THREE.Color();

    function updateProms(dt: number) {
      const pos = proms.geometry.getAttribute("position") as THREE.BufferAttribute;
      const col = proms.geometry.getAttribute("color") as THREE.BufferAttribute;
      for (let i = 0; i < PROM; i++) {
        promPhase[i] += promSpeed[i] * dt;
        if (promPhase[i] >= 1) {
          resetProm(i);
          continue;
        }
        const ph = promPhase[i];
        const arc = Math.pow(ph, 0.75) * promReach[i];
        // Helical twist around the base direction.
        const tw = ph * promTwist[i];
        const dx = promDir[i * 3];
        const dy = promDir[i * 3 + 1];
        const dz = promDir[i * 3 + 2];
        // Rotate base dir around world Y by `tw` for the arc curvature.
        const c = Math.cos(tw);
        const s = Math.sin(tw);
        const rx = dx * c + dz * s;
        const rz = -dx * s + dz * c;
        const rr = (SUN_R + 0.15) + arc;
        const x = rx * rr;
        const y = dy * rr + Math.sin(ph * Math.PI) * 0.4; // lift off the limb
        const z = rz * rr;
        pos.setXYZ(i, x, y, z);
        colorProm(i, _tmp, colA.value, colC.value);
        col.setXYZ(i, _tmp.r, _tmp.g, _tmp.b);
      }
      pos.needsUpdate = true;
      col.needsUpdate = true;
    }

    // ── Solar wind (slow radial stream) ───────────────────────────────────
    const WIND = 320;
    const windDir = new Float32Array(WIND * 3);
    const windPos = new Float32Array(WIND * 3);
    const windSpeed = new Float32Array(WIND);
    for (let i = 0; i < WIND; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      windDir[i * 3] = Math.sin(phi) * Math.cos(theta);
      windDir[i * 3 + 1] = Math.cos(phi);
      windDir[i * 3 + 2] = Math.sin(phi) * Math.sin(theta);
      windSpeed[i] = 1.2 + Math.random() * 2.2;
      const r0 = SUN_R + 0.3 + Math.random() * 4;
      windPos[i * 3] = windDir[i * 3] * r0;
      windPos[i * 3 + 1] = windDir[i * 3 + 1] * r0;
      windPos[i * 3 + 2] = windDir[i * 3 + 2] * r0;
    }
    const windGeo = new THREE.BufferGeometry();
    windGeo.setAttribute("position", new THREE.BufferAttribute(windPos, 3));
    const windMat = new THREE.PointsMaterial({
      color: colC.value,
      size: 0.16,
      transparent: true,
      opacity: 0.5,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
      map: glowTex,
      alphaTest: 0.05,
    });
    const wind = new THREE.Points(windGeo, windMat);
    particleScene.add(wind);

    function updateWind(dt: number) {
      const pos = wind.geometry.getAttribute("position") as THREE.BufferAttribute;
      for (let i = 0; i < WIND; i++) {
        let r =
          Math.hypot(pos.getX(i), pos.getY(i), pos.getZ(i)) + windSpeed[i] * dt;
        if (r > 26) r = SUN_R + 0.3 + Math.random() * 3;
        pos.setXYZ(
          i,
          (windDir[i * 3] * r),
          (windDir[i * 3 + 1] * r),
          (windDir[i * 3 + 2] * r),
        );
      }
      pos.needsUpdate = true;
    }

    // ── Eruption jets (click + ambient) ───────────────────────────────────
    const JETS = 160;
    const jetPos = new Float32Array(JETS * 3);
    const jetVel = new Float32Array(JETS * 3);
    const jetLife = new Float32Array(JETS);
    const jetMax = new Float32Array(JETS);
    for (let i = 0; i < JETS; i++) jetLife[i] = 0;
    const jetGeo = new THREE.BufferGeometry();
    jetGeo.setAttribute("position", new THREE.BufferAttribute(jetPos, 3));
    const jetMat = new THREE.PointsMaterial({
      color: colC.value,
      size: 0.3,
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
      map: glowTex,
      alphaTest: 0.05,
    });
    const jets = new THREE.Points(jetGeo, jetMat);
    particleScene.add(jets);

    function spawnEruption(strength: number) {
      // Pick a surface point biased to face the camera.
      const towardCam = camera.position.clone().normalize();
      for (let i = 0; i < JETS; i++) {
        const u = new THREE.Vector3(
          Math.random() * 2 - 1,
          Math.random() * 2 - 1,
          Math.random() * 2 - 1,
        )
          .addScaledVector(towardCam, 1.4)
          .normalize();
        jetPos[i * 3] = u.x * (SUN_R + 0.1);
        jetPos[i * 3 + 1] = u.y * (SUN_R + 0.1);
        jetPos[i * 3 + 2] = u.z * (SUN_R + 0.1);
        jetVel[i * 3] = u.x * (3 + Math.random() * 4) * strength;
        jetVel[i * 3 + 1] = u.y * (3 + Math.random() * 4) * strength;
        jetVel[i * 3 + 2] = u.z * (3 + Math.random() * 4) * strength;
        jetLife[i] = 0.001;
        jetMax[i] = 0.7 + Math.random() * 0.7;
      }
    }
    const flashTex = new THREE.CanvasTexture(makeFlashTexture());
    disposables.push(flashTex);
    const flashMat = new THREE.SpriteMaterial({
      map: flashTex,
      color: 0xfff3e0,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const flash = new THREE.Sprite(flashMat);
    flash.scale.setScalar(46);
    particleScene.add(flash);
    let flashLevel = 0;

    function updateJets(dt: number) {
      const pos = jets.geometry.getAttribute("position") as THREE.BufferAttribute;
      let anyAlive = false;
      for (let i = 0; i < JETS; i++) {
        if (jetLife[i] <= 0) continue;
        anyAlive = true;
        jetLife[i] += dt;
        const p = jetLife[i] / jetMax[i];
        if (p >= 1) {
          jetLife[i] = 0;
          pos.setXYZ(i, 0, -999, 0);
          continue;
        }
        const decay = Math.pow(1 - p, 0.5);
        jetPos[i * 3] += jetVel[i * 3] * dt * decay;
        jetPos[i * 3 + 1] += jetVel[i * 3 + 1] * dt * decay + dt * 1.2;
        jetPos[i * 3 + 2] += jetVel[i * 3 + 2] * dt * decay;
      }
      pos.needsUpdate = true;
      const targetOpacity = anyAlive ? 0.9 : 0;
      jetMat.opacity += (targetOpacity - jetMat.opacity) * Math.min(1, dt * 6);
      flashLevel = Math.max(0, flashLevel - dt * 2.4);
      flashMat.opacity = flashLevel;
    }

    // ── Interaction state ─────────────────────────────────────────────────
    const orbit = {
      azimuth: 0,
      polar: 1.35,
      radius: 14,
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
      orbit.azimuth -= (e.clientX - lastPX) * 0.005;
      orbit.polar = Math.max(
        0.2,
        Math.min(1.5, orbit.polar - (e.clientY - lastPY) * 0.005),
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
      emitCosmosEvent({ type: "solar-eruption", heat: 0.65, hue: 24, x: ndcX, y: ndcY });
      spawnEruption(1);
      flashLevel = 1;
    };
    const onWheel = (e: WheelEvent) => {
      if (!e.shiftKey) return;
      e.preventDefault();
      orbit.radius = Math.max(6, Math.min(30, orbit.radius * (1 + e.deltaY * 0.001)));
    };
    const onContextMenu = (e: Event) => e.preventDefault();
    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      camU.uAspect.value = camera.aspect;
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("wheel", onWheel, { passive: false });
    window.addEventListener("contextmenu", onContextMenu);
    window.addEventListener("resize", onResize);

    // ── Ambient prominences ───────────────────────────────────────────────
    let ambientTimer = 9 + Math.random() * 6;
    function updateAmbient(dt: number) {
      ambientTimer -= dt;
      if (ambientTimer > 0) return;
      ambientTimer = 10 + Math.random() * 6;
      spawnEruption(0.45);
      flashLevel = Math.max(flashLevel, 0.35);
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

      if (!reduced) orbit.azimuth += dt * 0.015;

      const pa = orbit.polar;
      const aa = orbit.azimuth;
      const r = orbit.radius;
      camera.position.set(
        orbit.target.x + r * Math.sin(pa) * Math.sin(aa) + mouse.x * 1.2,
        orbit.target.y + r * Math.cos(pa) + -mouse.y * 0.6,
        orbit.target.z + r * Math.sin(pa) * Math.cos(aa),
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

      updateProms(dt);
      updateWind(dt);
      updateJets(dt);
      if (!reduced) updateAmbient(dt);

      // Pass order: stars → photosphere (writes depth) → corona (additive,
      // no depth test) → particles (additive, depth-test against the disk).
      renderer.clear();
      renderer.render(scene, camera);
      renderer.render(diskScene, postCamera);
      renderer.render(coronaScene, postCamera);
      renderer.render(particleScene, camera);
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
      [scene, particleScene, diskScene, coronaScene].forEach((s) => {
        s.traverse((obj) => {
          const mesh = obj as THREE.Mesh;
          if (mesh.geometry) mesh.geometry.dispose();
          const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
          if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
          else if (mat) mat.dispose();
        });
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
