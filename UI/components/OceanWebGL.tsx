"use client";

// ── Ocean Depths — the underwater world for the "ocean-depths" palette ────
//
// Self-contained WebGL background modeled on CosmosWebGL.tsx's skeleton, but
// with the water upgraded to a screen-space ray-traced ocean. A fullscreen
// pass replaces the old displaced water mesh + god-ray sprites:
//
//   1. Underwater scene → render target. The seabed gradient plane, the
//      gliding whale, sinking marine snow and bioluminescent bursts are all
//      rendered below the surface into a color-only WebGLRenderTarget.
//   2. Fullscreen ocean pass (half-res, blitted up). Per pixel it ray-marches
//      the animated Gerstner height field to find the surface point + normal,
//      then composites a fresnel-weighted mix of:
//        • sky reflection (Schlick fresnel, sun glint on the wave normals),
//        • Snell refraction (1/1.33) with a parallax sample of the underwater
//          scene and per-channel Beer–Lambert absorption (red fades fastest),
//        • volumetric god rays marched down the refracted ray — caustic focus
//          from surface curvature, slanted with the sun, shadowed by an
//          analytic sphere approximation of the whale.
//
// The splash rings draw over the ocean as a final overlay. Colors read live
// from --palette-* so the PaletteEditor retunes the sea in real time.

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { supportsWebGL } from "./webglSupport";
import { readPaletteColors, makePalettePoller } from "./worldUtils";
import { emitCosmosEvent } from "./cosmosEvents";

// ── Seabed gradient plane ──────────────────────────────────────────────────
const SEABED_VERT = `
varying vec3 vWorld;
void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorld = wp.xyz;
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;
const SEABED_FRAG = `
uniform float uTime;
uniform vec3 uColorB;
uniform vec3 uColorC;
uniform vec3 uColorD;
varying vec3 vWorld;
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 x){
  vec2 i = floor(x); vec2 f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1,0)), f.x),
             mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), f.x), f.y);
}
float fbm(vec2 p){
  float v = 0.0; float a = 0.5;
  for (int i = 0; i < 4; i++) { v += a * noise(p); p *= 2.1; a *= 0.5; }
  return v;
}
void main() {
  vec3 deep = mix(uColorB, uColorC, 0.35) * 0.65;
  vec3 col = deep;
  float ri = fbm(vWorld.xz * 0.14 + uTime * 0.02);
  col *= 0.8 + 0.35 * ri;
  // Faint drifting bioluminescent patches on the floor.
  float glow = fbm(vWorld.xz * 0.5 + vec2(uTime * 0.04, 0.0));
  col += uColorD * smoothstep(0.6, 0.92, glow) * 0.3;
  float d = length(cameraPosition - vWorld);
  col *= exp(-d * 0.02);
  gl_FragColor = vec4(col, 1.0);
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

// ── Fullscreen ocean pass: surface march + reflection/refraction/absorption ─
const OCEAN_FRAG = `
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
uniform vec3 uColorD;
uniform vec3 uWhalePos;
uniform sampler2D uUnderTex;
varying vec2 vUv;

float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 x){
  vec2 i = floor(x); vec2 f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
             mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}
float fbm(vec2 p){
  float v = 0.0; float a = 0.5;
  for (int i = 0; i < 5; i++) { v += a * noise(p); p *= 2.1; a *= 0.5; }
  return v;
}

// Gerstner swell — the same three waves the old water mesh displaced.
float waveHeight(vec2 base) {
  float h = 0.0;
  float t = uTime;
  {
    float amp = 0.8, wl = 26.0, sp = 1.25;
    vec2 dir = vec2(0.82, 0.57);
    float k = 6.283185307 / wl;
    h += amp * sin(k * dot(dir, base) + sp * t);
  }
  {
    float amp = 0.48, wl = 13.0, sp = 1.8;
    vec2 dir = vec2(-0.5, 0.87);
    float k = 6.283185307 / wl;
    h += amp * sin(k * dot(dir, base) + sp * t);
  }
  {
    float amp = 0.28, wl = 7.0, sp = 2.6;
    vec2 dir = vec2(0.3, -0.95);
    float k = 6.283185307 / wl;
    h += amp * sin(k * dot(dir, base) + sp * t);
  }
  return h;
}
vec3 surfaceNormal(vec2 base) {
  vec3 n = vec3(0.0, 1.0, 0.0);
  float t = uTime;
  {
    float amp = 0.8, wl = 26.0, sp = 1.25;
    vec2 dir = vec2(0.82, 0.57);
    float k = 6.283185307 / wl;
    float q = amp * cos(k * dot(dir, base) + sp * t);
    n.x -= dir.x * k * q; n.z -= dir.y * k * q;
  }
  {
    float amp = 0.48, wl = 13.0, sp = 1.8;
    vec2 dir = vec2(-0.5, 0.87);
    float k = 6.283185307 / wl;
    float q = amp * cos(k * dot(dir, base) + sp * t);
    n.x -= dir.x * k * q; n.z -= dir.y * k * q;
  }
  {
    float amp = 0.28, wl = 7.0, sp = 2.6;
    vec2 dir = vec2(0.3, -0.95);
    float k = 6.283185307 / wl;
    float q = amp * cos(k * dot(dir, base) + sp * t);
    n.x -= dir.x * k * q; n.z -= dir.y * k * q;
  }
  return normalize(n);
}
// Laplacian of the height field — where strongly negative the surface focuses
// light → caustics.
float waveLaplacian(vec2 base) {
  float lap = 0.0;
  float t = uTime;
  {
    float amp = 0.8, wl = 26.0, sp = 1.25;
    vec2 dir = vec2(0.82, 0.57);
    float k = 6.283185307 / wl;
    lap -= amp * k * k * sin(k * dot(dir, base) + sp * t);
  }
  {
    float amp = 0.48, wl = 13.0, sp = 1.8;
    vec2 dir = vec2(-0.5, 0.87);
    float k = 6.283185307 / wl;
    lap -= amp * k * k * sin(k * dot(dir, base) + sp * t);
  }
  {
    float amp = 0.28, wl = 7.0, sp = 2.6;
    vec2 dir = vec2(0.3, -0.95);
    float k = 6.283185307 / wl;
    lap -= amp * k * k * sin(k * dot(dir, base) + sp * t);
  }
  return lap;
}
float causticAt(vec2 base) {
  return clamp(exp(-waveLaplacian(base) * 3.2), 0.12, 2.2);
}
// Whale shadow cast on the god rays: 0 inside the sphere silhouette, 1 outside.
float whaleShadow(vec3 Q, vec3 W, float Rw, vec3 L) {
  vec3 oc = Q - W;
  float tL = dot(oc, L);
  float perp = sqrt(max(dot(oc, oc) - tL * tL, 0.0));
  float s = smoothstep(Rw * 1.15, Rw * 0.5, perp);
  return (tL > 0.0) ? s : 1.0;
}
vec3 skyColor(vec3 d, vec3 L, vec3 sunCol, vec3 horizonCol, vec3 zenithCol) {
  float h = clamp(d.y, 0.0, 1.0);
  vec3 col = mix(horizonCol, zenithCol, pow(h, 0.55));
  col += sunCol * pow(max(dot(d, L), 0.0), 10.0) * 0.14;
  col += sunCol * pow(max(dot(d, L), 0.0), 140.0) * 0.4;
  return col;
}

void main() {
  vec2 ndc = vUv * 2.0 - 1.0;
  vec3 ro = uCamPos;
  vec3 rd = normalize(uCamForward + uCamRight * ndc.x * uTanHalfFovY * uAspect + uCamUp * ndc.y * uTanHalfFovY);

  vec3 L = normalize(vec3(0.4, 0.75, -0.45));
  vec3 sunCol = mix(vec3(1.0, 0.96, 0.9), uColorD, 0.25);
  vec3 horizonCol = mix(uColorC, vec3(0.72, 0.8, 0.88), 0.3);
  vec3 zenithCol = mix(uColorB, uColorA, 0.5) * 0.3;

  // Rays above the horizon never reach the water → sky.
  vec3 col = skyColor(rd, L, sunCol, horizonCol, zenithCol);

  if (rd.y < 0.0) {
    // March down the view ray until it dips below the animated surface.
    float tMax = min((ro.y + 5.0) / (-rd.y), 280.0);
    float stepL = tMax / 28.0;
    float t = stepL * 0.5;
    bool hit = false;
    for (int i = 0; i < 28; i++) {
      vec3 Q = ro + rd * t;
      if (Q.y < waveHeight(Q.xz)) { hit = true; break; }
      t += stepL;
    }
    if (hit) {
      float tHi = t;
      float tLo = max(0.0, t - stepL);
      for (int i = 0; i < 6; i++) {
        float tm = (tLo + tHi) * 0.5;
        vec3 Q = ro + rd * tm;
        if (Q.y < waveHeight(Q.xz)) tHi = tm; else tLo = tm;
      }
      t = (tLo + tHi) * 0.5;
      vec3 P = ro + rd * t;
      vec3 N = surfaceNormal(P.xz);

      vec3 V = -rd;
      // Softer fresnel — the surface stays a mirror at grazing angles but
      // lets the refracted world show through at most viewing angles.
      float fres = 0.04 + 0.7 * pow(1.0 - max(dot(N, V), 0.0), 5.0);

      // Reflection — mostly sky at grazing angles.
      vec3 R = reflect(rd, N);
      vec3 reflecCol = skyColor(R, L, sunCol, horizonCol, zenithCol);
      vec3 H = normalize(L + V);
      reflecCol += sunCol * pow(max(dot(N, H), 0.0), 220.0) * 0.5;

      vec3 waterCol = reflecCol;
      // Snell refraction into the water (air → water, 1/1.33).
      vec3 T = refract(rd, N, 0.7519);
      if (T.y < -0.001) {
        float tDeep = min((P.y + 24.0) / (-T.y), 70.0);
        // Parallax-sample the underwater scene where the ray lands.
        vec3 seabed = P + T * tDeep;
        vec4 clip = uProj * (uView * vec4(seabed, 1.0));
        vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
        vec3 under = texture2D(uUnderTex, clamp(uv, 0.0, 1.0)).rgb;
        // Spectral absorption along the path (red is eaten fastest), but the
        // depth never crushes to black — residual light scatters to a mid-blue.
        vec3 sigma = vec3(0.11, 0.05, 0.024);
        vec3 absorb = exp(-sigma * tDeep);
        vec3 scatter = mix(uColorB, uColorA, 0.5) * 0.4;
        float absorbA = dot(absorb, vec3(0.299, 0.587, 0.114));
        waterCol = under * absorb + scatter * (1.0 - absorbA);
        // Volumetric god rays down the refracted ray, slanted with the sun.
        // The per-step factor is a small constant (not march-length-scaled) so
        // the shafts read as soft shimmer instead of blowing out to white.
        float stepD = tDeep / 18.0;
        for (int i = 0; i < 18; i++) {
          float tp = (float(i) + 0.5) * stepD;
          vec3 Q = P + T * tp;
          vec2 Sxz = Q.xz - (L.xz / L.y) * Q.y; // surface entry of this light
          float cau = causticAt(Sxz);
          float occl = whaleShadow(Q, uWhalePos, 5.0, L);
          vec3 downAtten = exp(-sigma * tp * 0.55);
          float shaft = cau * occl * exp(-tp * 0.045);
          waterCol += downAtten * sunCol * shaft * 0.02;
        }
      }
      col = mix(waterCol, reflecCol, clamp(fres * 0.8, 0.0, 1.0));
    }
  }
  // Soft exponential rolloff — compresses highlights so the water never clips
  // to pure white, while a touch of exposure lifts the dark end.
  col = 1.0 - exp(-col * 1.3);
  gl_FragColor = vec4(col, 1.0);
}
`;

// ── Blit: upscale the half-res ocean pass to the screen ────────────────────
const BLIT_FRAG = `
uniform sampler2D uOceanTex;
varying vec2 vUv;
void main() {
  gl_FragColor = vec4(texture2D(uOceanTex, vUv).rgb, 1.0);
}
`;

// Radial glow texture for bioluminescent bursts.
function makeGlowTexture(): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const ctx = c.getContext("2d")!;
  const g = ctx.createRadialGradient(64, 64, 2, 64, 64, 62);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.4, "rgba(255,255,255,0.5)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 128, 128);
  return c;
}

interface Burst {
  points: THREE.Points;
  glow: THREE.Sprite;
  pos: Float32Array;
  vel: Float32Array;
  life: number;
  maxLife: number;
  count: number;
}

export function OceanWebGL({ className = "" }: { className?: string }) {
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
    renderer.setClearColor(0x04121c, 1);
    renderer.autoClear = false; // manual per-target clearing below
    host.appendChild(renderer.domElement);
    renderer.domElement.style.position = "fixed";
    renderer.domElement.style.inset = "0";
    renderer.domElement.style.display = "block";

    const underScene = new THREE.Scene();
    const oceanScene = new THREE.Scene();
    const blitScene = new THREE.Scene();
    const overlayScene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      65,
      window.innerWidth / window.innerHeight,
      0.1,
      400,
    );
    const clock = new THREE.Clock();
    const disposables: THREE.Texture[] = [];
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    // ── Palette → live uniforms ───────────────────────────────────────────
    const colA = { value: new THREE.Color(0x0ea5e9) }; // primary (cyan)
    const colB = { value: new THREE.Color(0x3b82f6) }; // secondary (blue)
    const colC = { value: new THREE.Color(0x14b8a6) }; // tertiary (teal)
    const colD = { value: new THREE.Color(0x2dd4bf) }; // accent
    const applyColors = (p: ReturnType<typeof readPaletteColors>) => {
      colA.value.copy(p.primary);
      colB.value.copy(p.secondary);
      colC.value.copy(p.tertiary);
      colD.value.copy(p.accent);
    };
    applyColors(readPaletteColors());
    const poll = makePalettePoller(400, applyColors);

    // ── Camera basis uniforms for the ocean pass ──────────────────────────
    const camU = {
      uCamPos: { value: new THREE.Vector3() },
      uCamRight: { value: new THREE.Vector3() },
      uCamUp: { value: new THREE.Vector3() },
      uCamForward: { value: new THREE.Vector3() },
      uTanHalfFovY: { value: Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) },
      uAspect: { value: camera.aspect },
    };

    // ── Render targets: underwater (full-res), ocean pass (half-res) ──────
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let rtW = Math.floor(window.innerWidth * dpr);
    let rtH = Math.floor(window.innerHeight * dpr);
    const underRT = new THREE.WebGLRenderTarget(rtW, rtH, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
    });
    const oceanRT = new THREE.WebGLRenderTarget(Math.floor(rtW * 0.75), Math.floor(rtH * 0.75), {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
    });

    // ── Ocean fullscreen pass ─────────────────────────────────────────────
    const oceanMat = new THREE.ShaderMaterial({
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
        uColorB: colB,
        uColorC: colC,
        uColorD: colD,
        uWhalePos: { value: new THREE.Vector3(30, -9, -40) },
        uUnderTex: { value: underRT.texture },
      },
      vertexShader: QUAD_VERT,
      fragmentShader: OCEAN_FRAG,
    });
    const oceanQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), oceanMat);
    oceanQuad.frustumCulled = false;
    oceanScene.add(oceanQuad);

    // ── Blit pass ─────────────────────────────────────────────────────────
    const blitMat = new THREE.ShaderMaterial({
      uniforms: {
        uOceanTex: { value: oceanRT.texture },
      },
      vertexShader: QUAD_VERT,
      fragmentShader: BLIT_FRAG,
    });
    const blitQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), blitMat);
    blitQuad.frustumCulled = false;
    blitScene.add(blitQuad);

    const postCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    // ── Seabed ────────────────────────────────────────────────────────────
    const seabedMat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uColorB: colB,
        uColorC: colC,
        uColorD: colD,
      },
      vertexShader: SEABED_VERT,
      fragmentShader: SEABED_FRAG,
    });
    const seabed = new THREE.Mesh(new THREE.PlaneGeometry(500, 500), seabedMat);
    seabed.rotation.x = -Math.PI / 2;
    seabed.position.y = -24;
    underScene.add(seabed);

    // ── Whale silhouette (under the surface, revealed by refraction) ───────
    const whale = new THREE.Group();
    const whaleBody = new THREE.Mesh(
      new THREE.SphereGeometry(1.6, 10, 8),
      new THREE.MeshBasicMaterial({ color: 0x071019 }),
    );
    whaleBody.scale.set(1.8, 0.85, 0.6);
    const whaleTail = new THREE.Mesh(
      new THREE.ConeGeometry(0.55, 1.8, 4),
      new THREE.MeshBasicMaterial({ color: 0x071019 }),
    );
    whaleTail.position.set(0, 0, -1.7);
    whaleTail.rotation.x = Math.PI / 2;
    whaleTail.scale.set(0.6, 1, 0.18);
    whale.add(whaleBody, whaleTail);
    whale.position.set(30, -9, -40);
    whale.scale.setScalar(2.2);
    underScene.add(whale);

    // ── Marine snow (sinking below the surface) ───────────────────────────
    const SNOW = 420;
    const snowPos = new Float32Array(SNOW * 3);
    const snowSpeed = new Float32Array(SNOW);
    for (let i = 0; i < SNOW; i++) {
      snowPos[i * 3] = (Math.random() - 0.5) * 130;
      snowPos[i * 3 + 1] = -1.5 - Math.random() * 20;
      snowPos[i * 3 + 2] = 40 - Math.random() * 130;
      snowSpeed[i] = 0.4 + Math.random() * 1.1;
    }
    const snowGeo = new THREE.BufferGeometry();
    snowGeo.setAttribute("position", new THREE.BufferAttribute(snowPos, 3));
    const snowMat = new THREE.PointsMaterial({
      color: 0xcfe8ff,
      size: 0.26,
      transparent: true,
      opacity: 0.5,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });
    const snow = new THREE.Points(snowGeo, snowMat);
    underScene.add(snow);

    // ── Bioluminescent bursts (underwater) ────────────────────────────────
    const glowTex = new THREE.CanvasTexture(makeGlowTexture());
    disposables.push(glowTex);
    const bursts: Burst[] = [];
    function spawnBurst(at: THREE.Vector3, count: number, strength: number) {
      const pos = new Float32Array(count * 3);
      const vel = new Float32Array(count * 3);
      for (let i = 0; i < count; i++) {
        pos[i * 3] = at.x;
        pos[i * 3 + 1] = at.y;
        pos[i * 3 + 2] = at.z;
        const th = Math.random() * Math.PI * 2;
        const ph = Math.acos(2 * Math.random() - 1);
        const sp = (1.5 + Math.random() * 2.5) * strength;
        vel[i * 3] = Math.sin(ph) * Math.cos(th) * sp;
        vel[i * 3 + 1] = Math.cos(ph) * sp + 0.8;
        vel[i * 3 + 2] = Math.sin(ph) * Math.sin(th) * sp;
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      const mat = new THREE.PointsMaterial({
        color: 0xd8fbff,
        size: 0.45,
        transparent: true,
        opacity: 0.95,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        sizeAttenuation: true,
      });
      const points = new THREE.Points(geo, mat);
      underScene.add(points);
      const glow = new THREE.Sprite(
        new THREE.SpriteMaterial({
          map: glowTex,
          color: colD.value,
          transparent: true,
          opacity: 0.8,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
        }),
      );
      glow.scale.setScalar(9 * strength);
      glow.position.copy(at);
      underScene.add(glow);
      bursts.push({ points, glow, pos, vel, life: 0, maxLife: 1.2, count });
    }

    // ── Splash rings — surface overlay drawn after the ocean pass ─────────
    const splashGeo = new THREE.RingGeometry(0.5, 0.7, 48);
    const splashes: { mesh: THREE.Mesh; life: number; maxLife: number }[] = [];
    function spawnSplash(at: THREE.Vector3, strength: number) {
      const m = new THREE.Mesh(
        splashGeo,
        new THREE.MeshBasicMaterial({
          color: 0xc9f4ff,
          transparent: true,
          opacity: 0.95 * strength,
          depthWrite: false,
          depthTest: false,
          blending: THREE.AdditiveBlending,
          side: THREE.DoubleSide,
        }),
      );
      m.rotation.x = -Math.PI / 2;
      m.position.set(at.x, 0.12, at.z);
      overlayScene.add(m);
      splashes.push({ mesh: m, life: 0, maxLife: 1.0 });
    }

    // ── Interaction state ─────────────────────────────────────────────────
    const orbit = {
      azimuth: 0,
      polar: 1.05,
      radius: 28,
      target: new THREE.Vector3(0, -6, -16),
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
      orbit.azimuth -= (e.clientX - lastPX) * 0.004;
      orbit.polar = Math.max(
        0.95,
        Math.min(1.28, orbit.polar - (e.clientY - lastPY) * 0.004),
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
      emitCosmosEvent({ type: "ocean-surge", heat: 0.55, hue: 189, x: ndcX, y: ndcY });
      // Ray against the y=0 surface plane, then push the burst just under it.
      const ndc = new THREE.Vector3(ndcX, ndcY, 0.5).unproject(camera);
      const dir = ndc.sub(camera.position).normalize();
      const tHit = -camera.position.y / dir.y;
      const hit = camera.position.clone().addScaledVector(dir, Math.max(tHit, 0));
      spawnBurst(hit.clone().setY(Math.min(hit.y, 0) - 2.5), 130, 1.6);
      spawnSplash(hit, 1);
    };
    const onWheel = (e: WheelEvent) => {
      if (!e.shiftKey) return;
      e.preventDefault();
      orbit.radius = Math.max(20, Math.min(50, orbit.radius * (1 + e.deltaY * 0.001)));
    };
    const onContextMenu = (e: Event) => e.preventDefault();
    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      camU.uAspect.value = camera.aspect;
      rtW = Math.floor(window.innerWidth * dpr);
      rtH = Math.floor(window.innerHeight * dpr);
      underRT.setSize(rtW, rtH);
      oceanRT.setSize(Math.floor(rtW * 0.75), Math.floor(rtH * 0.75));
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("wheel", onWheel, { passive: false });
    window.addEventListener("contextmenu", onContextMenu);
    window.addEventListener("resize", onResize);

    // ── Ambient plankton glows ────────────────────────────────────────────
    let ambientTimer = 8 + Math.random() * 5;
    function updateAmbient(dt: number) {
      ambientTimer -= dt;
      if (ambientTimer > 0) return;
      ambientTimer = 9 + Math.random() * 6;
      spawnBurst(
        new THREE.Vector3((Math.random() - 0.5) * 60, -6, -30 - Math.random() * 30),
        26,
        0.6,
      );
      emitCosmosEvent({ type: "ocean-surge", heat: 0.2, hue: 166, x: 0, y: 0 });
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

      if (!reduced) orbit.azimuth += Math.sin(t * 0.05) * dt * 0.015;

      const pa = orbit.polar;
      const aa = orbit.azimuth;
      const r = orbit.radius;
      camera.position.set(
        orbit.target.x + r * Math.sin(pa) * Math.sin(aa) + mouse.x * 1.0,
        orbit.target.y + r * Math.cos(pa) + -mouse.y * 0.5,
        orbit.target.z + r * Math.sin(pa) * Math.cos(aa),
      );
      if (!reduced) camera.position.y += Math.sin(t * 0.4) * 0.15;
      camera.lookAt(orbit.target);
      // Gentle roll about the view axis — must come after lookAt (it resets
      // orientation from the target matrix).
      if (!reduced) camera.rotateZ(Math.sin(t * 0.3) * 0.012);
      camera.updateMatrixWorld(true);
      camera.matrixWorldInverse.copy(camera.matrixWorld).invert();

      // Rebuild the camera ray basis for the ocean pass.
      camera.matrixWorld.extractBasis(_right, _up, _fwd);
      _fwd.negate();
      camU.uCamPos.value.copy(camera.position);
      camU.uCamRight.value.copy(_right);
      camU.uCamUp.value.copy(_up);
      camU.uCamForward.value.copy(_fwd);

      oceanMat.uniforms.uTime.value = t;
      seabedMat.uniforms.uTime.value = t;

      // Whale glides across the underwater scene.
      if (!reduced) {
        whale.position.x -= dt * 2.0;
        whale.position.z += dt * 0.6;
        whale.rotation.y = 0.35;
        if (whale.position.x < -60) whale.position.set(32, -8, -35);
      }
      oceanMat.uniforms.uWhalePos.value.copy(whale.position);

      // Marine snow sinks and wraps.
      const sp = snow.geometry.getAttribute("position") as THREE.BufferAttribute;
      for (let i = 0; i < SNOW; i++) {
        let y = sp.getY(i) - snowSpeed[i] * dt;
        if (y < -22) y = -1.5;
        sp.setY(i, y);
      }
      sp.needsUpdate = true;

      // Bursts expand + fade.
      for (let i = bursts.length - 1; i >= 0; i--) {
        const b = bursts[i];
        b.life += dt;
        const p = Math.min(1, b.life / b.maxLife);
        const attr = b.points.geometry.getAttribute("position") as THREE.BufferAttribute;
        for (let j = 0; j < b.count; j++) {
          b.pos[j * 3] += b.vel[j * 3] * dt;
          b.pos[j * 3 + 1] += b.vel[j * 3 + 1] * dt;
          b.pos[j * 3 + 2] += b.vel[j * 3 + 2] * dt;
          attr.setXYZ(j, b.pos[j * 3], b.pos[j * 3 + 1], b.pos[j * 3 + 2]);
        }
        attr.needsUpdate = true;
        (b.points.material as THREE.PointsMaterial).opacity = (1 - p) * 0.9;
        (b.glow.material as THREE.SpriteMaterial).opacity = (1 - p) * 0.8;
        b.glow.scale.setScalar((1 + p * 2) * (b.count > 40 ? 5 : 3));
        if (p >= 1) {
          underScene.remove(b.points);
          underScene.remove(b.glow);
          b.points.geometry.dispose();
          (b.points.material as THREE.Material).dispose();
          (b.glow.material as THREE.Material).dispose();
          bursts.splice(i, 1);
        }
      }

      // Splash rings expand and fade on the surface.
      for (let i = splashes.length - 1; i >= 0; i--) {
        const s = splashes[i];
        s.life += dt;
        const p = Math.min(1, s.life / s.maxLife);
        s.mesh.scale.setScalar(1 + p * 26);
        (s.mesh.material as THREE.MeshBasicMaterial).opacity = (1 - p) * 0.95;
        if (p >= 1) {
          overlayScene.remove(s.mesh);
          (s.mesh.material as THREE.Material).dispose();
          splashes.splice(i, 1);
        }
      }

      if (!reduced) updateAmbient(dt);

      // Pipeline: underwater scene → rt, ocean ray-march → half-res rt,
      // blit to screen, then surface overlays.
      renderer.setRenderTarget(underRT);
      renderer.clear();
      renderer.render(underScene, camera);

      renderer.setRenderTarget(oceanRT);
      renderer.clear();
      renderer.render(oceanScene, postCamera);

      renderer.setRenderTarget(null);
      renderer.clear();
      renderer.render(blitScene, postCamera);

      renderer.render(overlayScene, camera);
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
      splashes.forEach((s) => {
        (s.mesh.material as THREE.Material).dispose();
      });
      splashGeo.dispose();
      bursts.forEach((b) => {
        b.points.geometry.dispose();
        (b.points.material as THREE.Material).dispose();
        (b.glow.material as THREE.Material).dispose();
      });
      [underScene, oceanScene, blitScene, overlayScene].forEach((s) => {
        s.traverse((obj) => {
          const mesh = obj as THREE.Mesh;
          if (mesh.geometry) mesh.geometry.dispose();
          const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
          if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
          else if (mat) mat.dispose();
        });
      });
      underRT.dispose();
      oceanRT.dispose();
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
            "radial-gradient(ellipse at 50% 100%, hsl(var(--palette-primary) / 0.35), hsl(var(--palette-secondary) / 0.1) 55%, transparent 75%)",
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
