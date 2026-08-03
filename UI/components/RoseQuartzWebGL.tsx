"use client";

// ── Rose Quartz — the crystal-field world for "rose-quartz" ────────────────
//
// Self-contained WebGL background modeled on AuroraWebGL.tsx's skeleton. Scene:
// a field of rose-quartz points standing on a glossy dark slab, studio-lit like
// a museum specimen. Each crystal is a real hexagonal prism terminated in a
// hexagonal pyramid point — flat-faceted (non-indexed geometry + computed
// normals) so it reads as genuine quartz — rendered with MeshPhysicalMaterial
// transmission (ior ≈ 1.544, real refraction of the slab + backdrop behind it),
// clearcoat for the vitreous luster, and a procedural transmissionMap /
// thicknessMap pair that frosts the interior with milky veins. A PMREM studio
// dome (violet→rose gradient sky + emissive "softboxes") supplies the
// reflections and refractions; crystal clusters cast soft shadows on the slab;
// a sparkle Points field twinkles at the tips. Click → a rose pulse ring on the
// slab plus an emissive surge, broadcast to the UI over the cosmosEvents bus.
// Colors read live from --palette-* so the PaletteEditor retunes the scene in
// real time.

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { supportsWebGL } from "./webglSupport";
import { readPaletteColors, makePalettePoller, type WorldPalette } from "./worldUtils";
import { emitCosmosEvent } from "./cosmosEvents";

// ── Crystal-field tuning ───────────────────────────────────────────────────
// Kept at the top of scope so they can be tuned in place without touching the
// shader strings (same pattern as OceanWebGL's wave constants).
const CRYSTAL_IOR = 1.544; // quartz
const FIELD_RADIUS = 3.6; // field half-width on the slab
const MAIN_COUNT = 24; // full crystal points
const CHIPPED_COUNT = 7; // partial / broken points
const DRUSY_COUNT = 320; // tiny "drusy" points clustered at the bases
const MIN_H = 0.9; // crystal height scale, min
const MAX_H = 2.2; // crystal height scale, max
const PULSE_EMISSIVE = 0.35; // click-pulse emissive surge
const RING_MAX = 3.4; // click ring max radius
const SPARKLE_COUNT = 70; // twinkling tip glints

// The shared crystal geometry's dimensions (world units at scale 1).
const GEOM_BASE_R = 0.55;
const GEOM_TIP_R = 0.42;
const GEOM_H = 1.5;
const GEOM_TIPH = 0.75;
const GEOM_TOTAL = GEOM_H + GEOM_TIPH; // 2.25 — apex height at unit scale

// ── Sparkle shaders (crystal-tip glints) ────────────────────────────────────
const SPARK_VERT = `
attribute float aPhase;
attribute float aSize;
varying float vTw;
uniform float uTime;
void main() {
  vTw = 0.55 + 0.45 * sin(uTime * 1.4 + aPhase);
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = clamp(aSize * vTw * (150.0 / -mv.z), 1.0, 64.0);
  gl_Position = projectionMatrix * mv;
}
`;
const SPARK_FRAG = `
varying float vTw;
uniform vec3 uColor;
uniform float uPulse;
void main() {
  vec2 p = gl_PointCoord - 0.5;
  float d = length(p);
  float halo = max(0.0, 1.0 - d * 2.6);
  float star = pow(max(0.0, 1.0 - d * 4.0), 3.0);
  float cross = pow(max(0.0, 1.0 - abs(p.x) * 6.0), 3.0)
              * pow(max(0.0, 1.0 - abs(p.y) * 6.0), 3.0);
  float a = (halo * halo * 0.55 + star * 0.5 + cross * 0.8)
          * (0.55 + 0.45 * vTw)
          * (1.0 + uPulse * 2.0);
  gl_FragColor = vec4(uColor, a);
}
`;

// ── Crystal geometry builders ───────────────────────────────────────────────
// A hexagonal prism with a hexagonal-pyramid tip, built NON-indexed so
// computeVertexNormals() gives every triangle its own flat facet normal — the
// crisp-faceted quartz read. The base is capped so transmission doesn't see
// through the bottom, and UVs are cylindrical (u = angle around, v = height) to
// drive the milky-vein maps coherently across the facets.

type V3 = [number, number, number];

function hexVerts(r: number, y: number): V3[] {
  const out: V3[] = [];
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    out.push([Math.cos(a) * r, y, Math.sin(a) * r]);
  }
  return out;
}

function buildCrystalGeometry(
  baseR: number,
  tipR: number,
  height: number,
  tipH: number,
): THREE.BufferGeometry {
  const bottom = hexVerts(baseR, 0);
  const top = hexVerts(tipR, height);
  const apex: V3 = [0, height + tipH, 0];
  const center: V3 = [0, 0, 0];
  const positions: number[] = [];
  const uvs: number[] = [];
  const totalH = height + tipH;
  const pushTri = (a: V3, b: V3, c: V3) => {
    for (const p of [a, b, c]) {
      positions.push(p[0], p[1], p[2]);
      const u = Math.atan2(p[2], p[0]) / (Math.PI * 2) + 0.5;
      const v = p[1] / totalH;
      uvs.push(u, v);
    }
  };
  for (let i = 0; i < 6; i++) {
    const j = (i + 1) % 6;
    pushTri(bottom[i], bottom[j], top[j]); // side quad (1)
    pushTri(bottom[i], top[j], top[i]); // side quad (2)
    pushTri(top[i], top[j], apex); // pyramid point
    pushTri(center, bottom[j], bottom[i]); // base cap
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geo.computeVertexNormals();
  return geo;
}

// A prism with no point: the top hexagon is sheared onto a slanted plane and
// capped with a fan — reads as a chipped / broken crystal.
function buildChippedCrystalGeometry(
  baseR: number,
  tipR: number,
  height: number,
): THREE.BufferGeometry {
  const bottom = hexVerts(baseR, 0);
  const top = hexVerts(tipR, height);
  for (let i = 0; i < 6; i++) {
    top[i][1] += (i / 5 - 0.5) * 0.8;
  }
  const cx = top.reduce((s, p) => s + p[0], 0) / 6;
  const cy = top.reduce((s, p) => s + p[1], 0) / 6;
  const cz = top.reduce((s, p) => s + p[2], 0) / 6;
  const topCenter: V3 = [cx, cy, cz];
  const center: V3 = [0, 0, 0];
  const positions: number[] = [];
  const uvs: number[] = [];
  const pushTri = (a: V3, b: V3, c: V3) => {
    for (const p of [a, b, c]) {
      positions.push(p[0], p[1], p[2]);
      const u = Math.atan2(p[2], p[0]) / (Math.PI * 2) + 0.5;
      const v = Math.min(1, p[1] / height);
      uvs.push(u, v);
    }
  };
  for (let i = 0; i < 6; i++) {
    const j = (i + 1) % 6;
    pushTri(bottom[i], bottom[j], top[j]);
    pushTri(bottom[i], top[j], top[i]);
    pushTri(top[i], top[j], topCenter);
    pushTri(center, bottom[j], bottom[i]);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geo.computeVertexNormals();
  return geo;
}

// ── Procedural textures ─────────────────────────────────────────────────────
function hash2(x: number, y: number): number {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return s - Math.floor(s);
}
function vnoise(x: number, y: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;
  const u = xf * xf * (3 - 2 * xf);
  const v = yf * yf * (3 - 2 * yf);
  const a = hash2(xi, yi);
  const b = hash2(xi + 1, yi);
  const c = hash2(xi, yi + 1);
  const d = hash2(xi + 1, yi + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

// Two aligned 256² canvases from the SAME milky-blob layout:
//  - trans (transmissionMap): white base, dark-gray blobs → blobs read frosted.
//  - thick (thicknessMap):    black base, white blobs     → blobs read thicker
//    (deeper pink interior via attenuation).
function makeVeinTextures(): {
  trans: THREE.CanvasTexture;
  thick: THREE.CanvasTexture;
} {
  const S = 256;
  const blobs: { x: number; y: number; r: number; a: number }[] = [];
  for (let i = 0; i < 14; i++) {
    blobs.push({
      x: Math.random() * S,
      y: Math.random() * S,
      r: 24 + Math.random() * 60,
      a: 0.22 + Math.random() * 0.3,
    });
  }
  const make = (darkBase: boolean) => {
    const c = document.createElement("canvas");
    c.width = S;
    c.height = S;
    const ctx = c.getContext("2d")!;
    ctx.fillStyle = darkBase ? "#000" : "#fff";
    ctx.fillRect(0, 0, S, S);
    for (const b of blobs) {
      const g = ctx.createRadialGradient(b.x, b.y, 0, b.x, b.y, b.r);
      g.addColorStop(
        0,
        darkBase ? `rgba(255,255,255,${b.a})` : `rgba(58,58,64,${b.a})`,
      );
      g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, S, S);
    }
    // Fine frost speckle (3-octave value noise) so the veins aren't flat discs.
    const img = ctx.getImageData(0, 0, S, S);
    const d = img.data;
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const i = (y * S + x) * 4;
        let n = 0;
        let amp = 1;
        let f = 6;
        for (let o = 0; o < 3; o++) {
          n += amp * vnoise(x / f, y / f);
          amp *= 0.5;
          f *= 2.1;
        }
        const g = d[i + 1] * (0.86 + 0.28 * n);
        d[i] = g;
        d[i + 1] = g;
        d[i + 2] = g;
      }
    }
    ctx.putImageData(img, 0, 0);
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = THREE.RepeatWrapping;
    return tex;
  };
  return { trans: make(false), thick: make(true) };
}

// ── Backdrop canvas (scene.background AND the env-dome sky share this) ──────
function makeBackdropCanvas(): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = 512;
  c.height = 512;
  return c;
}
function redrawBackdrop(canvas: HTMLCanvasElement, p: WorldPalette) {
  const ctx = canvas.getContext("2d")!;
  const H = canvas.height;
  const rgb = (col: THREE.Color, a = 1) =>
    `rgba(${Math.round(col.r * 255)},${Math.round(col.g * 255)},${Math.round(
      col.b * 255,
    )},${a})`;
  const g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, rgb(p.tertiary)); // violet sky overhead
  g.addColorStop(0.42, rgb(p.tertiary, 0.85));
  g.addColorStop(0.55, rgb(p.primary)); // rose horizon
  g.addColorStop(0.75, rgb(p.primary, 0.5));
  g.addColorStop(1, "rgba(8,4,10,1)"); // near-black below the slab
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, canvas.width, H);
}

export function RoseQuartzWebGL({ className = "" }: { className?: string }) {
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
    renderer.setClearColor(0x160b16, 1);
    renderer.shadowMap.enabled = true;
    // In three 0.185 PCFSoftShadowMap was folded into PCFShadowMap (soft PCF is
    // now the default); use the current constant to avoid the deprecation.
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    host.appendChild(renderer.domElement);
    renderer.domElement.style.position = "fixed";
    renderer.domElement.style.inset = "0";
    renderer.domElement.style.display = "block";

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      50,
      window.innerWidth / window.innerHeight,
      0.1,
      200,
    );
    const clock = new THREE.Clock();
    const disposables: THREE.Texture[] = [];
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const WHITE = new THREE.Color(0xffffff);

    // ── Palette → live uniforms ───────────────────────────────────────────
    const p0 = readPaletteColors();
    const colA = { value: p0.primary.clone() }; // primary (rose)
    const colC = { value: p0.tertiary.clone() }; // tertiary (violet)
    const colD = { value: p0.accent.clone() }; // accent (bright rose)

    // ── Milky-vein textures + shared backdrop ─────────────────────────────
    const veins = makeVeinTextures();
    disposables.push(veins.trans, veins.thick);
    const backdropCanvas = makeBackdropCanvas();
    redrawBackdrop(backdropCanvas, p0);
    const backdropTex = new THREE.CanvasTexture(backdropCanvas);
    backdropTex.colorSpace = THREE.SRGBColorSpace;
    disposables.push(backdropTex);
    scene.background = backdropTex;

    // ── Studio environment (PMREM) — gradient sky + emissive softboxes ───
    const pmrem = new THREE.PMREMGenerator(renderer);
    let envRT: THREE.WebGLRenderTarget | null = null;
    const buildEnv = (p: WorldPalette) => {
      if (envRT) {
        envRT.dispose();
        envRT.texture.dispose();
        envRT = null;
      }
      const envScene = new THREE.Scene();
      const dome = new THREE.Mesh(
        new THREE.SphereGeometry(40, 16, 12),
        new THREE.MeshBasicMaterial({ map: backdropTex, side: THREE.BackSide }),
      );
      envScene.add(dome);
      const boxGeo = new THREE.CircleGeometry(2.0, 24);
      const boxes: { pos: [number, number, number]; color: THREE.Color }[] = [
        { pos: [6, 8, 5], color: WHITE.clone().multiplyScalar(2.4) },
        { pos: [-6, 7, 4], color: WHITE.clone().multiplyScalar(1.7) },
        { pos: [0, 4, 9], color: p.accent.clone().multiplyScalar(2.0) },
        { pos: [5, 2, -8], color: p.tertiary.clone().multiplyScalar(1.2) },
        { pos: [-5, 1.5, -7], color: p.secondary.clone().multiplyScalar(1.0) },
      ];
      for (const b of boxes) {
        const m = new THREE.Mesh(
          boxGeo,
          new THREE.MeshBasicMaterial({
            color: b.color,
            side: THREE.DoubleSide,
          }),
        );
        m.position.set(b.pos[0], b.pos[1], b.pos[2]);
        m.lookAt(0, 0, 0);
        envScene.add(m);
      }
      envRT = pmrem.fromScene(envScene, 0.04, 0.1, 1000);
      scene.environment = envRT.texture;
      scene.environmentIntensity = 0.85;
      envScene.traverse((o) => {
        const mesh = o as THREE.Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
        const mat = mesh.material as THREE.Material | undefined;
        if (mat) mat.dispose();
      });
    };

    // ── Lights + shadows (first world to use them) ────────────────────────
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.6);
    keyLight.position.set(3.2, 6, 2.5);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(1024, 1024);
    keyLight.shadow.camera.left = -7;
    keyLight.shadow.camera.right = 7;
    keyLight.shadow.camera.top = 7;
    keyLight.shadow.camera.bottom = -7;
    keyLight.shadow.camera.near = 0.5;
    keyLight.shadow.camera.far = 20;
    keyLight.shadow.bias = -0.0004;
    keyLight.shadow.normalBias = 0.02;
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(colD.value, 1.2);
    rimLight.position.set(-4, 3.5, -5);
    scene.add(rimLight);

    const hemiLight = new THREE.HemisphereLight(colC.value, new THREE.Color(0x1a0f1c), 0.7);
    scene.add(hemiLight);

    const fillLight = new THREE.PointLight(colA.value, 10, 14, 2);
    fillLight.position.set(0, 0.6, 2.5);
    scene.add(fillLight);

    // ── Glossy dark slab ──────────────────────────────────────────────────
    const slabMat = new THREE.MeshStandardMaterial({
      color: 0x0d0710,
      metalness: 1.0,
      roughness: 0.16,
      envMapIntensity: 1.2,
    });
    const slab = new THREE.Mesh(new THREE.CircleGeometry(20, 48), slabMat);
    slab.rotation.x = -Math.PI / 2;
    slab.receiveShadow = true;
    scene.add(slab);

    // ── Crystal materials ─────────────────────────────────────────────────
    const crystalMat = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(0xf4c7d9),
      metalness: 0,
      roughness: 0.12,
      transmission: 0.92,
      transmissionMap: veins.trans,
      thickness: 1.6,
      thicknessMap: veins.thick,
      ior: CRYSTAL_IOR,
      clearcoat: 1.0,
      clearcoatRoughness: 0.12,
      attenuationColor: new THREE.Color(0xd65b8a),
      attenuationDistance: 4.2,
      sheen: 0.35,
      sheenColor: new THREE.Color(0xffd9e6),
      emissive: new THREE.Color(0xf28abf),
      emissiveIntensity: 0,
      envMapIntensity: 0.9,
      shadowSide: THREE.BackSide,
    });
    const drusyMat = new THREE.MeshStandardMaterial({
      color: 0xf5c7da,
      roughness: 0.35,
      metalness: 0,
      flatShading: true,
      envMapIntensity: 0.6,
    });

    // ── Crystal field (instanced) ─────────────────────────────────────────
    const crystalGeo = buildCrystalGeometry(GEOM_BASE_R, GEOM_TIP_R, GEOM_H, GEOM_TIPH);
    const chippedGeo = buildChippedCrystalGeometry(GEOM_BASE_R, GEOM_TIP_R, GEOM_H);
    const dummy = new THREE.Object3D();

    const mainPlacements: { x: number; z: number; h: number }[] = [];
    const mainMesh = new THREE.InstancedMesh(crystalGeo, crystalMat, MAIN_COUNT);
    mainMesh.castShadow = true;
    // Instances spread far beyond the source geometry's bounding sphere, which
    // three.js uses for frustum culling — disable it so edge crystals never pop.
    mainMesh.frustumCulled = false;
    for (let i = 0; i < MAIN_COUNT; i++) {
      const r = Math.pow(Math.random(), 0.65) * FIELD_RADIUS;
      const a = Math.random() * Math.PI * 2;
      const x = Math.cos(a) * r;
      const z = Math.sin(a) * r;
      const hScale = MIN_H + Math.random() * (MAX_H - MIN_H);
      const wScale = 0.7 + Math.random() * 0.5;
      dummy.position.set(x, 0, z);
      dummy.rotation.set(
        (Math.random() - 0.5) * 0.8,
        Math.random() * Math.PI * 2,
        (Math.random() - 0.5) * 0.8,
      );
      dummy.scale.set(wScale, hScale, wScale);
      dummy.updateMatrix();
      mainMesh.setMatrixAt(i, dummy.matrix);
      mainMesh.setColorAt(
        i,
        new THREE.Color().setHSL(
          0.92 + Math.random() * 0.05,
          0.2 + Math.random() * 0.35,
          0.55 + Math.random() * 0.25,
        ),
      );
      mainPlacements.push({ x, z, h: hScale });
    }
    mainMesh.instanceMatrix.needsUpdate = true;
    if (mainMesh.instanceColor) mainMesh.instanceColor.needsUpdate = true;
    scene.add(mainMesh);

    const chippedMesh = new THREE.InstancedMesh(chippedGeo, crystalMat, CHIPPED_COUNT);
    chippedMesh.castShadow = true;
    chippedMesh.frustumCulled = false;
    for (let i = 0; i < CHIPPED_COUNT; i++) {
      const r = 0.5 + Math.random() * FIELD_RADIUS * 0.8;
      const a = Math.random() * Math.PI * 2;
      dummy.position.set(Math.cos(a) * r, 0, Math.sin(a) * r);
      dummy.rotation.set(
        (Math.random() - 0.5) * 1.0,
        Math.random() * Math.PI * 2,
        (Math.random() - 0.5) * 1.0,
      );
      const s = 0.8 + Math.random() * 0.8;
      dummy.scale.set(s, s * (0.6 + Math.random() * 0.4), s);
      dummy.updateMatrix();
      chippedMesh.setMatrixAt(i, dummy.matrix);
      chippedMesh.setColorAt(
        i,
        new THREE.Color().setHSL(
          0.92 + Math.random() * 0.05,
          0.25 + Math.random() * 0.3,
          0.6 + Math.random() * 0.2,
        ),
      );
    }
    chippedMesh.instanceMatrix.needsUpdate = true;
    if (chippedMesh.instanceColor) chippedMesh.instanceColor.needsUpdate = true;
    scene.add(chippedMesh);

    const drusyMesh = new THREE.InstancedMesh(crystalGeo, drusyMat, DRUSY_COUNT);
    drusyMesh.frustumCulled = false;
    for (let i = 0; i < DRUSY_COUNT; i++) {
      const p = mainPlacements[i % mainPlacements.length];
      const ang = Math.random() * Math.PI * 2;
      const rad = 0.15 + Math.random() * 0.55;
      dummy.position.set(p.x + Math.cos(ang) * rad, 0, p.z + Math.sin(ang) * rad);
      dummy.rotation.set(
        (Math.random() - 0.5) * 1.0,
        Math.random() * Math.PI * 2,
        (Math.random() - 0.5) * 1.0,
      );
      const s = 0.05 + Math.random() * 0.12;
      dummy.scale.set(s, s * (1.2 + Math.random() * 0.8), s);
      dummy.updateMatrix();
      drusyMesh.setMatrixAt(i, dummy.matrix);
      drusyMesh.setColorAt(
        i,
        new THREE.Color().setHSL(
          0.93 + Math.random() * 0.04,
          0.3 + Math.random() * 0.3,
          0.55 + Math.random() * 0.25,
        ),
      );
    }
    drusyMesh.instanceMatrix.needsUpdate = true;
    if (drusyMesh.instanceColor) drusyMesh.instanceColor.needsUpdate = true;
    scene.add(drusyMesh);

    // ── Sparkle glints at the tips ────────────────────────────────────────
    const sparkPositions: number[] = [];
    const sparkPhases: number[] = [];
    const sparkSizes: number[] = [];
    const sparkCount = Math.min(SPARKLE_COUNT, mainPlacements.length * 3);
    for (let s = 0; s < sparkCount; s++) {
      const p = mainPlacements[Math.floor(s / 3) % mainPlacements.length];
      const frac = 0.55 + Math.random() * 0.55;
      sparkPositions.push(
        p.x + (Math.random() - 0.5) * 0.8,
        Math.max(0.05, GEOM_TOTAL * p.h * frac),
        p.z + (Math.random() - 0.5) * 0.8,
      );
      sparkPhases.push(Math.random() * Math.PI * 2);
      sparkSizes.push(1.2 + Math.random() * 2.3);
    }
    const sparkGeo = new THREE.BufferGeometry();
    sparkGeo.setAttribute("position", new THREE.Float32BufferAttribute(sparkPositions, 3));
    sparkGeo.setAttribute("aPhase", new THREE.Float32BufferAttribute(sparkPhases, 1));
    sparkGeo.setAttribute("aSize", new THREE.Float32BufferAttribute(sparkSizes, 1));
    const sparkMat = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 }, uColor: colD, uPulse: { value: 0 } },
      vertexShader: SPARK_VERT,
      fragmentShader: SPARK_FRAG,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const sparkles = new THREE.Points(sparkGeo, sparkMat);
    sparkles.renderOrder = 4;
    scene.add(sparkles);

    // ── Click pulse rings (flat, on the slab) ─────────────────────────────
    const ringGeo = new THREE.RingGeometry(0.5, 0.62, 64);
    const clickRings: { mesh: THREE.Mesh; life: number; maxLife: number }[] = [];
    function spawnRing(at: THREE.Vector3, strength: number) {
      const m = new THREE.Mesh(
        ringGeo,
        new THREE.MeshBasicMaterial({
          color: colD.value,
          transparent: true,
          opacity: 0.8 * strength,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
          side: THREE.DoubleSide,
        }),
      );
      m.rotation.x = -Math.PI / 2;
      m.position.set(at.x, 0.03, at.z);
      m.renderOrder = 3;
      scene.add(m);
      clickRings.push({ mesh: m, life: 0, maxLife: 1.2 });
    }

    // ── Palette → scene retune ────────────────────────────────────────────
    const applyColors = (p: WorldPalette) => {
      // Pale rose body (lightened primary) so the glassy tint reads bright;
      // the per-instance colors carry the variation.
      crystalMat.color.copy(p.primary).lerp(WHITE, 0.45);
      crystalMat.attenuationColor.copy(p.primary);
      crystalMat.sheenColor.copy(p.accent).lerp(WHITE, 0.4);
      crystalMat.emissive.copy(p.accent);
      drusyMat.color.copy(p.primary).lerp(WHITE, 0.35);
      rimLight.color.copy(p.accent);
      hemiLight.color.copy(p.tertiary);
      fillLight.color.copy(p.primary);
      sparkMat.uniforms.uColor.value.copy(p.accent);
      redrawBackdrop(backdropCanvas, p);
      backdropTex.needsUpdate = true;
      buildEnv(p);
    };
    const poll = makePalettePoller(400, applyColors);
    applyColors(p0);

    // ── Interaction state ─────────────────────────────────────────────────
    const orbit = {
      azimuth: 0,
      polar: 1.25,
      radius: 11,
      target: new THREE.Vector3(0, 1.7, 0),
    };
    const mouse = { x: 0, y: 0 };
    let dragging = false;
    let lastPX = 0;
    let lastPY = 0;
    let pressStart = 0;
    let downX = 0;
    let downY = 0;
    let pulse = 0;
    let emissivePulse = 0;
    const ringCenter = new THREE.Vector3(0, 0, 0);

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
      emitCosmosEvent({ type: "rose-pulse", heat: 0.55, hue: 330, x: ndcX, y: ndcY });
      // Slab-plane hit for the ring centre.
      const ndc = new THREE.Vector3(ndcX, ndcY, 0.5).unproject(camera);
      const dir = ndc.sub(camera.position).normalize();
      const tHit = (0 - camera.position.y) / dir.y;
      if (tHit > 0) {
        ringCenter.copy(camera.position).addScaledVector(dir, tHit);
      }
      pulse = 1;
      emissivePulse = 1;
      spawnRing(ringCenter, 1);
    };
    const onWheel = (e: WheelEvent) => {
      if (!e.shiftKey) return;
      e.preventDefault();
      orbit.radius = Math.max(8, Math.min(18, orbit.radius * (1 + e.deltaY * 0.001)));
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

    // ── Ambient pulses ────────────────────────────────────────────────────
    let ambientTimer = 10 + Math.random() * 6;
    function updateAmbient(dt: number) {
      ambientTimer -= dt;
      if (ambientTimer > 0) return;
      ambientTimer = 12 + Math.random() * 8;
      pulse = Math.max(pulse, 0.5);
      emissivePulse = Math.max(emissivePulse, 0.5);
      ringCenter.set(
        (Math.random() - 0.5) * 6,
        0,
        (Math.random() - 0.5) * 6,
      );
      spawnRing(ringCenter, 0.5);
      emitCosmosEvent({ type: "rose-pulse", heat: 0.18, hue: 330, x: 0, y: 0 });
    }

    // ── Animate loop ──────────────────────────────────────────────────────
    let raf = 0;
    let sceneTime = 0;
    function animate() {
      raf = requestAnimationFrame(animate);
      const dt = Math.min(clock.getDelta(), 0.05);
      sceneTime += reduced ? 0 : dt;
      const t = sceneTime;
      poll(performance.now());

      if (!reduced) {
        orbit.azimuth += Math.sin(t * 0.06) * dt * 0.02;
        pulse = Math.max(0, pulse - dt * 0.8);
        emissivePulse = Math.max(0, emissivePulse - dt * 0.9);
      }

      const pa = orbit.polar;
      const aa = orbit.azimuth;
      const r = orbit.radius;
      camera.position.set(
        orbit.target.x + r * Math.sin(pa) * Math.sin(aa) + mouse.x * 0.8,
        orbit.target.y + r * Math.cos(pa) + -mouse.y * 0.4,
        orbit.target.z + r * Math.sin(pa) * Math.cos(aa),
      );
      camera.lookAt(orbit.target);

      crystalMat.emissiveIntensity = emissivePulse * PULSE_EMISSIVE;
      sparkMat.uniforms.uTime.value = t;
      sparkMat.uniforms.uPulse.value = pulse;

      for (let i = clickRings.length - 1; i >= 0; i--) {
        const rr = clickRings[i];
        rr.life += dt;
        const p = Math.min(1, rr.life / rr.maxLife);
        rr.mesh.scale.setScalar(1 + p * RING_MAX);
        (rr.mesh.material as THREE.MeshBasicMaterial).opacity = (1 - p) * 0.8;
        if (p >= 1) {
          scene.remove(rr.mesh);
          (rr.mesh.material as THREE.Material).dispose();
          clickRings.splice(i, 1);
        }
      }

      if (!reduced) updateAmbient(dt);

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
      clickRings.forEach((rr) => {
        (rr.mesh.material as THREE.Material).dispose();
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
      if (envRT) {
        envRT.dispose();
        envRT.texture.dispose();
      }
      pmrem.dispose();
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
            "radial-gradient(ellipse at 50% 100%, hsl(var(--palette-primary) / 0.30), hsl(var(--palette-tertiary) / 0.12) 55%, transparent 75%)",
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
