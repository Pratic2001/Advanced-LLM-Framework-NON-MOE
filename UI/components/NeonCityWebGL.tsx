"use client";

// ── Neon City — the synthwave grid-city world for the "neon-cyber" palette ─
//
// A self-contained WebGL background modeled on CosmosWebGL.tsx's skeleton:
// probe WebGL, mount a renderer into a host div, own the rAF loop, dispose
// everything (incl. forceContextLoss) on unmount, respect reduced-motion,
// and emit a signature event on guarded click so PaletteEventSync tints the
// chrome.
//
// Scene: a glossy wet street under a glowing perspective grid that scrolls
// toward the camera, a striped retro-synthwave sun on the horizon, dark tower
// silhouettes flanking a central avenue, neon pylons, floating wireframe
// holograms, rising data motes, and click-spawned shockwave rings. Colors are
// read live from the palette's --palette-* CSS vars (no fixed colors) so the
// in-app PaletteEditor retunes the city in real time.
//
// Ray tracing: the street is a real planar mirror. Each frame the city is
// rendered a second time into a color render target from a camera reflected
// across the y=0 street plane (the standard three.js planar-mirror technique,
// plumbed like OceanWebGL.tsx's render-target pipeline). The floor mesh then
// samples that target — every reflected ray traced exactly — with a fresnel
// + wet-street falloff, a soft glossy blur, and the neon grid composited on
// top so the scene reads as a wet synthwave boulevard rather than a flat
// printed grid.

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { supportsWebGL } from "./webglSupport";
import { readPaletteColors, makePalettePoller } from "./worldUtils";
import { emitCosmosEvent } from "./cosmosEvents";

// A click-spawned shockwave ring expanding across the grid floor.
interface Ring {
  mesh: THREE.Mesh;
  life: number;
  maxLife: number;
}

// ── Glossy mirror-floor tuning ───────────────────────────────────────────
const MIRROR_RT_SCALE = 0.75;   // reflection render-target resolution factor
const FLOOR_FRESNEL_POW = 3.5;  // higher = reflections hug the horizon tighter
const FLOOR_WET_GROWTH = 0.14;  // per-unit reflection growth with distance
const FLOOR_GLOSS_TEXELS = 2.5; // glossy blur radius (in reflection texels)
const FLOOR_GRID_ALPHA = 0.9;   // neon grid brightness over the mirror
const TOWER_RIM_POW = 3.0;      // tower "glass" rim falloff
const TOWER_RIM_BOOST = 0.85;   // tower rim brightness

// ── Glossy mirror street shader ──────────────────────────────────────────
const FLOOR_VERT = `
varying vec3 vWorld;
void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorld = wp.xyz;
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

const FLOOR_FRAG = `
#define FLOOR_GLOSS ${FLOOR_GLOSS_TEXELS.toFixed(2)}
#define FLOOR_FRESNEL ${FLOOR_FRESNEL_POW.toFixed(2)}
#define FLOOR_WET ${FLOOR_WET_GROWTH.toFixed(3)}
#define FLOOR_GRID ${FLOOR_GRID_ALPHA.toFixed(2)}
uniform float uTime;
uniform vec3 uCamPos;
uniform mat4 uProj;
uniform mat4 uMirrorView;
uniform sampler2D uCityTex;
uniform vec2 uTexelSize;
uniform vec3 uColorA;
uniform vec3 uColorB;
uniform vec3 uColorC;
varying vec3 vWorld;

float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 x){
  vec2 i = floor(x); vec2 f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
             mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}

// Sample the reflected city. Instead of a symmetric 4-tap gloss, smear the
// tap primarily along the street's receding (vertical-in-reflection) axis —
// the UV axis that maps to "away from the camera" on the ground plane — so
// bright horizontal neon bands of the towers draw as tall wet-streak
// reflections, the classic neon-on-rain asphalt look.
vec3 sampleReflection(vec2 uv) {
  float o = FLOOR_GLOSS * uTexelSize.y;
  float w1 = 1.0, w2 = 0.82, w3 = 0.55, w4 = 0.3;
  vec3 c  = texture2D(uCityTex, clamp(uv, 0.002, 0.998)).rgb * w1;
  // Vertical smear (keep symmetric so both up/down contribute to the streak,
  // but weight the far taps so streaks read as a continuous column).
  c += texture2D(uCityTex, clamp(uv + vec2(0.0, o * 1.0), 0.002, 0.998)).rgb * w2;
  c += texture2D(uCityTex, clamp(uv - vec2(0.0, o * 1.0), 0.002, 0.998)).rgb * w2;
  c += texture2D(uCityTex, clamp(uv + vec2(0.0, o * 2.4), 0.002, 0.998)).rgb * w3;
  c += texture2D(uCityTex, clamp(uv - vec2(0.0, o * 2.4), 0.002, 0.998)).rgb * w3;
  c += texture2D(uCityTex, clamp(uv + vec2(0.0, o * 4.2), 0.002, 0.998)).rgb * w4;
  c += texture2D(uCityTex, clamp(uv - vec2(0.0, o * 4.2), 0.002, 0.998)).rgb * w4;
  // A weaker horizontal tap keeps a faint cross-hatch so it still reads as a
  // plane, not vertical-only.
  c += texture2D(uCityTex, clamp(uv + vec2(o * 1.4, 0.0), 0.002, 0.998)).rgb * 0.4;
  c += texture2D(uCityTex, clamp(uv - vec2(o * 1.4, 0.0), 0.002, 0.998)).rgb * 0.4;
  float w = w1 + w2 * 2.0 + w3 * 2.0 + w4 * 2.0 + 0.8;
  return c / w;
}

void main() {
  // Perspective-correct planar reflection: where this street fragment lands
  // in the mirrored camera's view of the city.
  vec4 clip = uProj * (uMirrorView * vec4(vWorld, 1.0));
  vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
  bool inMirror = clip.w > 0.01 && uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;

  // Distance + horizon fade (port of the original grid fade) — the street
  // melts into the sky near the horizon.
  float d = distance(vWorld, vec3(uCamPos.x, 0.0, uCamPos.z));
  float fade = 1.0 - smoothstep(8.0, 68.0, d);
  float horizon = smoothstep(-55.0, -95.0, vWorld.z);
  float skyMask = 1.0 - horizon;

  // Scrolling neon grid (port of the original GRID_FRAG) laid over the mirror.
  float t = uTime * 3.0;
  vec3 wz = vWorld + vec3(0.0, 0.0, t);
  float dx = abs(fract(wz.x / 2.0) - 0.5) * 2.0;
  float dz = abs(fract(wz.z / 2.0) - 0.5) * 2.0;
  float minor = smoothstep(0.10, 0.0, min(dx, dz));
  float sx = abs(fract(wz.x / 10.0) - 0.5) * 10.0;
  float sz = abs(fract(wz.z / 10.0) - 0.5) * 10.0;
  float major = smoothstep(0.4, 0.0, min(sx, sz));
  float line = max(minor, major * 1.5);
  float gridI = line * fade * skyMask;
  vec3 gridCol = mix(uColorA, uColorC, 0.5 + 0.5 * sin(vWorld.x * 0.05 + uTime * 0.7));

  // Ray-traced reflection of the city on the glossy asphalt.
  vec3 refl = vec3(0.0);
  if (inMirror && fade > 0.001) {
    refl = sampleReflection(uv);
    // Fresnel: grazing angles (the horizon) reflect the most, the street
    // under the camera is a flat dark gloss.
    vec3 viewDir = normalize(uCamPos - vWorld);
    float fres = 0.04 + 0.96 * pow(1.0 - max(dot(viewDir, vec3(0.0, 1.0, 0.0)), 0.0), FLOOR_FRESNEL);
    // Wet-street growth: the mirror image reads in the mid-field and out.
    float wet = 1.0 - exp(-d * FLOOR_WET);
    refl *= fres * wet;
  }

  // Dark glossy asphalt where the reflection is weak.
  vec3 asphalt = vec3(0.045, 0.05, 0.09) * skyMask;
  vec3 col = asphalt + refl + gridCol * gridI * FLOOR_GRID;

  // Melt into the sky/fog colour at the horizon (matches the clear colour
  // 0x05060c so the street and sky meet seamlessly).
  vec3 fogCol = vec3(0.02, 0.024, 0.047);
  col = mix(col, fogCol, horizon);

  gl_FragColor = vec4(col, 1.0);
}
`;

// ── Tower scene shader: dark glass, procedural window facade, LED strips ──
// Local box coords are passed pre-instance-scale (x/z in [-0.8,0.8] across a
// wall, y in [-0.5,0.5]) along with the instanced height, so the fragment
// lays a world-unit window grid onto every wall — tall towers carry many
// floors, short ones just a few. Windows mix between dark and lit (tinted
// across the palette + warm offices, some flickering), and every tower gets
// neon edge strips + a lit roof band, then the fresnel glass rim.
const TOWER_VERT = `
varying vec3 vWNormal;
varying vec3 vWorld;
varying vec3 vLocal;
varying float vTowerH;
void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  #ifdef USE_INSTANCING
    wp = modelMatrix * instanceMatrix * vec4(position, 1.0);
    vTowerH = length((instanceMatrix * vec4(0.0, 1.0, 0.0, 0.0)).xyz);
  #else
    vTowerH = 1.0;
  #endif
  vWorld = wp.xyz;
  vLocal = position; // unit-box local coords, pre-Y-scale
  // Towers are axis-aligned boxes scaled only on Y, so the object-space
  // normal is already the world normal (mat3(modelMatrix) is identity).
  vWNormal = normalize(mat3(modelMatrix) * normal);
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

const TOWER_FRAG = `
#define TOWER_RIM ${TOWER_RIM_POW.toFixed(2)}
#define TOWER_RIMB ${TOWER_RIM_BOOST.toFixed(2)}
#define WIN_W 0.30
#define WIN_H 0.36
#define WIN_GAPX 0.38
#define WIN_GAPY 0.46
uniform float uTime;
uniform vec3 uColorA;
uniform vec3 uColorB;
uniform vec3 uColorC;
uniform vec3 uColorD;
varying vec3 vWNormal;
varying vec3 vWorld;
varying vec3 vLocal;
varying float vTowerH;

float hashF(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

void main() {
  vec3 N = normalize(vWNormal);
  vec3 V = normalize(cameraPosition - vWorld);
  float fres = pow(1.0 - abs(dot(N, V)), TOWER_RIM);

  // ---- Procedural facade -----------------------------------------------
  // Wall coords: side walls are vertical (N.y small) so windows only appear
  // there; the roof band is handled separately at the top of the wall.
  vec2 face = abs(vLocal.x) >= abs(vLocal.z)
    ? vec2(vLocal.z, vLocal.y)   // x-face: wall runs along z
    : vec2(vLocal.x, vLocal.y);  // z-face: wall runs along x
  vec2 pos = vec2(face.x, (face.y + 0.5) * vTowerH); // world units along + up the wall

  vec2 cell = floor(pos / vec2(WIN_GAPX, WIN_GAPY));
  vec2 fc = fract(pos / vec2(WIN_GAPX, WIN_GAPY)) - 0.5;
  float cellHash = hashF(cell + fract(vWorld.xz));

  // Window pane inside each cell (thin dark frame between panes).
  float pane = 1.0 - smoothstep(WIN_W * 0.5, WIN_W * 0.5 + 0.02, abs(fc.x))
                   - smoothstep(WIN_H * 0.5, WIN_H * 0.5 + 0.02, abs(fc.y));
  pane = clamp(pane, 0.0, 1.0);

  // Lit vs dark; lit panes tint across the palette (cyan -> blue -> violet)
  // plus a few warm-white offices, and a subset flicker (AC/server floors).
  float lit = step(0.48, cellHash);
  float tint = fract(cellHash * 7.31);
  vec3 winCol = mix(uColorA, uColorB, smoothstep(0.0, 0.45, tint));
  winCol = mix(winCol, uColorC, smoothstep(0.45, 0.78, tint));
  winCol = mix(winCol, vec3(1.0, 0.85, 0.58), smoothstep(0.78, 1.0, tint));
  float flick = step(0.55, 0.68 + 0.32 * sin(uTime * (1.5 + fract(cellHash * 13.1) * 5.0) + cellHash * 40.0));

  // Windows only on side walls (not the flat roof); scale brightness so the
  // mirror pass reflects a believable skyline.
  float sideMask = 1.0 - smoothstep(0.55, 0.85, abs(N.y));
  float winLight = pane * lit * flick * sideMask;

  // Neon edge strips down the tower corners + a lit roof band.
  float edge = smoothstep(0.72, 0.80, abs(face.x)); // wall's own ends = corners
  float roofBand = smoothstep(vTowerH - 1.4, vTowerH - 0.6, pos.y) * sideMask;
  vec3 neon = mix(uColorA, uColorD, 0.55);

  vec3 col = vec3(0.016, 0.02, 0.032);                    // dark glass
  col += winCol * winLight * 1.5;                         // lit windows
  col += neon * (edge * 1.1 + roofBand * 0.8);            // corner + roof LED
  col += mix(uColorA, uColorD, 0.5) * fres * TOWER_RIMB;  // fresnel glass rim
  float top = smoothstep(-0.2, 0.9, N.y);
  col += uColorA * top * 0.05;
  gl_FragColor = vec4(col, 1.0);
}
`;

// ── Stripped retro-synthwave sun texture ───────────────────────────────────
function makeSunTexture(): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const ctx = c.getContext("2d")!;
  const g = ctx.createRadialGradient(128, 128, 8, 128, 128, 122);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.7, "rgba(255,255,255,0.85)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 256, 256);
  // Horizontal stripes on the lower half, growing apart downward.
  for (let i = 0; i < 20; i++) {
    const y = 150 + i * i * 0.5;
    ctx.clearRect(34, y, 188, 5 + i * 1.6);
  }
  return c;
}

export function NeonCityWebGL({ className = "" }: { className?: string }) {
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
    renderer.setClearColor(0x05060c, 1);
    host.appendChild(renderer.domElement);
    renderer.domElement.style.position = "fixed";
    renderer.domElement.style.inset = "0";
    renderer.domElement.style.display = "block";

    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x05060c, 32, 110);
    const camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      0.1,
      300,
    );
    const clock = new THREE.Clock();
    const disposables: THREE.Texture[] = [];
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    // ── Palette → live uniforms ───────────────────────────────────────────
    // Shared THREE.Color uniforms mutated in place by the throttled poller.
    const colA = { value: new THREE.Color(0x22d3ee) }; // primary (cyan)
    const colB = { value: new THREE.Color(0x3b82f6) }; // secondary (blue)
    const colC = { value: new THREE.Color(0x8b5cf6) }; // tertiary (violet)
    const colD = { value: new THREE.Color(0xec4899) }; // accent (magenta)
    const applyColors = (p: ReturnType<typeof readPaletteColors>) => {
      colA.value.copy(p.primary);
      colB.value.copy(p.secondary);
      colC.value.copy(p.tertiary);
      colD.value.copy(p.accent);
    };
    applyColors(readPaletteColors());
    const poll = makePalettePoller(400, applyColors);

    // ── Mirrored camera + reflection render target ────────────────────────
    const mirrorCam = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      0.1,
      300,
    );
    // We set matrixWorld + matrixWorldInverse by hand each frame (reflected
    // across the street). Stop the renderer from recomposing them from the
    // camera's own identity position/quaternion, which would break the mirror.
    mirrorCam.matrixWorldAutoUpdate = false;
    const reflMatrix = new THREE.Matrix4().makeScale(1, -1, 1); // across y=0
    const _reflM = new THREE.Matrix4();

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let rtW = Math.floor(window.innerWidth * dpr);
    let rtH = Math.floor(window.innerHeight * dpr);
    const cityRT = new THREE.WebGLRenderTarget(
      Math.floor(rtW * MIRROR_RT_SCALE),
      Math.floor(rtH * MIRROR_RT_SCALE),
      {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat,
        // MSAA the mirror pass so the wet-street reflection isn't jagged.
        samples: renderer.capabilities.isWebGL2 ? 4 : 0,
      },
    );

    // ── Glossy mirror street (replaces the old flat grid plane) ───────────
    const floorMat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uCamPos: { value: new THREE.Vector3() },
        uProj: { value: camera.projectionMatrix.clone() },
        uMirrorView: { value: new THREE.Matrix4() },
        uCityTex: { value: cityRT.texture },
        uTexelSize: { value: new THREE.Vector2(1 / cityRT.width, 1 / cityRT.height) },
        uColorA: colA,
        uColorB: colB,
        uColorC: colC,
      },
      vertexShader: FLOOR_VERT,
      fragmentShader: FLOOR_FRAG,
      side: THREE.DoubleSide,
    });
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(300, 300, 1, 1), floorMat);
    floor.rotation.x = -Math.PI / 2;
    scene.add(floor);

    // ── Retro sun + halo ──────────────────────────────────────────────────
    const sunTex = new THREE.CanvasTexture(makeSunTexture());
    disposables.push(sunTex);
    const sunMat = new THREE.MeshBasicMaterial({
      map: sunTex,
      color: colA.value,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });
    const sun = new THREE.Mesh(new THREE.PlaneGeometry(26, 26), sunMat);
    sun.position.set(0, 7, -75);
    scene.add(sun);

    const haloTex = new THREE.CanvasTexture(makeSunTexture());
    disposables.push(haloTex);
    const haloMat = new THREE.MeshBasicMaterial({
      map: haloTex,
      color: colC.value,
      transparent: true,
      opacity: 0.4,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });
    const halo = new THREE.Mesh(new THREE.PlaneGeometry(52, 52), haloMat);
    halo.position.set(0, 7, -75);
    scene.add(halo);

    // ── Tower silhouettes (instanced, now glossy black glass) ─────────────
    const towerGeo = new THREE.BoxGeometry(1.6, 1, 1.6);
    const towerMat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uColorA: colA,
        uColorB: colB,
        uColorC: colC,
        uColorD: colD,
      },
      vertexShader: TOWER_VERT,
      fragmentShader: TOWER_FRAG,
      side: THREE.DoubleSide,
    });
    const towers = new THREE.InstancedMesh(towerGeo, towerMat, 48);
    const dummy = new THREE.Object3D();
    for (let i = 0; i < 48; i++) {
      const side = i % 2 === 0 ? -1 : 1;
      const z = -8 - Math.floor(i / 2) * 4.6;
      const x = side * (4 + ((i * 37) % 26));
      const h = 5 + ((i * 53) % 19);
      dummy.position.set(x, h / 2, z);
      dummy.scale.set(1, h, 1);
      dummy.updateMatrix();
      towers.setMatrixAt(i, dummy.matrix);
    }
    towers.instanceMatrix.needsUpdate = true;
    scene.add(towers);

    // ── Neon pylons along the avenue (pulsing, additive) ──────────────────
    const pylonGeo = new THREE.BoxGeometry(0.5, 16, 0.5);
    const pylons: THREE.Mesh[] = [];
    for (let i = 0; i < 10; i++) {
      const side = i % 2 === 0 ? -1 : 1;
      const mat = new THREE.MeshBasicMaterial({
        color: i % 3 === 0 ? colA.value : colD.value,
        transparent: true,
        opacity: 0.8,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
      });
      const m = new THREE.Mesh(pylonGeo, mat);
      m.position.set(side * 3.5, 8, -12 - i * 5);
      m.scale.y = 0.6 + Math.random() * 0.9;
      scene.add(m);
      pylons.push(m);
    }

    // ── Floating holograms (smooth high-poly geometry + projection shader)
    // A fresnel-glow core, a moving scan band, an interference shimmer and a
    // faint data-dot grid make them read as lit volumetric holos rather than
    // wireframes. Each takes one of the live palette colors as its tint.
    const HOLO_VERT = `
varying vec3 vWNormal;
varying vec3 vWorld;
void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorld = wp.xyz;
  vWNormal = normalize(mat3(modelMatrix) * normal);
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;
    const HOLO_FRAG = `
uniform float uTime;
uniform vec3 uTint;
varying vec3 vWNormal;
varying vec3 vWorld;
float hashH(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
void main() {
  vec3 N = normalize(vWNormal);
  vec3 V = normalize(cameraPosition - vWorld);
  float fres = pow(1.0 - abs(dot(N, V)), 2.5);
  // Local Y position for the scan band (pass in world-y, integer-ish floors).
  float scan = smoothstep(0.04, 0.0, abs(fract(vWorld.y * 0.5 + uTime * 0.35) - 0.5) - 0.2);
  // Faint interference shimmer.
  float shimmer = 0.5 + 0.5 * sin(vWorld.y * 1.7 + vWorld.x * 2.3 - uTime * 2.0);
  // Dot grid projected on the surface (world-space, so it feels holographic).
  vec2 g = abs(fract(vWorld.xz * 3.0) - 0.5);
  float dot = smoothstep(0.32, 0.0, min(g.x, g.y));
  float core = fres * 0.9 + scan * 0.9 + dot * 0.12 * shimmer;
  vec3 col = uTint * (0.16 + core);
  gl_FragColor = vec4(col, core * 0.9 + 0.12);
}
`;
    const holos: THREE.Mesh[] = [];
    const holoTints = [colA.value, colB.value, colC.value];
    for (let i = 0; i < 3; i++) {
      const geo =
        i % 2 === 0
          ? new THREE.TorusGeometry(1.5, 0.45, 24, 96)
          : new THREE.IcosahedronGeometry(1.7, 3);
      const mat = new THREE.ShaderMaterial({
        uniforms: {
          uTime: { value: 0 },
          uTint: { value: holoTints[i] },
        },
        vertexShader: HOLO_VERT,
        fragmentShader: HOLO_FRAG,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
      });
      const m = new THREE.Mesh(geo, mat);
      m.position.set((i - 1) * 14, 9 + i * 2.5, -22 - i * 14);
      scene.add(m);
      holos.push(m);
    }

    // ── Rising data motes (above the street so the mirror doesn't hide them)
    const MOTES = 220;
    const motePos = new Float32Array(MOTES * 3);
    const moteCol = new Float32Array(MOTES * 3);
    const moteSpeeds = new Float32Array(MOTES);
    const moteTints = [colA.value, colC.value, colD.value];
    for (let i = 0; i < MOTES; i++) {
      motePos[i * 3] = (Math.random() - 0.5) * 70;
      motePos[i * 3 + 1] = Math.random() * 14;
      motePos[i * 3 + 2] = 20 - Math.random() * 95;
      const tint = moteTints[i % 3];
      moteCol[i * 3] = tint.r;
      moteCol[i * 3 + 1] = tint.g;
      moteCol[i * 3 + 2] = tint.b;
      moteSpeeds[i] = 0.6 + Math.random() * 1.4;
    }
    const moteGeo = new THREE.BufferGeometry();
    moteGeo.setAttribute("position", new THREE.BufferAttribute(motePos, 3));
    moteGeo.setAttribute("color", new THREE.BufferAttribute(moteCol, 3));
    const moteMat = new THREE.PointsMaterial({
      size: 0.35,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });
    const motes = new THREE.Points(moteGeo, moteMat);
    scene.add(motes);

    // ── Shockwave rings live in their own group so the mirror pass can
    //    hide them (they sit on the street and shouldn't double-render). ──
    const ringGroup = new THREE.Group();
    scene.add(ringGroup);

    // ── Interaction state ─────────────────────────────────────────────────
    const orbit = {
      azimuth: 0,
      polar: 1.22,
      radius: 30,
      target: new THREE.Vector3(0, 2, -10),
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
        0.15,
        Math.min(1.55, orbit.polar - (e.clientY - lastPY) * 0.005),
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
      // Fire the signature event (chrome tint) + a shockwave at the hit point.
      const ndcX = (e.clientX / window.innerWidth) * 2 - 1;
      const ndcY = -(e.clientY / window.innerHeight) * 2 + 1;
      emitCosmosEvent({ type: "neon-surge", heat: 0.6, hue: 199, x: ndcX, y: ndcY });
      const ndc = new THREE.Vector3(ndcX, ndcY, 0.5).unproject(camera);
      const dir = ndc.sub(camera.position).normalize();
      const tHit = -camera.position.y / dir.y;
      const hit = camera.position.clone().add(dir.clone().multiplyScalar(tHit));
      spawnRing(hit, colA.value);
    };
    const onWheel = (e: WheelEvent) => {
      if (!e.shiftKey) return;
      e.preventDefault();
      orbit.radius = Math.max(12, Math.min(60, orbit.radius * (1 + e.deltaY * 0.001)));
    };
    const onContextMenu = (e: Event) => e.preventDefault();
    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      mirrorCam.aspect = camera.aspect;
      rtW = Math.floor(window.innerWidth * dpr);
      rtH = Math.floor(window.innerHeight * dpr);
      cityRT.setSize(Math.floor(rtW * MIRROR_RT_SCALE), Math.floor(rtH * MIRROR_RT_SCALE));
      floorMat.uniforms.uTexelSize.value.set(1 / cityRT.width, 1 / cityRT.height);
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("wheel", onWheel, { passive: false });
    window.addEventListener("contextmenu", onContextMenu);
    window.addEventListener("resize", onResize);

    // ── Shockwave rings ───────────────────────────────────────────────────
    const rings: Ring[] = [];
    function spawnRing(at: THREE.Vector3, color: THREE.Color) {
      const geo = new THREE.RingGeometry(0.6, 0.85, 48);
      const mat = new THREE.MeshBasicMaterial({
        color: color.clone(),
        transparent: true,
        opacity: 0.9,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.set(at.x, 0.2, at.z);
      ringGroup.add(mesh);
      rings.push({ mesh, life: 0, maxLife: 1.4 });
    }

    // ── Ambient data bursts (visual + soft chrome pulse) ──────────────────
    let ambientTimer = 12 + Math.random() * 8;
    function updateAmbient(dt: number, t: number) {
      ambientTimer -= dt;
      if (ambientTimer > 0) return;
      ambientTimer = 14 + Math.random() * 8;
      const x = (Math.random() - 0.5) * 24;
      const z = -50 - Math.random() * 18;
      const hit = new THREE.Vector3(x, 0, z);
      spawnRing(hit, colC.value);
      emitCosmosEvent({
        type: "neon-surge",
        heat: 0.2,
        hue: 262,
        x: 0,
        y: 0,
      });
    }

    // Scratch vectors for orienting the billboards to whichever camera is
    // rendering, without allocating per frame.
    const _sunTarget = new THREE.Vector3();
    function orientSunTo(worldPos: THREE.Vector3) {
      _sunTarget.copy(worldPos);
      sun.lookAt(_sunTarget);
      halo.lookAt(_sunTarget);
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

      if (!reduced) orbit.azimuth += dt * 0.02;

      // Camera from orbit + mouse parallax.
      const pa = orbit.polar;
      const aa = orbit.azimuth;
      const r = orbit.radius;
      camera.position.set(
        orbit.target.x + r * Math.sin(pa) * Math.sin(aa) + mouse.x * 0.9,
        orbit.target.y + r * Math.cos(pa) + -mouse.y * 0.5,
        orbit.target.z + r * Math.sin(pa) * Math.cos(aa),
      );
      camera.lookAt(orbit.target);
      camera.updateMatrixWorld(true);

      // Mirrored camera: reflect the real camera across the y=0 street plane.
      const reflM = _reflM.copy(reflMatrix).multiply(camera.matrixWorld);
      mirrorCam.matrixWorld.copy(reflM);
      mirrorCam.matrixWorldInverse.copy(reflM).invert();
      mirrorCam.projectionMatrix.copy(camera.projectionMatrix);

      floorMat.uniforms.uTime.value = t;
      floorMat.uniforms.uCamPos.value.copy(camera.position);
      floorMat.uniforms.uProj.value.copy(camera.projectionMatrix);
      floorMat.uniforms.uMirrorView.value.copy(mirrorCam.matrixWorldInverse);

      // Motes rise and wrap.
      const pos = motes.geometry.getAttribute("position") as THREE.BufferAttribute;
      for (let i = 0; i < MOTES; i++) {
        let y = pos.getY(i) + moteSpeeds[i] * dt;
        if (y > 15) y = 0;
        pos.setY(i, y);
      }
      pos.needsUpdate = true;

      // Tower facade / hologram live uniforms.
      towerMat.uniforms.uTime.value = t;

      // Holograms rotate + bob + drive their scan shimmer.
      for (let i = 0; i < holos.length; i++) {
        holos[i].rotation.y += dt * 0.5;
        holos[i].rotation.x += dt * 0.2;
        holos[i].position.y = 9 + i * 2.5 + Math.sin(t * 0.8 + i * 2) * 0.7;
        (holos[i].material as THREE.ShaderMaterial).uniforms.uTime.value = t;
      }

      // Pylons pulse.
      for (let i = 0; i < pylons.length; i++) {
        const m = pylons[i].material as THREE.MeshBasicMaterial;
        m.opacity = 0.5 + 0.4 * Math.sin(t * 2.2 + i * 0.7);
      }

      // Rings expand + fade.
      for (let i = rings.length - 1; i >= 0; i--) {
        const ring = rings[i];
        ring.life += dt;
        const p = Math.min(1, ring.life / ring.maxLife);
        ring.mesh.scale.setScalar(1 + p * 22);
        (ring.mesh.material as THREE.MeshBasicMaterial).opacity = (1 - p) * 0.9;
        if (p >= 1) {
          ringGroup.remove(ring.mesh);
          ring.mesh.geometry.dispose();
          (ring.mesh.material as THREE.Material).dispose();
          rings.splice(i, 1);
        }
      }

      if (!reduced) updateAmbient(dt, t);

      // ── Pass A: render the city into the reflection target from the
      //    mirrored camera (street + rings hidden, billboards face the mirror).
      _sunTarget.setFromMatrixPosition(mirrorCam.matrixWorld);
      orientSunTo(_sunTarget);
      floor.visible = false;
      ringGroup.visible = false;
      renderer.setRenderTarget(cityRT);
      renderer.render(scene, mirrorCam);

      // ── Pass B: render the main scene — the street is now the mirror.
      orientSunTo(camera.position);
      floor.visible = true;
      ringGroup.visible = true;
      renderer.setRenderTarget(null);
      renderer.render(scene, camera);
    }
    animate();

    // ── Cleanup (palette switch / unmount) ────────────────────────────────
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("wheel", onWheel);
      window.removeEventListener("contextmenu", onContextMenu);
      window.removeEventListener("resize", onResize);
      rings.forEach((r) => {
        r.mesh.geometry.dispose();
        (r.mesh.material as THREE.Material).dispose();
      });
      scene.traverse((obj) => {
        const mesh = obj as THREE.Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
        const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else if (mat) mat.dispose();
      });
      disposables.forEach((tt) => tt.dispose());
      cityRT.dispose();
      renderer.forceContextLoss();
      renderer.dispose();
      if (renderer.domElement.parentNode === host) {
        host.removeChild(renderer.domElement);
      }
    };
  }, [webglOk]);

  if (!webglOk || broken) {
    // Defensive fallback — the dispatcher normally routes no-WebGL palettes to
    // the 2D canvas, so this only shows if the component is mounted directly.
    return (
      <div
        aria-hidden
        className={`fixed inset-0 -z-10 ${className}`}
        style={{
          background:
            "radial-gradient(ellipse at 50% 110%, hsl(var(--palette-primary) / 0.35), hsl(var(--palette-secondary) / 0.12) 45%, transparent 70%)",
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
