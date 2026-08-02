"use client";

// ── Ocean Depths — the underwater world for the "ocean-depths" palette ────
//
// Self-contained WebGL background modeled on CosmosWebGL.tsx's skeleton. Scene:
// a vertex-displaced wave plane with fresnel colouring, additive god-ray shafts
// slanting from above, sinking marine snow, a distant whale silhouette, and
// click-triggered bioluminescent bursts. Colors read live from --palette-* so
// the PaletteEditor retunes the water in real time.

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { supportsWebGL } from "./webglSupport";
import { readPaletteColors, makePalettePoller } from "./worldUtils";
import { emitCosmosEvent } from "./cosmosEvents";

// ── Water plane shader: Gerstner swell + lighting ─────────────────────────
const WATER_VERT = `
uniform float uTime;
uniform float uAmp;
varying vec3 vWorld;
varying vec3 vNormalW;
varying float vCrest;
// Gerstner wave: displaces the surface along its direction and tilts the
// normal, producing the sharp crest / round trough shape real swell has.
void gerstner(float amp, float wl, vec2 dir, float sp, float t, vec2 base, inout vec3 p, inout vec3 n) {
  float k = 6.283185307 / wl;
  float c = cos(k * dot(dir, base) + sp * t);
  float s = sin(k * dot(dir, base) + sp * t);
  float q = amp * c;
  p.x += dir.x * q; p.z += dir.y * q;
  p.y += amp * s;
  n.x -= dir.x * k * q; n.z -= dir.y * k * q;
}
void main() {
  vec3 p = position;
  vec2 base = p.xz;
  vec3 n = vec3(0.0, 1.0, 0.0);
  float t = uTime;

  // Wave A — long slow swell.
  gerstner(uAmp * 1.0, 26.0, vec2(0.82, 0.57), 1.25, t, base, p, n);
  // Wave B — mid-length chop.
  gerstner(uAmp * 0.6, 13.0, vec2(-0.5, 0.87), 1.8, t, base, p, n);
  // Wave C — short sparkle.
  gerstner(uAmp * 0.35, 7.0, vec2(0.3, -0.95), 2.6, t, base, p, n);

  // Normalized wave height (0..1) → foam mask in the fragment.
  float hMax = uAmp * 1.95 + 0.001;
  vCrest = clamp(0.5 + 0.5 * (p.y / hMax), 0.0, 1.0);
  vNormalW = normalize(n);

  vec4 wp = modelMatrix * vec4(p, 1.0);
  vWorld = wp.xyz;
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

const WATER_FRAG = `
uniform float uTime;
uniform vec3 uColorA;
uniform vec3 uColorB;
uniform vec3 uColorC;
varying vec3 vWorld;
varying vec3 vNormalW;
varying float vCrest;
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
void main() {
  vec3 N = normalize(vNormalW);
  vec3 V = normalize(cameraPosition - vWorld);

  // Fresnel — flatter look overhead, deep-sky colour toward the horizon.
  float fres = pow(1.0 - max(dot(N, V), 0.0), 3.0);

  // Body of the water: deep teal-blue below, brighter near the surface.
  vec3 deepCol = mix(uColorA, uColorC, 0.5) * 0.6;
  vec3 shallowCol = mix(uColorB, uColorA, 0.35);
  vec3 col = mix(deepCol, shallowCol, clamp((vWorld.y + 1.5) * 0.22, 0.0, 1.0) * (1.0 - fres));

  // Sun glitter — tight specular path along the god-ray light direction.
  vec3 L = normalize(vec3(0.45, 0.75, -0.4));
  vec3 H = normalize(L + V);
  float spec = pow(max(dot(N, H), 0.0), 140.0);
  col += vec3(0.9, 0.97, 1.0) * spec * 1.8;

  // Animated caustic shimmer under the surface.
  float ca = fbm(vWorld.xz * 0.6 + vec2(uTime * 0.4, uTime * 0.25));
  col *= 0.8 + 0.5 * ca;

  // Foam — broken white lines where the surface nears a wave crest.
  float foam = smoothstep(0.86, 1.0, vCrest) * (0.5 + 0.5 * noise(vWorld.xz * 2.4 + uTime * 0.5));
  col = mix(col, vec3(0.95, 0.98, 1.0), foam * 0.9);

  // Depth haze with distance.
  float d = length(cameraPosition.xz - vWorld.xz);
  col *= 0.82 + 0.18 * exp(-d * 0.015);

  float alpha = 0.72 + fres * 0.22;
  gl_FragColor = vec4(col, alpha);
}
`;

// Vertical gradient for god-ray shafts.
function makeRayTexture(): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = 16;
  c.height = 128;
  const ctx = c.getContext("2d")!;
  const g = ctx.createLinearGradient(0, 0, 0, 128);
  g.addColorStop(0, "rgba(255,255,255,0.85)");
  g.addColorStop(0.6, "rgba(255,255,255,0.28)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 16, 128);
  return c;
}

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
    host.appendChild(renderer.domElement);
    renderer.domElement.style.position = "fixed";
    renderer.domElement.style.inset = "0";
    renderer.domElement.style.display = "block";

    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x04121c, 40, 130);
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

    // ── Water plane ───────────────────────────────────────────────────────
    const waterMat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: {
        uTime: { value: 0 },
        uAmp: { value: reduced ? 0 : 0.8 },
        uColorA: colA,
        uColorB: colB,
        uColorC: colC,
      },
      vertexShader: WATER_VERT,
      fragmentShader: WATER_FRAG,
    });
    const water = new THREE.Mesh(new THREE.PlaneGeometry(220, 220, 64, 64), waterMat);
    water.rotation.x = -Math.PI / 2;
    water.renderOrder = 1;
    scene.add(water);

    // ── God rays ──────────────────────────────────────────────────────────
    const rayTex = new THREE.CanvasTexture(makeRayTexture());
    disposables.push(rayTex);
    const rays: THREE.Mesh[] = [];
    const rayCfg = [
      { x: -14, z: -12, rot: 0.5 },
      { x: 6, z: -22, rot: 0.35 },
      { x: -4, z: -2, rot: 0.62 },
      { x: 18, z: -6, rot: 0.28 },
    ];
    for (const cfg of rayCfg) {
      const m = new THREE.Mesh(
        new THREE.PlaneGeometry(13, 54),
        new THREE.MeshBasicMaterial({
          map: rayTex,
          color: colD.value,
          transparent: true,
          opacity: 0.5,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
          side: THREE.DoubleSide,
        }),
      );
      m.position.set(cfg.x, 22, cfg.z);
      m.rotation.x = cfg.rot;
      m.renderOrder = 2;
      scene.add(m);
      rays.push(m);
    }

    // ── Marine snow ───────────────────────────────────────────────────────
    const SNOW = 420;
    const snowPos = new Float32Array(SNOW * 3);
    const snowSpeed = new Float32Array(SNOW);
    for (let i = 0; i < SNOW; i++) {
      snowPos[i * 3] = (Math.random() - 0.5) * 140;
      snowPos[i * 3 + 1] = -4 + Math.random() * 36;
      snowPos[i * 3 + 2] = 30 - Math.random() * 130;
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
    snow.renderOrder = 3;
    scene.add(snow);

    // ── Distant whale silhouette ──────────────────────────────────────────
    const whale = new THREE.Group();
    const whaleBody = new THREE.Mesh(
      new THREE.SphereGeometry(1.6, 10, 8),
      new THREE.MeshBasicMaterial({ color: 0x030a12 }),
    );
    whaleBody.scale.set(1.8, 0.85, 0.6);
    const whaleTail = new THREE.Mesh(
      new THREE.ConeGeometry(0.55, 1.8, 4),
      new THREE.MeshBasicMaterial({ color: 0x030a12 }),
    );
    whaleTail.position.set(0, 0, -1.7);
    whaleTail.rotation.x = Math.PI / 2;
    whaleTail.scale.set(0.6, 1, 0.18);
    whale.add(whaleBody, whaleTail);
    whale.position.set(60, 12, -60);
    whale.scale.setScalar(1.6);
    whale.renderOrder = 0;
    scene.add(whale);

    // ── Bioluminescent bursts ─────────────────────────────────────────────
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
      points.renderOrder = 4;
      scene.add(points);
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
      glow.renderOrder = 5;
      scene.add(glow);
      bursts.push({ points, glow, pos, vel, life: 0, maxLife: 1.2, count });
    }

    // ── Splash rings — expanding circles on the surface ───────────────────
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
          blending: THREE.AdditiveBlending,
          side: THREE.DoubleSide,
        }),
      );
      m.rotation.x = -Math.PI / 2;
      m.position.set(at.x, 0.12, at.z);
      m.renderOrder = 4;
      scene.add(m);
      splashes.push({ mesh: m, life: 0, maxLife: 1.0 });
    }

    // ── Interaction state ─────────────────────────────────────────────────
    const orbit = {
      azimuth: 0,
      polar: 1.28,
      radius: 24,
      target: new THREE.Vector3(0, 6, -10),
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
      emitCosmosEvent({ type: "ocean-surge", heat: 0.55, hue: 189, x: ndcX, y: ndcY });
      // Ray against the y=0 water plane.
      const ndc = new THREE.Vector3(ndcX, ndcY, 0.5).unproject(camera);
      const dir = ndc.sub(camera.position).normalize();
      const tHit = -camera.position.y / dir.y;
      const hit = camera.position.clone().addScaledVector(dir, Math.max(tHit, 0));
      spawnBurst(hit, 130, 1.6);
      spawnSplash(hit, 1);
    };
    const onWheel = (e: WheelEvent) => {
      if (!e.shiftKey) return;
      e.preventDefault();
      orbit.radius = Math.max(10, Math.min(50, orbit.radius * (1 + e.deltaY * 0.001)));
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

    // ── Ambient plankton glows ────────────────────────────────────────────
    let ambientTimer = 8 + Math.random() * 5;
    function updateAmbient(dt: number) {
      ambientTimer -= dt;
      if (ambientTimer > 0) return;
      ambientTimer = 9 + Math.random() * 6;
      spawnBurst(
        new THREE.Vector3((Math.random() - 0.5) * 60, 4, -30 - Math.random() * 30),
        26,
        0.6,
      );
      emitCosmosEvent({ type: "ocean-surge", heat: 0.2, hue: 166, x: 0, y: 0 });
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

      waterMat.uniforms.uTime.value = t;

      // God rays sway gently.
      for (let i = 0; i < rays.length; i++) {
        rays[i].rotation.y = Math.sin(t * 0.2 + i * 1.7) * 0.12;
      }

      // Marine snow sinks and wraps.
      const sp = snow.geometry.getAttribute("position") as THREE.BufferAttribute;
      for (let i = 0; i < SNOW; i++) {
        let y = sp.getY(i) - snowSpeed[i] * dt;
        if (y < -4) y = 30;
        sp.setY(i, y);
      }
      sp.needsUpdate = true;

      // Whale glides slowly across the distance.
      if (!reduced) {
        whale.position.x -= dt * 1.1;
        whale.position.z += dt * 0.35;
        whale.rotation.y = 0.35;
        if (whale.position.x < -80) whale.position.set(70, 12, -55);
      }

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
          scene.remove(b.points);
          scene.remove(b.glow);
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
          scene.remove(s.mesh);
          (s.mesh.material as THREE.Material).dispose();
          splashes.splice(i, 1);
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
      splashes.forEach((s) => {
        (s.mesh.material as THREE.Material).dispose();
      });
      splashGeo.dispose();
      bursts.forEach((b) => {
        b.points.geometry.dispose();
        (b.points.material as THREE.Material).dispose();
        (b.glow.material as THREE.Material).dispose();
      });
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
