"use client";

// ── Aurora Borealis — the northern-lights world for "aurora-borealis" ─────
//
// Self-contained WebGL background modeled on CosmosWebGL.tsx's skeleton. Scene:
// undulating fbm aurora curtains overhead, a dark low-poly mountain ridge,
// sparse twinkling stars, occasional shooting stars, and a click-triggered
// brightness wave that ripples across the curtains. Colors read live from
// --palette-* so the PaletteEditor retunes the lights in real time.

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { supportsWebGL } from "./webglSupport";
import { readPaletteColors, makePalettePoller } from "./worldUtils";
import { emitCosmosEvent } from "./cosmosEvents";

// ── Aurora curtain shader: layered noise folds + click ripple ─────────────
const CURTAIN_VERT = `
varying vec2 vUv;
varying vec3 vWorld;
void main() {
  vUv = uv;
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorld = wp.xyz;
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

const CURTAIN_FRAG = `
uniform float uTime;
uniform vec3 uColorA;
uniform vec3 uColorB;
uniform vec3 uColorC;
uniform vec3 uAccent;
uniform float uPulse;
uniform vec3 uPulseCenter;
uniform float uFlash;
varying vec2 vUv;
varying vec3 vWorld;
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
  float v = vUv.y;

  // Curtain drapery — drifting fbm folds blended with a travelling wave.
  float fold = clamp(fbm(vec2(vUv.x * 2.6 + uTime * 0.16, vUv.y * 1.3 + uTime * 0.05)), 0.0, 1.0);
  float wavy = sin(vUv.x * 4.0 - uTime * 0.25) * 0.5 + 0.5;
  float drape = clamp(fold * 0.75 + wavy * 0.25, 0.0, 1.0);

  // Real-aurora colour ladder: bright green base -> teal -> cyan -> violet,
  // with a rose/pink nitrogen fringe at the top of tall curtains.
  float band = clamp(v * 0.85 + drape * 0.3 - 0.08, 0.0, 1.0);
  vec3 col = uAccent;                                    // green base
  col = mix(col, uColorA, smoothstep(0.04, 0.42, band)); // teal
  col = mix(col, uColorB, smoothstep(0.35, 0.62, band)); // cyan
  col = mix(col, uColorC, smoothstep(0.6, 0.85, band));  // violet
  vec3 rose = mix(uColorC, vec3(0.9, 0.42, 0.62), 0.85);
  col = mix(col, rose, smoothstep(0.82, 1.0, band));     // pink fringe

  // Vertical auroral rays — bright striations that ebb and flow.
  float rays = noise(vec2(vUv.x * 24.0 - uTime * 0.06, vUv.y * 4.0 + uTime * 0.03));
  rays = rays * rays;
  float rayEnv = 0.35 + 0.65 * fbm(vec2(vUv.x * 5.0 - uTime * 0.12, vUv.y * 2.5 + uTime * 0.06));
  float rayMask = smoothstep(0.42, 0.9, rays * rayEnv);
  col += uColorB * rayMask * 0.4;
  col += rose * rayMask * 0.25;

  // Bright curtain base hugging the lower edge.
  float baseGlow = exp(-v * 5.0) * (0.35 + 0.65 * drape);

  // Dissolve the top and bottom edges into the sky.
  float vfade = smoothstep(0.0, 0.14, v) * smoothstep(1.0, 0.78, v);

  float alpha = (0.10 + 0.5 * drape + 0.45 * baseGlow + rayMask * 0.22) * vfade;

  // Click: a radial ripple from uPulseCenter plus a whole-scene flash.
  vec2 d = vWorld.xz - uPulseCenter.xz;
  float pd = dot(d, d);
  float wave = exp(-pd * 0.004);
  alpha += uPulse * (0.9 * wave + 0.35) * vfade;
  col += uColorB * uPulse * wave * 0.45;
  col += uAccent * uFlash * 0.55;
  alpha += uFlash * 0.3 * vfade;

  gl_FragColor = vec4(col, alpha);
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
    const curtainMat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      uniforms: {
        uTime: { value: 0 },
        uColorA: colA,
        uColorB: colB,
        uColorC: colC,
        uAccent: colD,
        uPulse: { value: 0 },
        uPulseCenter: { value: pulseCenter },
        uFlash: { value: 0 },
      },
      vertexShader: CURTAIN_VERT,
      fragmentShader: CURTAIN_FRAG,
    });
    // Six curtain planes spread across the width so auroras fill the whole
    // sky rather than a single central ribbon — near/mid/far layers for depth.
    const curtainCfg = [
      { x: -48, y: 13, z: -46, ry: 0.45, w: 56, h: 28 },
      { x: -26, y: 15, z: -52, ry: -0.4, w: 60, h: 32 },
      { x: -8, y: 12, z: -26, ry: 0.1, w: 48, h: 26 },
      { x: 12, y: 17, z: -44, ry: -0.28, w: 62, h: 34 },
      { x: 32, y: 14, z: -34, ry: 0.38, w: 54, h: 29 },
      { x: 52, y: 16, z: -50, ry: -0.48, w: 58, h: 31 },
    ];
    for (const cfg of curtainCfg) {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(cfg.w, cfg.h), curtainMat);
      m.position.set(cfg.x, cfg.y, cfg.z);
      m.rotation.y = cfg.ry;
      m.renderOrder = 2;
      scene.add(m);
    }

    // ── Mountain ridge (two layers for depth) ─────────────────────────────
    function makeRidge(mz: number, color: number, scale: number) {
      // Horizontal terrain, flat near the camera and rising into peaks toward
      // the horizon, so it reads as mountain silhouettes at the bottom.
      const geo = new THREE.PlaneGeometry(260, 70, 160, 30);
      geo.rotateX(-Math.PI / 2);
      const pos = geo.attributes.position as THREE.BufferAttribute;
      for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i);
        const wz = pos.getZ(i) + mz; // world z
        const far = THREE.MathUtils.smoothstep(wz, 12, -45);
        const peak =
          Math.abs(Math.sin(x * 0.012 + 0.7)) * 9 +
          Math.abs(Math.sin(x * 0.031 + 2.1)) * 7 +
          Math.abs(Math.sin(x * 0.007 + 5.3)) * 14;
        pos.setY(i, -3 + peak * scale * far * 0.6);
      }
      pos.needsUpdate = true;
      geo.computeVertexNormals();
      const m = new THREE.Mesh(
        geo,
        new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide }),
      );
      m.position.set(0, 0, mz);
      m.renderOrder = 1;
      scene.add(m);
    }
    makeRidge(-32, 0x04060c, 1);
    makeRidge(-48, 0x020408, 0.7);

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

      curtainMat.uniforms.uTime.value = t;
      curtainMat.uniforms.uPulse.value = pulse;
      curtainMat.uniforms.uFlash.value = flash;
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
