"use client";

// ── Neon City — the synthwave grid-city world for the "neon-cyber" palette ─
//
// A self-contained WebGL background modeled on CosmosWebGL.tsx's skeleton:
// probe WebGL, mount a renderer into a host div, own the rAF loop, dispose
// everything (incl. forceContextLoss) on unmount, respect reduced-motion,
// and emit a signature event on guarded click so PaletteEventSync tints the
// chrome.
//
// Scene: a glowing perspective grid scrolling toward the camera, a striped
// retro-synthwave sun on the horizon, dark tower silhouettes flanking a
// central avenue, neon pylons, floating wireframe holograms, rising data
// motes, and click-spawned shockwave rings. Colors are read live from the
// palette's --palette-* CSS vars (no fixed colors) so the in-app
// PaletteEditor retunes the city in real time.

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

// ── Grid floor shader ──────────────────────────────────────────────────────
const GRID_VERT = `
varying vec3 vWorld;
void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorld = wp.xyz;
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

const GRID_FRAG = `
uniform float uTime;
uniform vec3 uColorA;
uniform vec3 uColorB;
varying vec3 vWorld;
void main() {
  float t = uTime * 3.0;
  vec3 wz = vWorld + vec3(0.0, 0.0, t);        // grid scrolls toward the camera
  // Minor lines every 2 units
  float dx = abs(fract(wz.x / 2.0) - 0.5) * 2.0;
  float dz = abs(fract(wz.z / 2.0) - 0.5) * 2.0;
  float minor = smoothstep(0.10, 0.0, min(dx, dz));
  // Major lines every 10 units, brighter
  float sx = abs(fract(wz.x / 10.0) - 0.5) * 10.0;
  float sz = abs(fract(wz.z / 10.0) - 0.5) * 10.0;
  float major = smoothstep(0.4, 0.0, min(sx, sz));
  float line = max(minor, major * 1.5);
  // Fade with distance from the camera and into the horizon.
  float d = distance(vWorld, vec3(cameraPosition.x, 0.0, cameraPosition.z));
  float fade = 1.0 - smoothstep(8.0, 68.0, d);
  float horizon = smoothstep(-55.0, -95.0, vWorld.z);
  float intensity = line * fade * (1.0 - horizon);
  vec3 col = mix(uColorA, uColorB, 0.5 + 0.5 * sin(vWorld.x * 0.05 + uTime * 0.7));
  gl_FragColor = vec4(col, intensity * 0.9);
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

    // ── Grid floor ────────────────────────────────────────────────────────
    const gridMat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uTime: { value: 0 },
        uColorA: colA,
        uColorB: colC,
      },
      vertexShader: GRID_VERT,
      fragmentShader: GRID_FRAG,
    });
    const grid = new THREE.Mesh(new THREE.PlaneGeometry(240, 240, 1, 1), gridMat);
    grid.rotation.x = -Math.PI / 2;
    scene.add(grid);

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

    // ── Tower silhouettes (instanced) ─────────────────────────────────────
    const towerGeo = new THREE.BoxGeometry(1.6, 1, 1.6);
    const towerMat = new THREE.MeshBasicMaterial({ color: 0x04060c });
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
      });
      const m = new THREE.Mesh(pylonGeo, mat);
      m.position.set(side * 3.5, 8, -12 - i * 5);
      m.scale.y = 0.6 + Math.random() * 0.9;
      scene.add(m);
      pylons.push(m);
    }

    // ── Floating wireframe holograms ──────────────────────────────────────
    const holos: THREE.Mesh[] = [];
    const holoColors = [colA.value, colB.value, colC.value];
    for (let i = 0; i < 3; i++) {
      const geo =
        i % 2 === 0
          ? new THREE.TorusGeometry(1.5, 0.45, 10, 32)
          : new THREE.OctahedronGeometry(1.7, 0);
      const mat = new THREE.MeshBasicMaterial({
        color: holoColors[i],
        wireframe: true,
        transparent: true,
        opacity: 0.7,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      const m = new THREE.Mesh(geo, mat);
      m.position.set((i - 1) * 14, 9 + i * 2.5, -22 - i * 14);
      scene.add(m);
      holos.push(m);
    }

    // ── Rising data motes ─────────────────────────────────────────────────
    const MOTES = 220;
    const motePos = new Float32Array(MOTES * 3);
    const moteCol = new Float32Array(MOTES * 3);
    const moteSpeeds = new Float32Array(MOTES);
    const moteTints = [colA.value, colC.value, colD.value];
    for (let i = 0; i < MOTES; i++) {
      motePos[i * 3] = (Math.random() - 0.5) * 70;
      motePos[i * 3 + 1] = -2 + Math.random() * 14;
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
      scene.add(mesh);
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

      gridMat.uniforms.uTime.value = t;
      sun.quaternion.copy(camera.quaternion);
      halo.quaternion.copy(camera.quaternion);

      // Motes rise and wrap.
      const pos = motes.geometry.getAttribute("position") as THREE.BufferAttribute;
      for (let i = 0; i < MOTES; i++) {
        let y = pos.getY(i) + moteSpeeds[i] * dt;
        if (y > 15) y = -2;
        pos.setY(i, y);
      }
      pos.needsUpdate = true;

      // Holograms rotate + bob.
      for (let i = 0; i < holos.length; i++) {
        holos[i].rotation.y += dt * 0.5;
        holos[i].rotation.x += dt * 0.2;
        holos[i].position.y = 9 + i * 2.5 + Math.sin(t * 0.8 + i * 2) * 0.7;
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
          scene.remove(ring.mesh);
          ring.mesh.geometry.dispose();
          (ring.mesh.material as THREE.Material).dispose();
          rings.splice(i, 1);
        }
      }

      if (!reduced) updateAmbient(dt, t);

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
      disposables.forEach((t) => t.dispose());
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
