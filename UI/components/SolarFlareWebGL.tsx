"use client";

// ── Solar Flare — the living-sun world for the "solar-flare" palette ──────
//
// Self-contained WebGL background modeled on CosmosWebGL.tsx's skeleton. Scene:
// a granulated fbm sun surface with limb darkening, billboarded corona sprites,
// helical prominence particle arcs, a radial solar-wind field, faint background
// stars, and click-triggered flare eruptions (localized jet burst + screen
// flash). Colors read live from --palette-* so the PaletteEditor retunes the
// star in real time.

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { supportsWebGL } from "./webglSupport";
import { readPaletteColors, makePalettePoller } from "./worldUtils";
import { emitCosmosEvent } from "./cosmosEvents";

// ── Sun surface shader: layered fbm granulation + limb darkening ──────────
const SUN_VERT = `
varying vec3 vNormal;
varying vec3 vWorld;
void main() {
  vNormal = normalize(normalMatrix * normal);
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorld = wp.xyz;
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

const SUN_FRAG = `
uniform float uTime;
uniform vec3 uColorA;
uniform vec3 uColorB;
uniform vec3 uColorC;
varying vec3 vNormal;
varying vec3 vWorld;
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
void main() {
  float n = fbm(vWorld * 1.7 + uTime * 0.12);
  // Bright granulation, darker sunspots where noise dips.
  vec3 col = mix(uColorA, uColorB, n);
  col *= 1.0 - smoothstep(0.30, 0.05, n) * 0.85;
  // Gentle living flicker.
  col *= 0.93 + 0.07 * sin(uTime * 3.0 + n * 24.0);
  // Limb darkening toward the edge.
  vec3 V = normalize(cameraPosition - vWorld);
  float ndv = abs(dot(normalize(vNormal), V));
  col *= mix(0.5, 1.0, smoothstep(0.0, 0.9, ndv));
  // Hot edge glow.
  col += uColorB * smoothstep(0.9, 1.0, ndv) * 0.25;
  gl_FragColor = vec4(col, 1.0);
}
`;

// Radial spiky corona texture (tinted per-material).
function makeCoronaTexture(): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const ctx = c.getContext("2d")!;
  const g = ctx.createRadialGradient(128, 128, 20, 128, 128, 126);
  g.addColorStop(0, "rgba(255,255,255,0.9)");
  g.addColorStop(0.55, "rgba(255,255,255,0.35)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 256, 256);
  // Radial streaks.
  for (let i = 0; i < 26; i++) {
    const a = Math.random() * Math.PI * 2;
    const x0 = 128 + Math.cos(a) * 42;
    const y0 = 128 + Math.sin(a) * 42;
    const x1 = 128 + Math.cos(a) * (122 + Math.random() * 20);
    const y1 = 128 + Math.sin(a) * (122 + Math.random() * 20);
    ctx.strokeStyle = `rgba(255,255,255,${0.08 + Math.random() * 0.14})`;
    ctx.lineWidth = 2 + Math.random() * 5;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  }
  return c;
}

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
    host.appendChild(renderer.domElement);
    renderer.domElement.style.position = "fixed";
    renderer.domElement.style.inset = "0";
    renderer.domElement.style.display = "block";

    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x0a0503, 40, 120);
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

    // ── The sun ───────────────────────────────────────────────────────────
    const sunMat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uColorA: colA,
        uColorB: colC,
        uColorC: colB,
      },
      vertexShader: SUN_VERT,
      fragmentShader: SUN_FRAG,
    });
    const sun = new THREE.Mesh(new THREE.SphereGeometry(SUN_R, 64, 48), sunMat);
    scene.add(sun);

    const coronaTex = new THREE.CanvasTexture(makeCoronaTexture());
    disposables.push(coronaTex);
    const coronaMat = new THREE.MeshBasicMaterial({
      map: coronaTex,
      color: colA.value,
      transparent: true,
      opacity: 0.6,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });
    const corona = new THREE.Mesh(new THREE.PlaneGeometry(26, 26), coronaMat);
    scene.add(corona);
    const corona2 = new THREE.Mesh(
      new THREE.PlaneGeometry(20, 20),
      new THREE.MeshBasicMaterial({
        map: coronaTex,
        color: colD.value,
        transparent: true,
        opacity: 0.4,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
      }),
    );
    scene.add(corona2);

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
    });
    const proms = new THREE.Points(promGeo, promMat);
    scene.add(proms);
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
    });
    const wind = new THREE.Points(windGeo, windMat);
    scene.add(wind);

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
    });
    const jets = new THREE.Points(jetGeo, jetMat);
    scene.add(jets);

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
    scene.add(flash);
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

      sunMat.uniforms.uTime.value = t;
      corona.quaternion.copy(camera.quaternion);
      corona2.quaternion.copy(camera.quaternion);
      corona.rotation.z += dt * 0.05;
      corona2.rotation.z -= dt * 0.07;
      corona2.scale.setScalar(1 + 0.04 * Math.sin(t * 0.6));

      updateProms(dt);
      updateWind(dt);
      updateJets(dt);
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
