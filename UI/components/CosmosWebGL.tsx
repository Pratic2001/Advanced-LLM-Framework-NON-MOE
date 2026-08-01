"use client";

/**
 * Deep-space background — a faithful port of the "Event Horizon" reference
 * (event_horizon.html), including its latest improvements:
 *
 *   · Global colour-mood drift: the whole rig (disk, photon ring, halo bands,
 *     glow sprites, grade) slowly breathes between a warm gold/amber plasma
 *     and a searing blue-white one over a 95 s cycle (updatePalette + uPalette).
 *   · A black event-horizon sphere with a fresnel photon-ring rim upgraded
 *     with a hairline photon ring (pow 40), camera-relative Doppler beaming
 *     and a warm↔blue palette blend.
 *   · A 20-layer volumetric accretion disk — a stack of independently seeded,
 *     vertically-offset puffy torus sheets (seat profile, per-layer density,
 *     braided strands, temperature cells, camera-relative beaming) so it reads
 *     as a genuinely thick glowing torus from every angle, including edge-on.
 *   · Four nested polar-ring halo bands (makeHaloBand) wrapping the shadow in
 *     warm→blue lensed arcs, each with its own flow speed and Doppler beaming.
 *   · 9000 twinkling stars + seven colourful nebulae in a slowly-rotating
 *     background group, so stars visibly sweep behind the hole and get bent.
 *   · Meteor showers & comets: occasional streaks of light flying through the
 *     skyEventsGroup. Meteor showers fan a burst of short additive streaks
 *     out of a shared radiant; comets are rarer and slower with a glowing coma,
 *     a long streak, and a wispy particle trail (custom aSize/aAlpha points
 *     shader). All live in `scene`, so they pick up the same lensing bend as
 *     the stars when they pass near the hole.
 *   · Supernovae: asymmetric particle ejecta (bipolar jet, squash, point-
 *     symmetric clump pairs, per-particle speed → colour-age) PLUS a genuine
 *     3D spherical iris (not a camera-facing billboard) whose fragment shader
 *     wraps one of five random remnant species over the sphere's surface and
 *     whose vertices are noise-displaced per-explosion. Ambient big blasts
 *     every 9–15 s, plus any number detonated by clicking empty sky. Each
 *     blast feeds a `uFlash` into the post pass.
 *   · A real-geodesic post pass: every pixel's camera ray is integrated as an
 *     actual bent photon path in Schwarzschild spacetime (RK4), so the shadow,
 *     the Einstein ring and the lensed images of the disk — and of any live
 *     hypernova, treated as a real sphere obstruction along the ray — all
 *     emerge from the trace itself. Plus a thin proximity Einstein ring +
 *     near-white core, radial chromatic aberration, a 512-tap wide-radius
 *     bloom, vignette, filmic tone shaping, dual-tone grade, saturation lift
 *     and film grain.
 *
 * Controls (as requested, everything else matches the reference):
 *   · left click on empty sky  → supernova
 *   · right mouse + drag       → orbit the camera angle
 *   · shift + scroll           → dolly zoom in/out
 *   · plain mouse move         → gentle parallax
 * Interactive page elements are left untouched. `prefers-reduced-motion` stops
 * the camera auto-drift.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { emitCosmosEvent } from "./cosmosEvents";

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

      {/* Black hole — large dominant orb, centred (matches the WebGL composition) */}
      <div
        style={{
          position: "absolute", left: "50%", top: "50%", width: "72vmin", height: "72vmin",
          borderRadius: "50%", transform: "translate(-50%,-50%) rotateX(60deg)",
          background:
            "radial-gradient(circle, rgba(255,200,120,0.9) 0%, rgba(255,120,40,0.55) 22%, rgba(40,20,10,0.65) 45%, rgba(0,0,0,0.95) 62%, transparent 72%)",
          animation: "cfBHPulse 5s ease-in-out infinite",
        }}
      />

      {/* Supernova — just off-centre, far field */}
      <div
        style={{
          position: "absolute", left: "76%", top: "22%", width: "38vmin", height: "38vmin",
          borderRadius: "50%",
          background:
            "radial-gradient(circle, rgba(255,255,255,0.95) 0%, rgba(165,200,255,0.6) 20%, rgba(120,160,255,0.25) 45%, transparent 70%)",
          animation: "cfSn 9s ease-in-out infinite",
        }}
      />
    </div>
  );
}

function supportsWebGL(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

/* ────────────────── Faithful "Event Horizon" scene port ─────────────── */

interface Supernova {
  points: THREE.Points;
  geo: THREE.BufferGeometry;
  mat: THREE.ShaderMaterial;
  iris: THREE.Mesh;
  irisMat: THREE.ShaderMaterial;
  velocities: Float32Array;
  positions: Float32Array;
  birth: number;
  life: number;
  big: boolean;
  // Hue the ejecta cools toward (teal/violet/magenta) — the palette pulse
  // follows the flash with a softer afterglow of this colour.
  eventHue: number;
  afterglowSent: boolean;
}

export function CosmosWebGL({ className = "" }: { className?: string }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [webglOk] = useState(() => supportsWebGL());
  const [broken, setBroken] = useState(false);

  useEffect(() => {
    if (broken) return;
    const host = hostRef.current;
    if (!host) return;

    let renderer: THREE.WebGLRenderer;
    let raf = 0;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        powerPreference: "high-performance",
      });
    } catch {
      try {
        renderer = new THREE.WebGLRenderer({
          antialias: false,
          powerPreference: "default",
          failIfMajorPerformanceCaveat: false,
        });
      } catch {
        setBroken(true);
        return;
      }
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 1);
    renderer.autoClear = true;
    const cv = renderer.domElement;
    cv.style.position = "fixed";
    cv.style.inset = "0";
    cv.style.display = "block";
    host.appendChild(cv);

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      48,
      window.innerWidth / window.innerHeight,
      0.1,
      800,
    );
    const clock = new THREE.Clock();
    const orbitAxisWorld = new THREE.Vector3(0, 0, 1);

    // ---- camera orbit state (spherical, target = origin) ----
    const cam = {
      azimuth: 0.55, polar: 1.25, radius: 34,
      targetAzimuth: 0.55, targetPolar: 1.25, targetRadius: 34,
      autoDrift: reduced ? 0 : 0.028,
    };
    let mouseDriftX = 0;

    function updateCamera(dt: number) {
      cam.targetAzimuth += cam.autoDrift * dt * (orbitDragging ? 0 : 1);
      cam.azimuth += (cam.targetAzimuth - cam.azimuth) * Math.min(1, dt * 3.2);
      cam.polar += (cam.targetPolar - cam.polar) * Math.min(1, dt * 3.2);
      cam.radius += (cam.targetRadius - cam.radius) * Math.min(1, dt * 3.2);
      const polar = Math.max(0.25, Math.min(Math.PI - 0.25, cam.polar));
      const breathe = Math.sin(clock.elapsedTime * 0.17) * 0.035;
      const radiusBreathe = cam.radius + Math.sin(clock.elapsedTime * 0.11) * 0.25;
      camera.position.set(
        radiusBreathe * Math.sin(polar + breathe) * Math.sin(cam.azimuth),
        radiusBreathe * Math.cos(polar + breathe),
        radiusBreathe * Math.sin(polar + breathe) * Math.cos(cam.azimuth),
      );
      camera.lookAt(0, 0, 0);
    }

    // ---- procedural radial-gradient textures ----
    const disposables: THREE.Texture[] = [];
    function makeRadialTexture(
      inner: number,
      outer: number,
      stops: [number, string][],
    ): THREE.CanvasTexture {
      const size = 256;
      const c = document.createElement("canvas");
      c.width = c.height = size;
      const ctx = c.getContext("2d")!;
      const grad = ctx.createRadialGradient(
        size / 2, size / 2, inner * size / 2,
        size / 2, size / 2, outer * size / 2,
      );
      stops.forEach((s) => grad.addColorStop(s[0], s[1]));
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, size, size);
      const tex = new THREE.CanvasTexture(c);
      tex.needsUpdate = true;
      disposables.push(tex);
      return tex;
    }

    const glowTexWarm = makeRadialTexture(0, 1, [
      [0, "rgba(255,235,210,1)"],
      [0.25, "rgba(255,190,120,0.75)"],
      [0.6, "rgba(255,120,60,0.22)"],
      [1, "rgba(255,80,40,0)"],
    ]);
    const glowTexBlue = makeRadialTexture(0, 1, [
      [0, "rgba(235,248,255,1)"],
      [0.25, "rgba(150,205,255,0.78)"],
      [0.6, "rgba(70,140,255,0.24)"],
      [1, "rgba(40,90,255,0)"],
    ]);

    // ----------------------------------------------------------------------
    // Global colour-mood drift: the whole rig (disk, photon ring, halo bands,
    // glow sprites, grade) slowly breathes between the warm gold/amber mood of
    // a cooler accretion flow and a searing blue-white superheated one, so the
    // simulation never settles into one static "look" — same GR physics, two
    // very different plasma temperatures, drifting into each other over minutes.
    // ----------------------------------------------------------------------
    const palette = { value: 0.0 };
    const PALETTE_PERIOD = 95; // seconds for a full warm<->hot cycle
    function updatePalette(t: number) {
      palette.value = 0.5 + 0.5 * Math.sin((t / PALETTE_PERIOD) * Math.PI * 2 - Math.PI * 0.5);
    }
    const COLOR_WARM = new THREE.Color(0xffb46a);
    const COLOR_HOT = new THREE.Color(0x9fd2ff);

    // Shared unit sphere for the supernova "iris" — a genuine 3D shell, not a
    // camera-facing card. The fragment shader maps its whole radiant-shell
    // pattern onto this sphere's own surface direction, so the explosion has
    // real volume and reads correctly from any viewing angle instead of always
    // presenting the same flat cutout.
    const irisGeo = new THREE.SphereGeometry(1, 48, 32);

    // ---- starfield ----
    function createStarfield(count: number): THREE.Points {
      const positions = new Float32Array(count * 3);
      const colors = new Float32Array(count * 3);
      const sizes = new Float32Array(count);
      const phases = new Float32Array(count);

      const palette: [number, number, number][] = [
        [0.65, 0.75, 1.0], [1.0, 1.0, 1.0], [1.0, 0.92, 0.75],
        [0.75, 0.85, 1.0], [1.0, 0.8, 0.65],
      ];

      for (let i = 0; i < count; i++) {
        const r = 60 + Math.random() * 260;
        const theta = Math.acos(2 * Math.random() - 1);
        const phi = Math.random() * Math.PI * 2;
        positions[i * 3] = r * Math.sin(theta) * Math.cos(phi);
        positions[i * 3 + 1] = r * Math.cos(theta);
        positions[i * 3 + 2] = r * Math.sin(theta) * Math.sin(phi);

        const p = palette[(Math.random() * palette.length) | 0];
        const b = 0.6 + Math.random() * 0.6;
        colors[i * 3] = p[0] * b;
        colors[i * 3 + 1] = p[1] * b;
        colors[i * 3 + 2] = p[2] * b;

        sizes[i] = Math.random() * 1.6 + 0.4;
        phases[i] = Math.random();
      }

      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geo.setAttribute("aColor", new THREE.BufferAttribute(colors, 3));
      geo.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
      geo.setAttribute("aPhase", new THREE.BufferAttribute(phases, 1));

      const mat = new THREE.ShaderMaterial({
        uniforms: { uTime: { value: 0 } },
        vertexShader: `
          attribute float aSize;
          attribute vec3 aColor;
          attribute float aPhase;
          varying vec3 vColor;
          varying float vTwinkle;
          uniform float uTime;
          void main(){
            vColor = aColor;
            vTwinkle = 0.55 + 0.45*sin(uTime*1.6 + aPhase*6.2831);
            vec4 mv = modelViewMatrix * vec4(position,1.0);
            gl_PointSize = aSize * (340.0/-mv.z);
            gl_Position = projectionMatrix*mv;
          }
        `,
        fragmentShader: `
          varying vec3 vColor;
          varying float vTwinkle;
          void main(){
            vec2 uv = gl_PointCoord-0.5;
            float d = length(uv);
            float alpha = smoothstep(0.5,0.0,d);
            gl_FragColor = vec4(vColor*vTwinkle, alpha*vTwinkle);
          }
        `,
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      });

      return new THREE.Points(geo, mat);
    }

    // Everything in the deep background lives in this group so it can drift
    // independently of the camera — that relative motion is what makes the
    // gravitational lensing near the horizon read as *bending* rather than a
    // fixed distortion overlay.
    const bgGroup = new THREE.Group();

    const stars = createStarfield(9000);
    bgGroup.add(stars);

    // ---- distant colourful nebulae (additive sprites) ----
    const nebulaGroup = new THREE.Group();
    const nebulaColors: [number, string][] = [
      [0.5, "rgba(140,60,220,0.55)"],  // purple
      [0.5, "rgba(40,140,220,0.5)"],   // blue
      [0.5, "rgba(220,50,120,0.45)"],  // magenta
      [0.5, "rgba(230,120,40,0.4)"],   // amber
    ];
    function makeNebulaTexture(colorStr: string): THREE.CanvasTexture {
      return makeRadialTexture(0, 1, [
        [0, colorStr],
        [0.5, colorStr.replace(/[\d.]+\)$/, "0.15)")],
        [1, "rgba(0,0,0,0)"],
      ]);
    }
    for (let i = 0; i < 7; i++) {
      const c = nebulaColors[i % nebulaColors.length][1];
      const tex = makeNebulaTexture(c);
      const mat = new THREE.SpriteMaterial({
        map: tex,
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        opacity: 0.55,
      });
      const spr = new THREE.Sprite(mat);
      const r = 90 + Math.random() * 140;
      const theta = Math.acos(2 * Math.random() - 1);
      const phi = Math.random() * Math.PI * 2;
      spr.position.set(
        r * Math.sin(theta) * Math.cos(phi),
        r * Math.cos(theta) * 0.6,
        r * Math.sin(theta) * Math.sin(phi),
      );
      const s = 60 + Math.random() * 90;
      spr.scale.set(s, s, 1);
      spr.userData.driftSpeed = (Math.random() - 0.5) * 0.004;
      nebulaGroup.add(spr);
    }
    bgGroup.add(nebulaGroup);
    bgGroup.rotation.z = 0.3;
    scene.add(bgGroup);

    // ---- meteor showers & comets: occasional streaking light across deep space ----
    // These live directly in `scene` (not bgGroup) with an unrotated parent, so
    // a spawn-time world-space travel direction stays valid for the object's
    // whole lifetime without any per-frame matrix work — and since they're part
    // of the same render-to-texture pass as the stars, they pick up the same
    // gravitational-lensing bend from the post shader when they pass near the hole.
    const skyEventsGroup = new THREE.Group();
    scene.add(skyEventsGroup);

    function makeStreakTexture(): THREE.CanvasTexture {
      const w = 512, h = 128;
      const c = document.createElement("canvas");
      c.width = w;
      c.height = h;
      const ctx = c.getContext("2d")!;
      ctx.clearRect(0, 0, w, h);

      // A tapered needle — thin, transparent tail widening into a soft rounded
      // head — blurred so it reads as a glowing streak of light rather than a
      // hard-edged bar (a flat rectangle here is exactly what looks like a
      // "strip of paper" flying through the scene).
      ctx.filter = "blur(7px)";
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.quadraticCurveTo(w * 0.38, h * 0.5 - h * 0.05, w * 0.64, h * 0.5 - h * 0.15);
      ctx.lineTo(w * 0.86, h * 0.5 - h * 0.3);
      ctx.arc(w * 0.86, h / 2, h * 0.3, -Math.PI / 2, Math.PI / 2);
      ctx.lineTo(w * 0.64, h * 0.5 + h * 0.15);
      ctx.quadraticCurveTo(w * 0.38, h * 0.5 + h * 0.05, 0, h / 2);
      ctx.closePath();

      const grad = ctx.createLinearGradient(0, 0, w, 0);
      grad.addColorStop(0, "rgba(255,255,255,0)");
      grad.addColorStop(0.45, "rgba(255,255,255,0.22)");
      grad.addColorStop(0.8, "rgba(255,255,255,0.75)");
      grad.addColorStop(1, "rgba(255,255,255,1)");
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.filter = "none";

      const tex = new THREE.CanvasTexture(c);
      tex.needsUpdate = true;
      disposables.push(tex);
      return tex;
    }
    const streakTex = makeStreakTexture();

    function colorVec3(hex: number): THREE.Vector3 {
      const c = new THREE.Color(hex);
      return new THREE.Vector3(c.r, c.g, c.b);
    }

    // bright end of the texture (u≈1) leads; material.rotation aligns that edge
    // with the object's screen-space direction of travel every frame.
    const _sPos = new THREE.Vector3(), _sAhead = new THREE.Vector3();
    function orientStreak(sprite: THREE.Sprite, worldPos: THREE.Vector3, dirWorld: THREE.Vector3): void {
      _sPos.copy(worldPos).project(camera);
      _sAhead.copy(worldPos).addScaledVector(dirWorld, 3).project(camera);
      const aspect = window.innerWidth / window.innerHeight;
      const dx = (_sAhead.x - _sPos.x) * aspect;
      const dy = _sAhead.y - _sPos.y;
      if (dx * dx + dy * dy > 1e-9) sprite.material.rotation = Math.atan2(dy, dx);
    }

    function randOnSphere(r: number): THREE.Vector3 {
      const theta = Math.acos(2 * Math.random() - 1);
      const phi = Math.random() * Math.PI * 2;
      return new THREE.Vector3(
        r * Math.sin(theta) * Math.cos(phi),
        r * Math.cos(theta),
        r * Math.sin(theta) * Math.sin(phi),
      );
    }

    // ---- meteor showers — small bursts of quick, bright streaks that all fan
    // out from a shared "radiant" point, the way a real shower reads visually. ----
    interface Meteor {
      streak: THREE.Sprite;
      coma: THREE.Sprite;
      start: THREE.Vector3;
      dir: THREE.Vector3;
      length: number;
      duration: number;
      t: number;
      streakLen: number;
      streakWidth: number;
    }
    const meteors: Meteor[] = [];
    const meteorPalette: { tint: number; glow: THREE.Texture }[] = [
      { tint: 0xcfe6ff, glow: glowTexBlue }, // icy blue-white
      { tint: 0xffe3bd, glow: glowTexWarm }, // pale gold
      { tint: 0xe3d0ff, glow: glowTexBlue }, // soft violet
      { tint: 0xffffff, glow: glowTexBlue }, // pure white
    ];

    function spawnMeteorShower(): void {
      const radiant = randOnSphere(1).normalize();
      const helper = Math.abs(radiant.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
      const tangentA = new THREE.Vector3().crossVectors(radiant, helper).normalize();
      const tangentB = new THREE.Vector3().crossVectors(radiant, tangentA).normalize();
      const shellR = 65 + Math.random() * 45;
      const count = 8 + Math.floor(Math.random() * 10);

      for (let i = 0; i < count; i++) {
        const ang = Math.random() * Math.PI * 2;
        const spreadDir = tangentA.clone().multiplyScalar(Math.cos(ang)).addScaledVector(tangentB, Math.sin(ang));
        const start = radiant.clone().multiplyScalar(shellR).addScaledVector(spreadDir, Math.random() * 10);
        const pal = meteorPalette[(Math.random() * meteorPalette.length) | 0];

        const streak = new THREE.Sprite(new THREE.SpriteMaterial({
          map: streakTex, color: pal.tint, transparent: true, depthWrite: false,
          blending: THREE.AdditiveBlending, opacity: 0,
        }));
        const coma = new THREE.Sprite(new THREE.SpriteMaterial({
          map: pal.glow, color: pal.tint, transparent: true, depthWrite: false,
          blending: THREE.AdditiveBlending, opacity: 0,
        }));
        skyEventsGroup.add(streak, coma);

        meteors.push({
          streak, coma, start, dir: spreadDir,
          length: 26 + Math.random() * 22,
          duration: 0.5 + Math.random() * 0.55,
          t: -Math.random() * 1.1, // stagger within the burst
          streakLen: 10 + Math.random() * 7,
          streakWidth: 0.45 + Math.random() * 0.3,
        });
      }
    }

    function updateMeteors(dt: number): void {
      for (let i = meteors.length - 1; i >= 0; i--) {
        const m = meteors[i];
        m.t += dt;
        if (m.t < 0) continue;
        const p = m.t / m.duration;
        if (p >= 1) { skyEventsGroup.remove(m.streak, m.coma); meteors.splice(i, 1); continue; }

        const worldPos = m.start.clone().addScaledVector(m.dir, m.length * p);
        let alpha = 1;
        if (p < 0.15) alpha = p / 0.15;
        else if (p > 0.65) alpha = 1 - (p - 0.65) / 0.35;

        const sc = camera.position.distanceTo(worldPos) / 90;

        m.coma.position.copy(worldPos);
        m.coma.scale.set(1.5 * sc, 1.5 * sc, 1);
        m.coma.material.opacity = alpha * 0.85;

        const streakWorldLen = m.streakLen * sc;
        m.streak.position.copy(worldPos).addScaledVector(m.dir, -streakWorldLen / 2);
        m.streak.scale.set(streakWorldLen, m.streakWidth * sc, 1);
        m.streak.material.opacity = alpha * 0.9;
        orientStreak(m.streak, worldPos, m.dir);
      }
    }

    // ---- comets — rarer, slower, brighter: a glowing coma, a long streak, and
    // a wispy particle tail that trails rigidly behind the head (its offsets are
    // baked in world-space at spawn time since the flight path is a straight
    // line, so no per-frame recomputation is needed). ----
    interface Comet {
      streak: THREE.Sprite;
      coma: THREE.Sprite;
      trail: THREE.Points;
      start: THREE.Vector3;
      dir: THREE.Vector3;
      length: number;
      duration: number;
      t: number;
      streakLen: number;
      streakWidth: number;
    }
    const comets: Comet[] = [];
    const cometPalette: { tint: number; glow: THREE.Texture; hex: number }[] = [
      { tint: 0xffd9a3, glow: glowTexWarm, hex: 0xffcf96 }, // molten gold
      { tint: 0x9be8ff, glow: glowTexBlue, hex: 0x8be0ff }, // ice blue
      { tint: 0xc9ffe0, glow: glowTexBlue, hex: 0xa8ffdb }, // pale emerald
    ];

    function makeTrailMaterial(colorVec: THREE.Vector3): THREE.ShaderMaterial {
      return new THREE.ShaderMaterial({
        uniforms: { uColor: { value: colorVec }, uOpacity: { value: 1 } },
        vertexShader: `
          attribute float aSize;
          attribute float aAlpha;
          varying float vAlpha;
          void main(){
            vAlpha = aAlpha;
            vec4 mv = modelViewMatrix * vec4(position,1.0);
            gl_PointSize = aSize * (300.0/-mv.z);
            gl_Position = projectionMatrix*mv;
          }
        `,
        fragmentShader: `
          uniform vec3 uColor;
          uniform float uOpacity;
          varying float vAlpha;
          void main(){
            vec2 uv = gl_PointCoord-0.5;
            float d = length(uv);
            float a = smoothstep(0.5,0.0,d);
            gl_FragColor = vec4(uColor, a*vAlpha*uOpacity);
          }
        `,
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      });
    }

    function buildCometTrail(dirWorld: THREE.Vector3, colorVec: THREE.Vector3): THREE.Points {
      const N = 26;
      const positions = new Float32Array(N * 3);
      const sizes = new Float32Array(N);
      const alphas = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        positions[i * 3] = -dirWorld.x * i * 0.9 + (Math.random() - 0.5) * 0.6;
        positions[i * 3 + 1] = -dirWorld.y * i * 0.9 + (Math.random() - 0.5) * 0.6;
        positions[i * 3 + 2] = -dirWorld.z * i * 0.9 + (Math.random() - 0.5) * 0.6;
        sizes[i] = Math.max(0.3, 1.7 - i * 0.055);
        alphas[i] = Math.max(0, 1 - i / N);
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geo.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
      geo.setAttribute("aAlpha", new THREE.BufferAttribute(alphas, 1));
      return new THREE.Points(geo, makeTrailMaterial(colorVec));
    }

    function spawnComet(): void {
      const pal = cometPalette[(Math.random() * cometPalette.length) | 0];
      const colorVec = colorVec3(pal.hex);

      const start = randOnSphere(80 + Math.random() * 40);
      const helper = Math.abs(start.y) < 70 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
      const dir = new THREE.Vector3().crossVectors(start, helper).normalize();
      if (Math.random() < 0.5) dir.negate();

      const streak = new THREE.Sprite(new THREE.SpriteMaterial({
        map: streakTex, color: pal.tint, transparent: true, depthWrite: false,
        blending: THREE.AdditiveBlending, opacity: 0,
      }));
      const coma = new THREE.Sprite(new THREE.SpriteMaterial({
        map: pal.glow, color: pal.tint, transparent: true, depthWrite: false,
        blending: THREE.AdditiveBlending, opacity: 0,
      }));
      const trail = buildCometTrail(dir, colorVec);
      skyEventsGroup.add(streak, coma, trail);

      comets.push({
        streak, coma, trail, start, dir,
        length: 150 + Math.random() * 60,
        duration: 9 + Math.random() * 5,
        t: 0,
        streakLen: 22 + Math.random() * 9,
        streakWidth: 1.0 + Math.random() * 0.45,
      });
    }

    function updateComets(dt: number): void {
      for (let i = comets.length - 1; i >= 0; i--) {
        const c = comets[i];
        c.t += dt;
        const p = c.t / c.duration;
        if (p >= 1) { skyEventsGroup.remove(c.streak, c.coma, c.trail); comets.splice(i, 1); continue; }

        const worldPos = c.start.clone().addScaledVector(c.dir, c.length * p);
        let alpha = 1;
        if (p < 0.08) alpha = p / 0.08;
        else if (p > 0.82) alpha = 1 - (p - 0.82) / 0.18;

        const sc = camera.position.distanceTo(worldPos) / 90;

        c.coma.position.copy(worldPos);
        c.coma.scale.set(6 * sc, 6 * sc, 1);
        c.coma.material.opacity = alpha * 0.95;

        const streakWorldLen = c.streakLen * sc;
        c.streak.position.copy(worldPos).addScaledVector(c.dir, -streakWorldLen / 2);
        c.streak.scale.set(streakWorldLen, c.streakWidth * sc, 1);
        c.streak.material.opacity = alpha * 0.85;
        orientStreak(c.streak, worldPos, c.dir);

        c.trail.position.copy(worldPos);
        (c.trail.material as THREE.ShaderMaterial).uniforms.uOpacity.value = alpha;
      }
    }

    // ---- black hole: event horizon, photon rim, disk, halo ----
    const bhGroup = new THREE.Group();
    scene.add(bhGroup);

    const R_EH = 1.0;

    const horizonGeo = new THREE.SphereGeometry(R_EH, 64, 64);
    const horizonMat = new THREE.MeshBasicMaterial({ color: 0x000000 });
    const horizon = new THREE.Mesh(horizonGeo, horizonMat);
    bhGroup.add(horizon);

    // Photon-ring rim glow (fresnel, wide halo + bright core + hairline
    // photon ring, with camera-relative Doppler beaming and palette blend).
    const rimGeo = new THREE.SphereGeometry(R_EH * 1.035, 64, 64);
    const rimMat = new THREE.ShaderMaterial({
      uniforms: {
        uPalette: { value: 0 },
        uOrbitAxis: { value: new THREE.Vector3(0, 0, 1) },
      },
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vViewPos;
        varying vec3 vWorldPos;
        void main(){
          vNormal = normalize(normalMatrix*normal);
          vec4 mv = modelViewMatrix*vec4(position,1.0);
          vViewPos = mv.xyz;
          vWorldPos = (modelMatrix*vec4(position,1.0)).xyz;
          gl_Position = projectionMatrix*mv;
        }
      `,
      fragmentShader: `
        uniform float uPalette;
        uniform vec3 uOrbitAxis;
        varying vec3 vNormal;
        varying vec3 vViewPos;
        varying vec3 vWorldPos;
        void main(){
          vec3 viewDir = normalize(-vViewPos);
          float f = max(dot(viewDir,vNormal),0.0);
          float fresnel = pow(1.0-f, 4.5);
          float core = pow(1.0-f, 12.0);
          // A hairline-thin, high-contrast ring right at the shadow's own
          // silhouette edge — the sharp feature EHT images are actually
          // named for, distinct from the softer bloomy fresnel glow above it.
          float photonRing = pow(1.0-f, 40.0);

          // Same camera-relative beaming as the disk and halo bands, so the
          // photon ring itself reads with the asymmetric brightness real
          // EHT ring images show rather than glowing evenly all the way
          // around.
          vec3 radial = normalize(vWorldPos);
          vec3 vel = normalize(cross(uOrbitAxis, radial));
          vec3 viewDirW = normalize(cameraPosition - vWorldPos);
          float beam = dot(vel, viewDirW);
          float doppler = clamp(0.5 + 0.5*beam, 0.0, 1.0);
          float beamMod = mix(0.4, 2.2, doppler);

          vec3 rimTint = mix(vec3(1.0,0.86,0.62), vec3(0.72,0.9,1.0), uPalette);
          vec3 coreTint = mix(vec3(1.0,0.97,0.92), vec3(0.93,0.98,1.0), uPalette);
          vec3 glowColor = mix(rimTint, coreTint, core);
          glowColor = mix(glowColor, vec3(1.3,0.7,0.4)*max(glowColor.r,max(glowColor.g,glowColor.b)), smoothstep(0.45,0.0,doppler)*0.5);
          glowColor = mix(glowColor, vec3(0.75,0.9,1.35)*max(glowColor.r,max(glowColor.g,glowColor.b)), smoothstep(0.55,1.0,doppler)*0.5);
          vec3 photonCol = mix(vec3(1.0,0.5,0.3), vec3(0.85,0.95,1.4), doppler);

          vec3 finalCol = glowColor*(fresnel*2.0 + core*3.2) + photonCol*photonRing*3.0*beamMod;
          gl_FragColor = vec4(finalCol, min(1.0, fresnel*1.3 + core + photonRing*beamMod));
        }
      `,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    const rim = new THREE.Mesh(rimGeo, rimMat);
    bhGroup.add(rim);

    // Ambient warm halo — two layers pulsing out of phase so the glow itself
    // feels alive rather than a flat sprite.
    const haloMat = new THREE.SpriteMaterial({
      map: glowTexWarm,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      opacity: 0.9,
    });
    const halo = new THREE.Sprite(haloMat);
    halo.scale.set(9, 9, 1);
    bhGroup.add(halo);

    const haloMat2 = new THREE.SpriteMaterial({
      map: glowTexWarm,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      opacity: 0.35,
    });
    const halo2 = new THREE.Sprite(haloMat2);
    halo2.scale.set(13, 13, 1);
    bhGroup.add(halo2);

    // Second pair of sprites carrying the hot-blue texture, cross-faded against
    // the warm pair above via opacity so the glow itself migrates smoothly
    // between moods rather than just tinting (tinting alone can't reach the
    // icy blue-white of a superheated flow from an amber-only texture).
    const haloMatB = new THREE.SpriteMaterial({
      map: glowTexBlue,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      opacity: 0.0,
    });
    const haloB = new THREE.Sprite(haloMatB);
    haloB.scale.set(9, 9, 1);
    bhGroup.add(haloB);

    const haloMatB2 = new THREE.SpriteMaterial({
      map: glowTexBlue,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      opacity: 0.0,
    });
    const haloB2 = new THREE.Sprite(haloMatB2);
    haloB2.scale.set(13, 13, 1);
    bhGroup.add(haloB2);

    // Accretion disk — built as a genuine 3D torus, not a single flat sheet.
    // A lone displaced ring, however wavy, still collapses to a hairline
    // when the camera lines up exactly edge-on, because a wavy *surface* has
    // no actual volume — it's still infinitely thin in cross-section. To get
    // a disk with real thickness at every viewing angle (edge-on included),
    // we stack several copies of the same turbulent ring at different
    // vertical offsets, each independently seeded and tapered thinnest at
    // the disk's own thickness profile, so together they read as a solid
    // glowing torus with real depth rather than a flat printed record.
    const R_IN = R_EH * 1.9, R_OUT = R_EH * 8.5;
    const diskGeo = new THREE.RingGeometry(R_IN, R_OUT, 200, 28);
    const DISK_VSHADER = `
        uniform float uTime;
        uniform float uInner;
        uniform float uOuter;
        uniform float uYOffset;
        uniform float uLayerSeed;
        varying vec3 vPos;
        varying vec3 vWorldPos;
        varying float vShade;
        varying float vRim;

        float hash(vec2 p){ return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453123); }
        float noise(vec2 p){
          vec2 i=floor(p), f=fract(p);
          float a=hash(i), b=hash(i+vec2(1.0,0.0)), c=hash(i+vec2(0.0,1.0)), d=hash(i+vec2(1.0,1.0));
          vec2 u=f*f*(3.0-2.0*f);
          return mix(a,b,u.x) + (c-a)*u.y*(1.0-u.x) + (d-b)*u.x*u.y;
        }
        float fbm(vec2 p){
          float v=0.0, amp=0.5;
          for(int i=0;i<4;i++){ v += amp*noise(p); p *= 2.05; amp *= 0.55; }
          return v;
        }
        // Vertical (out-of-plane) height of the disk surface at a given local
        // (x,y) — this is what turns a flat ring into an actual puffy torus
        // of turbulent plasma instead of a painted-on record. uLayerSeed
        // offsets the noise field per-layer so the stack doesn't look like
        // identical sheets rigidly copy-pasted on top of each other.
        float heightAt(vec2 pos){
          float r = length(pos);
          float rn = clamp((r-uInner)/(uOuter-uInner), 0.0, 1.0);
          float angle = atan(pos.y, pos.x);
          vec2 c = vec2(cos(angle),sin(angle)) * (2.0+rn*3.0) + vec2(uTime*0.16 + uLayerSeed*4.1, r*0.35 - uTime*mix(1.6,0.25,rn) + uLayerSeed*6.3);
          float n = fbm(c*1.4);
          float profile = pow(sin(clamp(rn,0.1,0.98)*3.14159265), 0.55); // full-bodied near the inner edge, only pinches thin right at the outer rim
          return (n-0.5) * mix(0.55, 0.16, rn) * profile;
        }

        void main(){
          vec2 xy = position.xy;
          float h = heightAt(xy);
          float eps = 0.08;
          float hx = heightAt(xy+vec2(eps,0.0));
          float hy = heightAt(xy+vec2(0.0,eps));
          vec3 n = normalize(vec3(-(hx-h)/eps, -(hy-h)/eps, 1.0));

          // This layer's fixed vertical seat within the torus's cross-section,
          // tapered by the same thin-at-both-edges profile as the turbulent
          // wobble above — so the whole stack pinches down to a point at the
          // disk's inner and outer radii (a proper torus tube) rather than
          // extending as a flat-topped slab all the way to the edges.
          float r0 = length(xy);
          float rn0 = clamp((r0-uInner)/(uOuter-uInner), 0.0, 1.0);
          float seatProfile = pow(sin(clamp(rn0,0.1,0.98)*3.14159265), 0.55);

          vec3 displaced = vec3(xy, h + uYOffset*seatProfile);
          vPos = displaced;
          vWorldPos = (modelMatrix * vec4(displaced,1.0)).xyz;

          vec3 lightDir = normalize(vec3(0.35, 0.55, 0.75));
          vShade = clamp(dot(n, lightDir), 0.12, 1.0);

          vec3 viewNormal = normalize(normalMatrix * n);
          vec4 mvPos = modelViewMatrix * vec4(displaced,1.0);
          vec3 viewDir = normalize(-mvPos.xyz);
          // Limb brightening: material seen edge-on through more of the puffy
          // disk's depth scatters more light back at us, the way a real
          // optically-thick torus rim-lights rather than looking paper-flat.
          vRim = pow(1.0-clamp(abs(dot(viewDir, viewNormal)),0.0,1.0), 2.4);

          gl_Position = projectionMatrix*mvPos;
        }
      `;
    const DISK_FSHADER = `
        uniform float uTime;
        uniform float uInner;
        uniform float uOuter;
        uniform float uPalette;
        uniform float uDensity;
        uniform vec3 uOrbitAxis;
        varying vec3 vPos;
        varying vec3 vWorldPos;
        varying float vShade;
        varying float vRim;

        float hash(vec2 p){ return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453123); }
        float noise(vec2 p){
          vec2 i=floor(p), f=fract(p);
          float a=hash(i), b=hash(i+vec2(1.0,0.0)), c=hash(i+vec2(0.0,1.0)), d=hash(i+vec2(1.0,1.0));
          vec2 u=f*f*(3.0-2.0*f);
          return mix(a,b,u.x) + (c-a)*u.y*(1.0-u.x) + (d-b)*u.x*u.y;
        }
        float fbm(vec2 p){
          float v=0.0, amp=0.5;
          for(int i=0;i<5;i++){ v += amp*noise(p); p *= 2.05; amp *= 0.55; }
          return v;
        }

        void main(){
          float r = length(vPos.xy);
          float rn = clamp((r-uInner)/(uOuter-uInner), 0.0, 1.0);
          float angle = atan(vPos.y, vPos.x);

          float angSpeed = mix(9.0, 0.5, rn);
          float flowAngle = angle + uTime*angSpeed;
          vec2 noiseCoord = vec2(cos(flowAngle), sin(flowAngle)) * (2.0 + rn*3.0) + vec2(uTime*0.35, uTime*0.5 + r*0.4);
          float turb = fbm(noiseCoord*2.2 + uTime*0.15);
          float turb2 = fbm(noiseCoord*5.5 - uTime*0.4);
          float turbulence = mix(turb, turb2, 0.45);
          turbulence = pow(turbulence, 0.8);

          // Braided concentric strands: several thin sinusoidal streams offset
          // in radius and phase, each riding its own bit of turbulence — the
          // "ribbon of separate glowing threads" look of the reference
          // photography rather than one smooth gradient.
          float strandFreq = mix(26.0, 34.0, rn);
          float strand = 0.0;
          strand += 0.5+0.5*sin(rn*strandFreq*6.2831 + turb*4.0 + uTime*0.6);
          strand += (0.5+0.5*sin(rn*strandFreq*6.2831*1.7 + 2.1 + turb2*5.0 - uTime*0.4))*0.6;
          strand /= 1.6;
          strand = pow(strand, 1.8);

          float flicker = 0.85 + 0.3*sin(uTime*3.0 + r*6.0 + turb*8.0) * (1.0-rn*0.5);

          // Local "temperature" isn't purely a function of distance — hot and
          // cool turbulent cells interleave at the same radius the way real
          // accretion-disk plasma does, and the whole field slowly drifts in
          // time so the color mix is never static.
          float tempNoise = fbm(noiseCoord*1.3 - uTime*0.06 + 11.0);
          float coolNoise = fbm(noiseCoord*3.4 + uTime*0.09 + 5.0);
          float temp = clamp(rn*1.15 - (tempNoise-0.5)*0.5 + (coolNoise-0.5)*0.18, 0.0, 1.0);

          // Two plasma moods — a cooler warm-gold flow and a searing blue-white
          // one — blended by uPalette so the whole disk's temperature identity
          // can drift over time without changing the underlying structure.
          vec3 whiteHot = mix(vec3(1.0, 1.0, 0.99),  vec3(1.0, 1.0, 1.0),  uPalette);
          vec3 hot      = mix(vec3(1.0, 0.93, 0.82), vec3(0.85, 0.95, 1.0), uPalette);
          vec3 gold     = mix(vec3(1.0, 0.78, 0.38), vec3(0.55, 0.78, 1.0), uPalette);
          vec3 amber    = mix(vec3(1.0, 0.5, 0.12),  vec3(0.28, 0.55, 1.0), uPalette);
          vec3 rust     = mix(vec3(0.86, 0.24, 0.05),vec3(0.16, 0.32, 0.92), uPalette);
          vec3 deepRed  = mix(vec3(0.5, 0.07, 0.03), vec3(0.08, 0.14, 0.55), uPalette);
          vec3 dustBlue = mix(vec3(0.35, 0.5, 0.85), vec3(0.55, 0.3, 0.9), uPalette);

          vec3 col = mix(whiteHot, hot, smoothstep(0.0,0.12,temp));
          col = mix(col, gold, smoothstep(0.08,0.28,temp));
          col = mix(col, amber, smoothstep(0.22,0.5,temp));
          col = mix(col, rust, smoothstep(0.42,0.72,temp));
          col = mix(col, deepRed, smoothstep(0.66,1.0,temp));
          // Cool dusty knots in the outer, turbulent-cool cells — the faint
          // accent threads visible weaving through a real disk's outer reaches.
          float dustMask = smoothstep(0.55,1.0,rn) * smoothstep(0.35,0.7,coolNoise);
          col = mix(col, dustBlue*0.9, dustMask*0.35);

          col *= (0.4 + 1.3*turbulence) * flicker;
          col *= (0.55 + 0.75*strand);

          // Relativistic beaming: material sweeping toward the camera is
          // Doppler-boosted and blue-shifted; the receding side dims and
          // reddens. This is now genuinely camera-relative rather than a
          // time-rotating fake — orbital velocity at this point (perpendicular
          // to the radius, in the disk's own rotation plane) is dotted
          // against the actual direction to the viewer, so the bright limb
          // tracks wherever the camera currently is, and the effect
          // naturally fades toward neutral when looking straight down the
          // rotation axis (face-on), exactly as it should.
          vec3 radial = normalize(vWorldPos);
          vec3 vel = normalize(cross(uOrbitAxis, radial));
          vec3 viewDirW = normalize(cameraPosition - vWorldPos);
          float beam = dot(vel, viewDirW);
          float doppler = clamp(0.5 + 0.5*beam, 0.0, 1.0);
          float dopplerSharp = pow(doppler, 1.6);
          col *= mix(0.32, 2.6, dopplerSharp);
          col = mix(col, vec3(0.75,0.85,1.05)*length(col), smoothstep(0.55,1.0,doppler)*0.28);
          col = mix(col, vec3(1.15,0.55,0.28)*length(col), smoothstep(0.45,0.0,doppler)*0.22);

          // Fake volumetric lighting from the puffy-surface normal, plus a
          // warm limb-brightened rim — this is what reads as an actual body
          // of turbulent plasma instead of a flat printed disc.
          col *= mix(0.55, 1.4, vShade);
          col += vec3(1.0,0.82,0.55) * vRim * 0.6;

          float edgeFade = smoothstep(0.0,0.08,rn) * smoothstep(1.0,0.8,rn);
          float alpha = edgeFade * (0.6 + 0.6*turbulence) * flicker + vRim*0.25*edgeFade;

          gl_FragColor = vec4(col, clamp(alpha,0.0,1.4) * uDensity);
        }
      `;

    // A small stack of layers spread through the torus's vertical extent.
    // Density (and therefore visible brightness/alpha) tapers off toward the
    // top and bottom the way a real optically-thick plasma torus's density
    // falls off away from its mid-plane, so the cross-section reads as a
    // soft, rounded glowing tube rather than a stack of hard-edged sheets —
    // this is what finally gives the disk actual, real 3D depth: viewed dead
    // edge-on it now shows as a genuinely thick glowing band, not a line.
    const DISK_LAYER_OFFSETS = [-0.967165, -0.640324, -0.596942, -0.544649, -0.533872, -0.398153, -0.278787, -0.275193, -0.274899, -0.193609, -0.027831, 0.110460, 0.127309, 0.208466, 0.231603, 0.319092, 0.494153, 0.807735, 0.906307, 0.999499];
    const diskGroup = new THREE.Group();
    const diskMats: THREE.ShaderMaterial[] = [];
    DISK_LAYER_OFFSETS.forEach((yOff, i) => {
      const density = Math.exp(-Math.pow(yOff / 0.62, 2.0)) * (yOff === 0 ? 1.0 : 0.62);
      const mat = new THREE.ShaderMaterial({
        uniforms: {
          uTime: { value: 0 },
          uInner: { value: R_IN },
          uOuter: { value: R_OUT },
          uPalette: { value: 0 },
          uYOffset: { value: yOff * R_IN * 1.0 },
          uLayerSeed: { value: i * 7.3 + 1.0 },
          uDensity: { value: density },
          uOrbitAxis: { value: new THREE.Vector3(0, 0, 1) },
        },
        vertexShader: DISK_VSHADER,
        fragmentShader: DISK_FSHADER,
        transparent: true,
        side: THREE.DoubleSide,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      const mesh = new THREE.Mesh(diskGeo, mat);
      diskGroup.add(mesh);
      diskMats.push(mat);
    });
    diskGroup.rotation.x = Math.PI * 0.42;
    diskGroup.rotation.z = 0.15;
    bhGroup.add(diskGroup);
    const diskMat = diskMats[Math.floor(diskMats.length / 2)]; // the yOff=0 mid-plane layer

    // Secondary lensed rings — an approximation of the disk's light bent all
    // the way over the poles of the hole (the classic "Gargantua halo"
    // silhouette from the reference photography: several nested, differently
    // coloured arcs of light wrapping the shadow, brightest and bluest near
    // the horizon and cooling outward into gold, rust and violet). Each band
    // is a torus perpendicular to the equatorial disk, so as a full ring it
    // reads as a halo wrapping the sphere from every camera angle — and each
    // flows and drifts independently so the nested arcs never look painted-on.
    const polarRingMats: THREE.ShaderMaterial[] = [];
    function makeHaloBand(
      radiusF: number,
      tubeF: number,
      warmA: string,
      warmB: string,
      hotA: string,
      hotB: string,
      speed: number,
      brightness: number,
    ) {
      const geo = new THREE.TorusGeometry(R_EH * radiusF, R_EH * tubeF, 20, 220);
      const mat = new THREE.ShaderMaterial({
        uniforms: {
          uTime: { value: 0 },
          uPalette: { value: 0 },
          uWarmA: { value: new THREE.Color(warmA) },
          uWarmB: { value: new THREE.Color(warmB) },
          uHotA: { value: new THREE.Color(hotA) },
          uHotB: { value: new THREE.Color(hotB) },
          uSpeed: { value: speed },
          uBright: { value: brightness },
          uOrbitAxis: { value: new THREE.Vector3(0, 0, 1) },
        },
        vertexShader: `
          varying vec2 vUvV;
          varying vec3 vWorldPos;
          void main(){
            vUvV = uv;
            vWorldPos = (modelMatrix * vec4(position,1.0)).xyz;
            gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0);
          }
        `,
        fragmentShader: `
          uniform float uTime, uPalette, uSpeed, uBright;
          uniform vec3 uWarmA, uWarmB, uHotA, uHotB;
          uniform vec3 uOrbitAxis;
          varying vec2 vUvV;
          varying vec3 vWorldPos;
          float hash(float n){ return fract(sin(n)*43758.5453123); }
          float hash2(vec2 p){ return fract(sin(dot(p,vec2(41.3,289.1)))*43758.5453123); }
          float noise2(vec2 p){
            vec2 i=floor(p), f=fract(p);
            float a=hash2(i), b=hash2(i+vec2(1.0,0.0)), c=hash2(i+vec2(0.0,1.0)), d=hash2(i+vec2(1.0,1.0));
            vec2 u=f*f*(3.0-2.0*f);
            return mix(a,b,u.x) + (c-a)*u.y*(1.0-u.x) + (d-b)*u.x*u.y;
          }
          void main(){
            float ang = vUvV.x*6.2831 + uTime*uSpeed;
            float flow  = sin(ang*3.0)*0.5+0.5;
            float flow2 = sin(ang*7.0 - uTime*uSpeed*1.4)*0.5+0.5;
            float fine  = noise2(vec2(ang*9.0 + uTime*uSpeed*0.6, vUvV.y*4.0 - uTime*0.1));
            float turb = mix(flow, flow2, 0.5);
            turb = mix(turb, fine, 0.35);
            vec3 warm = mix(uWarmA, uWarmB, turb);
            vec3 hot  = mix(uHotA,  uHotB,  turb);
            vec3 col = mix(warm, hot, uPalette) * (0.7+0.7*turb);

            // This halo is lensed light from the same rotating disk, so it
            // carries the same approaching/receding beaming asymmetry that
            // the direct disk view does — one side of the ring reads
            // brighter and bluer, the other dimmer and redder, matching the
            // asymmetric brightness real EHT ring images show rather than
            // an evenly-lit torus of light.
            vec3 radial = normalize(vWorldPos);
            vec3 vel = normalize(cross(uOrbitAxis, radial));
            vec3 viewDirW = normalize(cameraPosition - vWorldPos);
            float beam = dot(vel, viewDirW);
            float doppler = clamp(0.5 + 0.5*beam, 0.0, 1.0);
            col *= mix(0.45, 1.9, doppler);
            col = mix(col, vec3(0.8,0.92,1.2)*max(col.r,max(col.g,col.b)), smoothstep(0.55,1.0,doppler)*0.35);
            col = mix(col, vec3(1.25,0.6,0.3)*max(col.r,max(col.g,col.b)), smoothstep(0.45,0.0,doppler)*0.3);

            float edge = smoothstep(0.0,0.3,vUvV.y) * smoothstep(1.0,0.7,vUvV.y);
            float alpha = edge * (0.4 + 0.4*turb) * uBright * mix(0.7,1.3,doppler);
            gl_FragColor = vec4(col*uBright, alpha);
          }
        `,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.rotation.y = Math.PI * 0.5;
      bhGroup.add(mesh);
      polarRingMats.push(mat);
      return mesh;
    }
    // Innermost — brightest, near-white/cyan, fastest flow (light grazing
    // closest to the photon sphere is the most energetic and time-dilated).
    makeHaloBand(1.10, 0.045, "#fff3d8", "#ffd9a0", "#eaf6ff", "#bfe6ff", 2.2, 1.3);
    // Mid-inner — gold shifting to sky blue.
    makeHaloBand(1.22, 0.07, "#ffcd7a", "#ff9a3d", "#9fd0ff", "#5fa0ff", 1.5, 1.0);
    // Mid-outer — amber/rust shifting to mid blue.
    makeHaloBand(1.37, 0.09, "#ff7a2e", "#c94a12", "#4d7dff", "#2b4fdd", 1.0, 0.75);
    // Outermost — cooling into violet/purple on both palettes, dimmest, slowest.
    makeHaloBand(1.55, 0.10, "#8a4fe0", "#4a2a90", "#6a5fe0", "#3a2a8f", 0.6, 0.5);

    // ---- supernovae: asymmetric particles + genuine 3D spherical iris ----
    const supernovae: Supernova[] = [];
    let flashLevel = 0;

    function makeSupernova(pos: THREE.Vector3, big: boolean): Supernova {
      const count = big ? 2048 : 900;
      const positions = new Float32Array(count * 3);
      const velocities = new Float32Array(count * 3);
      const sizes = new Float32Array(count);
      const rands = new Float32Array(count);
      const speeds = new Float32Array(count);

      // ---- Asymmetric ejecta rig — real core-collapse blasts are never
      // isotropic: a bipolar jet axis (the collapsing core's rotation axis)
      // punches material out much faster along one line, a handful of
      // Rayleigh-Taylor "clumps" outrun the smoother envelope because denser
      // knots feel less drag, and the whole ejecta cloud is mildly oblate/
      // prolate rather than a perfect sphere of directions. All three biases
      // are applied to the initial velocity field below, so the spark cloud
      // itself reads as lopsided ejecta instead of a uniform starburst.
      const jetAxis = new THREE.Vector3(
        Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5,
      ).normalize();
      const jetBoost = 1.6 + Math.random() * 1.6;
      const jetNarrow = 3.0 + Math.random() * 4.0;
      const squashAxis = new THREE.Vector3(
        Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5,
      ).normalize();
      const squashAmt = 0.15 + Math.random() * 0.3;

      // Point-symmetric dominant pair — recent Cas A analysis finds its
      // outer clumps line up in pairs directly opposite each other through
      // the explosion's center, consistent with a jittering/precessing jet
      // rather than scattered independent knots. One strong pair, sharing
      // an axis but boosted independently, anchors the shape; the rest of
      // the clumps stay weaker background structure so the pair actually
      // reads as the dominant feature instead of blending in.
      const pairAxis = new THREE.Vector3(
        Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5,
      ).normalize();
      const pairWidth = 0.1 + Math.random() * 0.09;
      const clumps: { dir: THREE.Vector3; width: number; strength: number }[] = [
        {
          dir: pairAxis.clone(),
          width: pairWidth,
          strength: 1.5 + Math.random() * 1.0,
        },
        {
          dir: pairAxis.clone().negate(),
          width: pairWidth,
          strength: (1.5 + Math.random() * 1.0) * (0.65 + Math.random() * 0.3),
        },
      ];
      const numClumps = 3 + Math.floor(Math.random() * 4);
      for (let c = 0; c < numClumps; c++) {
        clumps.push({
          dir: new THREE.Vector3(
            Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5,
          ).normalize(),
          width: 0.15 + Math.random() * 0.35,
          strength: 0.35 + Math.random() * 0.6,
        });
      }

      let maxSpeed = 0.0001;
      const dirV = new THREE.Vector3();
      for (let i = 0; i < count; i++) {
        const theta = Math.acos(2 * Math.random() - 1);
        const phi = Math.random() * Math.PI * 2;
        dirV.set(
          Math.sin(theta) * Math.cos(phi),
          Math.cos(theta),
          Math.sin(theta) * Math.sin(phi),
        );

        // oblate/prolate flattening along a random axis — breaks the perfect
        // sphere-of-directions before speed biasing even starts
        const sAlign = dirV.dot(squashAxis);
        dirV.addScaledVector(squashAxis, -sAlign * squashAmt).normalize();

        let speed = (2.0 + Math.random() * 6.0) * (big ? 1.6 : 1.0);

        // bipolar jet — sharply boosts speed near the jet axis (both ends)
        const jAlign = Math.abs(dirV.dot(jetAxis));
        speed *= 1.0 + jetBoost * Math.pow(jAlign, jetNarrow);

        // Rayleigh-Taylor clumps — a few denser fingers punch further than
        // the smooth surrounding ejecta, both individually (extra speed) and
        // collectively (extra particles land near a clump direction purely
        // because many independent draws share that speed boost)
        for (const cl of clumps) {
          const cAlign = Math.max(0, dirV.dot(cl.dir));
          speed *= 1.0 + cl.strength * Math.pow(cAlign, 1.0 / cl.width);
        }

        const dx = dirV.x, dy = dirV.y, dz = dirV.z;
        positions[i * 3] = pos.x;
        positions[i * 3 + 1] = pos.y;
        positions[i * 3 + 2] = pos.z;
        velocities[i * 3] = dx * speed;
        velocities[i * 3 + 1] = dy * speed;
        velocities[i * 3 + 2] = dz * speed;
        sizes[i] = Math.random() * 2.2 + 0.6;
        rands[i] = Math.random();
        speeds[i] = speed;
        if (speed > maxSpeed) maxSpeed = speed;
      }
      // Fast outrunning knots vs. the smoother bulk shell — normalize each
      // particle's initial speed into 0..1 so the fragment shader can make
      // the real fast clumps (the ones that actually outran the envelope,
      // like Cas A's 14,500 km/s knots against its ~5,000 km/s bulk shell)
      // stay visibly hot and ionized longer than the slower material at the
      // very same clock age, instead of every spark cooling in lockstep.
      const speedNorm = new Float32Array(count);
      for (let i = 0; i < count; i++) speedNorm[i] = speeds[i] / maxSpeed;

      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geo.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
      geo.setAttribute("aRand", new THREE.BufferAttribute(rands, 1));
      geo.setAttribute("aSpeed", new THREE.BufferAttribute(speedNorm, 1));

      const mat = new THREE.ShaderMaterial({
        uniforms: {
          uAge: { value: 0 },
          uTime: { value: 0 },
          uTint: {
            value: new THREE.Color(
              0.75 + Math.random() * 0.5,
              0.75 + Math.random() * 0.5,
              0.75 + Math.random() * 0.5,
            ),
          },
        },
        vertexShader: `
          attribute float aSize;
          attribute float aRand;
          attribute float aSpeed;
          varying float vRand;
          varying float vSpeed;
          uniform float uAge;
          void main(){
            vRand = aRand;
            vSpeed = aSpeed;
            vec4 mv = modelViewMatrix*vec4(position,1.0);
            float sizeFade = mix(1.4, 0.15, uAge);
            gl_PointSize = aSize*sizeFade*(420.0/-mv.z);
            gl_Position = projectionMatrix*mv;
          }
        `,
        fragmentShader: `
          varying float vRand;
          varying float vSpeed;
          uniform float uAge;
          uniform float uTime;
          uniform vec3 uTint;
          void main(){
            vec2 uv = gl_PointCoord-0.5;
            float d = length(uv);
            float alpha = smoothstep(0.5,0.0,d);
            vec3 hotc  = vec3(1.0,1.0,0.94);
            vec3 midc  = mix(vec3(1.0,0.72,0.32), vec3(0.55,0.78,1.0), vRand) * uTint;
            vec3 coolc = mix(vec3(0.55,0.22,0.08), vec3(0.22,0.4,0.85), vRand) * uTint;
            // Fast outrunning knots stay hot/ionized longer than the smoother
            // bulk shell at the very same clock age — real remnants show
            // exactly this split (Cas A's ~14,500 km/s knots against its
            // ~5,000 km/s bulk shell), rather than every spark cooling in
            // lockstep. Overall lifetime fade below still runs on the real
            // uAge so despawn timing is untouched.
            float colorAge = clamp(uAge * mix(1.55, 0.6, vSpeed), 0.0, 1.0);
            vec3 col = mix(hotc, midc, smoothstep(0.0,0.4,colorAge));
            col = mix(col, coolc, smoothstep(0.35,1.0,colorAge));
            float fade = 1.0 - smoothstep(0.42, 1.0, uAge);
            float sparkle = 0.75 + 0.35*sin(uTime*18.0 + vRand*60.0);
            gl_FragColor = vec4(col*sparkle, alpha*fade);
          }
        `,
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      });

      const points = new THREE.Points(geo, mat);
      scene.add(points);

      // The iris — a genuine 3D shell (a real sphere, not a camera-facing
      // card) whose fragment shader grows fine radial fibers out from a
      // bright core, the way a real supernova's ejecta forms filamentary
      // structure rather than fireworks sparks. It's given a fixed random
      // 3D orientation at spawn (see iris.rotation below) and never re-faces
      // the camera afterward, so orbiting around a live explosion reveals
      // actual volume and parallax instead of a flat cutout. Everything
      // (shape, color, growth) is procedural, mapped from the sphere's own
      // surface direction, so the silhouette always closes smoothly at both
      // poles — there is no flat edge to expose no matter the angle.
      const irisMat = new THREE.ShaderMaterial({
        uniforms: {
          uAge: { value: 0 },
          uTime: { value: 0 },
          uSeed: { value: Math.random() * 1000 },
          uPalette: { value: 0 },
          // Per-explosion randomization — every hypernova gets its own hue
          // drift on each band, its own band widths/positions, and its own
          // relative band weights, so no two ignitions settle into quite the
          // same "eye" even though the underlying stages (flash, collapse,
          // ring, iris, cloud) are always the same physics.
          uHueRing: { value: (Math.random() - 0.5) * 0.10 },
          uHueIris: { value: (Math.random() - 0.5) * 0.36 },
          uHueOuter: { value: (Math.random() - 0.5) * 0.30 },
          uSatJitter: { value: 0.85 + Math.random() * 0.3 },
          uRingW: { value: 0.08 + Math.random() * 0.07 },
          uIrisOut: { value: 0.52 + Math.random() * 0.20 },
          uOuterOut: { value: 0.86 + Math.random() * 0.12 },
          uW1: { value: 0.75 + Math.random() * 0.55 },
          uW2: { value: 0.75 + Math.random() * 0.55 },
          uW3: { value: 0.75 + Math.random() * 0.55 },
          // Per-explosion silhouette warp — a random stretch axis plus a
          // random strength so the overall outline is an irregular, lopsided
          // blast front rather than a concentric-circle "eye". Slowly rotates
          // over the explosion's life so the asymmetry itself drifts, the way
          // a real expanding shock front keeps reshaping rather than holding
          // a fixed non-circular-but-static outline.
          uAnisoAngle: { value: Math.random() * Math.PI * 2 },
          uAnisoAmt: { value: 0.16 + Math.random() * 0.26 },
          uAnisoSpin: { value: (Math.random() - 0.5) * 0.05 },
          // Which remnant "species" this explosion settles into — real
          // hypernovae don't all relax into the same radiant-iris shape,
          // some stay clumpy and one-sided (Cas A-like), some stay a smooth
          // rounded shell with a single jet (Tycho-like), some end up a
          // broken, gapped loop, some end up an elongated barrel with side
          // filaments (W49B-like). Picked randomly per explosion; every
          // species still draws its palette from the hue uniforms above so
          // color randomization applies across all of them, not just one.
          uDesign: { value: Math.floor(Math.random() * 5) },
          uMaxR: { value: 1.0 },
          uPlumeAngle: { value: Math.random() * Math.PI * 2 },
          uPlumeAmt: { value: 0.18 + Math.random() * 0.42 },
          // Aligned with the particle cloud's dominant jet-pair axis (see
          // pairAxis above) rather than an independent random angle, so the
          // shell's own bulge and the spark cloud's fastest ejecta agree on
          // which direction the jet actually points.
          uJetAngle: { value: Math.atan2(pairAxis.y, pairAxis.x) },
          uGapAngle: { value: Math.random() * Math.PI * 2 },
          uGapWidth: { value: 0.35 + Math.random() * 0.55 },
          uKnotAngle: { value: Math.random() * Math.PI * 2 },
        },
        transparent: true, depthWrite: false, depthTest: true,
        blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
        vertexShader: `
          varying vec3 vDir;
          uniform float uAge, uSeed, uTime;
          uniform float uAnisoAngle, uAnisoAmt, uAnisoSpin;
          uniform float uPlumeAngle, uPlumeAmt, uJetAngle, uGapAngle, uGapWidth, uDesign;

          // Cheap 3D value-noise/fbm so the shock front itself can be
          // displaced per-vertex, in true 3D, with no pole singularity —
          // this is what actually breaks the perfect-sphere silhouette,
          // since the SphereGeometry this mesh starts from is otherwise a
          // true sphere no matter how the fragment shader colors it.
          float vhash(vec3 p){ return fract(sin(dot(p,vec3(127.1,311.7,74.7)))*43758.5453123); }
          float vnoise(vec3 p){
            vec3 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
            float n000=vhash(i), n100=vhash(i+vec3(1,0,0));
            float n010=vhash(i+vec3(0,1,0)), n110=vhash(i+vec3(1,1,0));
            float n001=vhash(i+vec3(0,0,1)), n101=vhash(i+vec3(1,0,1));
            float n011=vhash(i+vec3(0,1,1)), n111=vhash(i+vec3(1,1,1));
            float nx00=mix(n000,n100,f.x), nx10=mix(n010,n110,f.x);
            float nx01=mix(n001,n101,f.x), nx11=mix(n011,n111,f.x);
            return mix(mix(nx00,nx10,f.y), mix(nx01,nx11,f.y), f.z);
          }
          float vfbm(vec3 p){
            float v=0.0, amp=0.5;
            for(int i=0;i<4;i++){ v += amp*vnoise(p); p*=2.15; amp*=0.55; }
            return v;
          }

          void main(){
            // Position on a unit sphere IS its own direction from center —
            // this replaces the old billboard's flat UV coordinate with a
            // genuine 3D direction, so the fragment shader below can wrap
            // its whole pattern fully around a real volume.
            vec3 dir = normalize(position);
            vDir = dir;

            // ---- Non-spherical shock-front growth. A real hypernova shell
            // is lumpy and lopsided from the instant it leaves the star:
            // Rayleigh-Taylor instabilities tear the front into fingers at
            // every scale, any jet/torus asymmetry from the progenitor
            // persists and grows, and the front never re-rounds itself back
            // into a sphere as it expands. Displace every vertex along its
            // own radial direction using low+mid+high frequency 3D noise
            // (bumps at multiple scales at once, not one soft wobble), plus
            // the same jet / plume / gap axes the fragment shader already
            // uses so the geometric silhouette and the surface texture agree.
            float growAge = 0.15 + uAge*0.85;
            float nLow  = vfbm(dir*1.6 + uSeed);
            float nMid  = vfbm(dir*4.2 + uSeed*1.7 + 30.0);
            float nHigh = vfbm(dir*9.5 + uSeed*3.1 + 70.0);
            float bump = (nLow-0.5)*0.6 + (nMid-0.5)*0.35*growAge + (nHigh-0.5)*0.18*growAge;

            // elliptical stretch — same rotating axis as the fragment
            // shader's aniso warp, so the lopsided outline it already paints
            // is backed by an actually-lopsided mesh
            float aa = uAnisoAngle + uTime*uAnisoSpin;
            vec2 dxy = vec2(dot(dir.xy, vec2(cos(aa),sin(aa))), dot(dir.xy, vec2(-sin(aa),cos(aa))));
            float stretch = 1.0 + uAnisoAmt*0.6*(dxy.x*dxy.x - dxy.y*dxy.y);

            // bipolar jet growth boost along its axis
            vec2 jetDir = vec2(cos(uJetAngle), sin(uJetAngle));
            float jetAlign = max(0.0, dot(normalize(dir.xy+1e-4), jetDir));
            float jetBoost = pow(jetAlign, 6.0) * (0.5+uPlumeAmt) * growAge;

            // gap pinch — for the broken-loop species, thins the shell right
            // where the ring is meant to gap open
            vec2 gapDir = vec2(cos(uGapAngle), sin(uGapAngle));
            float gapAlign = max(0.0, dot(normalize(dir.xy+1e-4), gapDir));
            float gapPinch = (uDesign > 2.5 && uDesign < 3.5) ? pow(gapAlign, 4.0)*0.35 : 0.0;

            float growth = clamp(1.0 + bump*0.9 + jetBoost - gapPinch, 0.35, 1.9) * stretch;

            vec3 displaced = dir * growth;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
          }
        `,
        fragmentShader: `
          uniform float uAge, uTime, uSeed, uPalette;
          uniform float uHueRing, uHueIris, uHueOuter, uSatJitter;
          uniform float uRingW, uIrisOut, uOuterOut, uW1, uW2, uW3;
          uniform float uAnisoAngle, uAnisoAmt, uAnisoSpin;
          uniform float uDesign, uMaxR, uPlumeAngle, uPlumeAmt, uJetAngle;
          uniform float uGapAngle, uGapWidth, uKnotAngle;
          varying vec3 vDir;

          // Minimal RGB<->HSV round trip so each band's base color can be hue-
          // rotated per instance while keeping the same brightness structure.
          vec3 rgb2hsv(vec3 c){
            vec4 K = vec4(0.0,-1.0/3.0,2.0/3.0,-1.0);
            vec4 p = mix(vec4(c.bg,K.wz), vec4(c.gb,K.xy), step(c.b,c.g));
            vec4 q = mix(vec4(p.xyw,c.r), vec4(c.r,p.yzx), step(p.x,c.r));
            float d = q.x-min(q.w,q.y);
            float e = 1.0e-10;
            return vec3(abs(q.z+(q.w-q.y)/(6.0*d+e)), d/(q.x+e), q.x);
          }
          vec3 hsv2rgb(vec3 c){
            vec4 K = vec4(1.0,2.0/3.0,1.0/3.0,3.0);
            vec3 p = abs(fract(c.xxx+K.xyz)*6.0-K.www);
            return c.z * mix(K.xxx, clamp(p-K.xxx,0.0,1.0), c.y);
          }
          vec3 hueShift(vec3 col, float amt, float satMul){
            vec3 hsv = rgb2hsv(col);
            hsv.x = fract(hsv.x + amt);
            hsv.y = clamp(hsv.y*satMul, 0.0, 1.0);
            return hsv2rgb(hsv);
          }

          float hash(vec2 p){ return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453123); }
          float noise(vec2 p){
            vec2 i=floor(p), f=fract(p);
            float a=hash(i), b=hash(i+vec2(1.0,0.0)), c=hash(i+vec2(0.0,1.0)), d=hash(i+vec2(1.0,1.0));
            vec2 u=f*f*(3.0-2.0*f);
            return mix(a,b,u.x) + (c-a)*u.y*(1.0-u.x) + (d-b)*u.x*u.y;
          }
          float fbm(vec2 p){
            float v=0.0, amp=0.5;
            for(int i=0;i<6;i++){ v += amp*noise(p); p *= 2.13; amp *= 0.55; }
            return v;
          }

          void main(){
            // True spherical silhouette: treat the fragment's direction on
            // the unit sphere as spherical coordinates around the mesh's own
            // local Z axis (its "pole", fixed at a random 3D orientation
            // when the explosion spawned — see iris.rotation in JS). theta
            // is 0 at the pole and PI at the antipode, so theta/PI is a
            // radius-like value that is *always* bounded to [0,1] by the
            // geometry itself — unlike a flat quad's UV, there is no literal
            // physical edge it can prematurely hit or expose as a hard line.
            vec3 dir = normalize(vDir);
            float theta = acos(clamp(dir.z, -1.0, 1.0));
            float ang0  = atan(dir.y, dir.x);
            // Reuse the original 2D warp/band math unchanged below by
            // feeding it a unit-disc proxy coordinate built from (theta,
            // ang0) — same lopsided, slowly-spinning silhouette as before,
            // just wrapped fully around a real 3D shell instead of stamped
            // onto a camera-facing card, so every band, ring, jet and plume
            // now genuinely propagates through 3D space and reads correctly
            // from any viewing angle, including face-on to the pole.
            vec2 p = vec2(cos(ang0), sin(ang0)) * (theta / 3.14159265);

            // Warp into a randomly-oriented, slowly-spinning elliptical frame
            // before doing any radial math. This alone breaks the "perfect
            // iris" symmetry: every downstream band (core, ring, iris, outer
            // cloud) inherits a lopsided, drifting silhouette instead of
            // concentric circles, because r and ang are now measured in the
            // warped frame, not screen space.
            float anisoAng = uAnisoAngle + uTime*uAnisoSpin;
            float aca = cos(anisoAng), asa = sin(anisoAng);
            vec2 pa = vec2(p.x*aca + p.y*asa, -p.x*asa + p.y*aca);
            pa.x *= (1.0 + uAnisoAmt);
            pa.y *= (1.0 - uAnisoAmt*0.55);
            vec2 pw = vec2(pa.x*aca - pa.y*asa, pa.x*asa + pa.y*aca);

            float r = length(pw);
            if(r > uMaxR) discard;
            float ang = atan(pw.y, pw.x);
            float TAU = 6.2831853;
            // angular distance from a to b, wrapped into [0, PI]
            #define ADIST(a,b) abs(mod((a)-(b)+3.14159265, TAU)-3.14159265)

            // Fine radial striations — the feather-like filament structure
            // running through the iris band and the fibrous ring right at the
            // collapsed core's edge, the way real ionized ejecta forms fine
            // threads rather than a smooth gradient.
            float fibers     = fbm(vec2(ang*46.0 + uSeed*11.0, r*3.2 - uTime*0.05));
            float fibers2    = fbm(vec2(ang*90.0 + uSeed*23.0, r*6.0 + uTime*0.08));
            float fineFibers = fbm(vec2(ang*160.0 - uSeed*17.0, r*11.0 + uTime*0.12));
            float texture1 = mix(fibers, fibers2, 0.5);
            texture1 = mix(texture1, fineFibers, 0.3);

            // Blotchy, uneven nebula texture for the outer cloud shell.
            float cloudy = fbm(vec2(ang*7.0 + uSeed*3.0 + uTime*0.04, r*4.0 - uTime*0.07));
            float clumpField = fbm(vec2(ang*11.0 + uSeed*2.3, r*3.6) - uTime*0.02);
            float rift = smoothstep(0.32,0.52,clumpField);
            float knot = smoothstep(0.62,0.88,clumpField);
            float speckle = pow(hash(floor(dir.xy*260.0 + dir.z*130.0) + uSeed), 24.0);

            // Organic, irregular band edges — real remnant shells are never
            // clean circles. Stack three octaves at different angular
            // frequencies — broad lobing, mid-scale scalloping, fine jagged
            // pokes — so the boundary reads as genuinely torn structure, the
            // way real ejecta fronts are uneven at every scale simultaneously,
            // not just a single soft wobble. The outer cloud gets an extra,
            // independently-seeded layer so its edge tears on its own rather
            // than staying perfectly concentric with the inner bands.
            float edgeLow  = fbm(vec2(ang*2.4  + uSeed*1.7, uTime*0.018)) - 0.5;
            float edgeMid  = fbm(vec2(ang*7.5  + uSeed*4.3, uTime*0.03 + 40.0)) - 0.5;
            float edgeHigh = fbm(vec2(ang*18.0 + uSeed*9.1, uTime*0.05 + 90.0)) - 0.5;
            float edgeNoise = edgeLow*0.85 + edgeMid*0.45 + edgeHigh*0.22;
            float edgeOuterExtra = fbm(vec2(ang*3.1 + uSeed*6.6, uTime*0.022 + 130.0)) - 0.5;
            float edgeNoiseOuter = edgeNoise + edgeOuterExtra*0.6;

            // ---- Ignition flash: a brilliant white-blue point with sharp
            // diffraction spikes and a thin horizontal streak, matching the
            // reference footage's first instant reading as an over-exposed
            // star rather than a nebula.
            float spikeAng  = abs(fract(ang/6.2831*6.0)-0.5)*2.0;
            float spike     = pow(1.0-spikeAng, 22.0) * pow(1.0-clamp(r,0.0,1.0), 0.6);
            float spikeAng2 = abs(fract((ang+0.5)/6.2831*4.0)-0.5)*2.0;
            float spike2    = pow(1.0-spikeAng2, 14.0) * pow(1.0-clamp(r,0.0,1.0), 0.4);
            float streak    = pow(1.0-abs(sin(ang)), 50.0) * (1.0-smoothstep(0.0,0.75,r));
            float ignite     = 1.0 - smoothstep(0.0,0.22,uAge);
            float igniteCore = pow(1.0-clamp(r,0.0,1.0), 2.2);
            vec3 igniteCol = hueShift(vec3(0.85,0.95,1.0), uHueIris*0.4, 1.0) * igniteCore * 2.2
                           + vec3(1.0,1.0,1.0) * (spike+spike2*0.7) * 1.6
                           + hueShift(vec3(0.8,0.9,1.0), uHueIris*0.4, 1.0) * streak * 1.8;
            float igniteAlpha = clamp(igniteCore*2.0 + spike + spike2 + streak, 0.0, 1.0);

            // ---- Remnant species: which overall structure this explosion
            // settles into. Real hypernovae don't all relax into the same
            // radiant-shell "eye" — some stay clumpy and one-sided, some
            // stay a smooth rounded cloud with a single jet, some end up a
            // broken gapped loop, some end up an elongated barrel with side
            // filaments. Picked randomly per explosion via uDesign; every
            // species still draws its palette from the hue uniforms above,
            // so color randomization always applies, whichever shape wins.
            vec3 col = vec3(0.0);
            float shellAlpha = 0.0;

            if(uDesign < 0.5){
              // ---- Species 0: refined radiant shell — concentric core/
              // ring/iris/outer bands, now layered with localized brightness
              // and hue patchiness (real shells never radiate perfectly
              // evenly) and an optional wispy plume punching past the
              // envelope.
              float coreGrow = smoothstep(0.1,0.42,uAge) * (1.0 - 0.35*smoothstep(0.55,1.0,uAge));
              float coreR = coreGrow*0.15 + edgeNoise*0.035;
              float coreMask = 1.0 - smoothstep(coreR-0.015, coreR+0.02, r);

              float ring1Out = coreR + uRingW + edgeNoise*0.06;
              float ring1 = smoothstep(coreR, coreR+0.02, r) * (1.0-smoothstep(ring1Out-0.05, ring1Out, r));
              vec3 ring1Col = hueShift(mix(vec3(1.0,0.97,0.85), vec3(1.0,0.6,0.22), fibers), uHueRing, uSatJitter);

              float irisOut = uIrisOut + edgeNoise*0.12;
              float irisBand = smoothstep(ring1Out-0.04, ring1Out+0.05, r) * (1.0-smoothstep(irisOut-0.08, irisOut, r));
              vec3 irisCol = mix(vec3(0.32,0.58,0.95), vec3(0.75,0.9,1.0), texture1);
              irisCol = mix(irisCol, vec3(0.9,0.96,1.0), pow(texture1,4.0)*0.5);
              irisCol = hueShift(irisCol, uHueIris, uSatJitter);

              float outerOut = clamp(uOuterOut + edgeNoiseOuter*0.17, 0.3, 0.97);
              float outerBand = smoothstep(irisOut-0.05, irisOut+0.06, r) * (1.0-smoothstep(outerOut-0.1, outerOut, r));
              vec3 outerCol = mix(vec3(1.0,0.55,0.2), vec3(0.5,0.16,0.07), cloudy);
              outerCol = mix(outerCol, vec3(0.3,0.45,0.78), smoothstep(0.6,0.9,cloudy)*0.3);
              outerCol = hueShift(outerCol, uHueOuter, uSatJitter);

              // Localized patchiness — breaks up the perfectly-even radiate
              // look band by band, both in brightness and in hue.
              float patchN = fbm(vec2(ang*3.3 + uSeed*5.1, r*2.1 - uTime*0.015));
              float patchMod = mix(0.55, 1.3, patchN);
              ring1Col = hueShift(ring1Col, (patchN-0.5)*0.05, 1.0);
              irisCol  = hueShift(irisCol,  (patchN-0.5)*0.05, 1.0);
              outerCol = hueShift(outerCol, (patchN-0.5)*0.05, 1.0);

              col += ring1Col * ring1 * (0.9 + 0.6*fineFibers) * uW1 * patchMod;
              col += irisCol * irisBand * (0.85 + 0.5*texture1) * uW2 * patchMod;
              col += outerCol * outerBand * (0.6 + 0.7*cloudy) * mix(1.0,0.35,rift) * uW3 * patchMod;
              col += outerCol * knot * outerBand * 0.4 * uW3;
              col += vec3(1.0,0.97,0.9) * speckle * (irisBand+outerBand);

              shellAlpha = clamp(ring1*uW1 + irisBand*uW2 + outerBand*uW3*0.9, 0.0, 1.0);
              shellAlpha *= mix(1.0, 0.4, rift*outerBand);
              shellAlpha = clamp(shellAlpha + speckle*0.4*(irisBand+outerBand), 0.0, 1.0);
              shellAlpha *= (1.0 - coreMask);

              // Wispy plume — some ignitions punch a filament out past the
              // main envelope, some don't; strength is randomized per blast.
              float dAngP = ADIST(ang, uPlumeAngle);
              float plumeWidth = 0.28 + uPlumeAmt*0.3;
              float plumeMask = pow(max(0.0, 1.0-dAngP/plumeWidth), 3.0);
              float plumeReach = outerOut + uPlumeAmt*(uMaxR-outerOut);
              float plumeRad = smoothstep(outerOut-0.05, outerOut+0.02, r) * (1.0-smoothstep(plumeReach-0.05, plumeReach+0.05, r));
              float plumeNoise = fbm(vec2(ang*20.0+uSeed*7.7, r*8.0 - uTime*0.1));
              float plumeVal = plumeMask*plumeRad*smoothstep(0.3,0.85,plumeNoise);
              col += hueShift(outerCol, 0.04, 1.0) * plumeVal * 1.3;
              shellAlpha = clamp(shellAlpha + plumeVal, 0.0, 1.0);

            } else if(uDesign < 1.5){
              // ---- Species 1: clumpy patchwork remnant — no clean
              // concentric rings, just irregular color zones, a fine bright
              // filament net laid over the top, and one prominent one-sided
              // horn of ejecta punching outward, the way an off-center blast
              // wave dumps most of its energy in one direction.
              float zoneA = fbm(vec2(ang*1.6 + uSeed*2.1,      r*1.3 - uTime*0.01));
              float zoneB = fbm(vec2(ang*1.9 + uSeed*5.3+50.0, r*1.5 + uTime*0.012));
              float zoneC = fbm(vec2(ang*1.4 + uSeed*8.7+90.0, r*1.1 - uTime*0.008));
              float wsum = zoneA+zoneB+zoneC+0.0001;
              float wA = zoneA/wsum, wB = zoneB/wsum, wC = zoneC/wsum;
              vec3 colA = hueShift(vec3(0.55,0.75,0.30), uHueRing,  uSatJitter);
              vec3 colB = hueShift(vec3(0.80,0.50,0.92), uHueIris,  uSatJitter);
              vec3 colC = hueShift(vec3(1.00,0.82,0.52), uHueOuter, uSatJitter);
              vec3 zoneCol = colA*wA + colB*wB + colC*wC;

              float outerOutD1 = clamp(uOuterOut*1.05 + edgeNoiseOuter*0.24, 0.4, uMaxR*0.98);
              float fillMask = 1.0 - smoothstep(outerOutD1-0.08, outerOutD1, r);
              float blotch = smoothstep(0.25,0.85, clumpField);
              fillMask *= mix(0.5, 1.0, blotch);
              fillMask *= mix(0.65, 1.0, smoothstep(0.0,0.3,r));

              float ridge = 1.0 - abs(fbm(vec2(ang*30.0+uSeed*13.0, r*9.0 - uTime*0.06))*2.0-1.0);
              float filament = pow(ridge, 10.0);
              vec3 filCol = hueShift(vec3(0.55,0.75,1.0), uHueRing*0.5, 1.0);

              col += zoneCol * fillMask * (0.7 + 0.5*texture1);
              col += filCol * filament * fillMask * 1.3;
              col += vec3(1.0,0.97,0.9) * speckle * fillMask * 0.8;
              shellAlpha = clamp(fillMask*0.95 + filament*0.6*fillMask + speckle*fillMask*0.5, 0.0, 1.0);

              float dAngP = ADIST(ang, uPlumeAngle);
              float plumeWidth = 0.34 + uPlumeAmt*0.35;
              float plumeMask = pow(max(0.0, 1.0-dAngP/plumeWidth), 2.5);
              float plumeReach = outerOutD1 + (0.35+uPlumeAmt*0.7)*(uMaxR-outerOutD1);
              float plumeRad = smoothstep(outerOutD1-0.06, outerOutD1+0.02, r) * (1.0-smoothstep(plumeReach-0.06, plumeReach+0.06, r));
              float plumeNoise = fbm(vec2(ang*16.0+uSeed*6.3, r*6.0 - uTime*0.08));
              float plumeVal = plumeMask*plumeRad*smoothstep(0.25,0.8,plumeNoise);
              col += colA * plumeVal * 1.1;
              shellAlpha = clamp(shellAlpha + plumeVal, 0.0, 1.0);

            } else if(uDesign < 2.5){
              // ---- Species 2: smooth rounded cloud with a thin bright rim
              // and a single curved jet — much less fine fiber texture than
              // the others, reads as a soft, blurred remnant rather than
              // filamentary.
              float outerOutD2 = clamp(uOuterOut*0.9 + edgeLow*0.16, 0.45, uMaxR*0.95);
              float fillMask = 1.0 - smoothstep(outerOutD2-0.07, outerOutD2, r);
              fillMask *= mix(0.7, 1.0, smoothstep(0.0,0.35,r));

              vec3 interior = mix(vec3(0.85,0.25,0.35), vec3(0.45,0.12,0.35), cloudy);
              interior = hueShift(interior, uHueOuter, uSatJitter);

              float rim = 1.0 - smoothstep(0.0,0.045,abs(r-outerOutD2));
              vec3 rimCol = hueShift(vec3(0.35,0.55,1.0), uHueRing, 1.15);

              float dAngJ = ADIST(ang, uJetAngle);
              float jetMask = pow(max(0.0, 1.0-dAngJ/0.085), 6.0);
              float jetOuter = mix(outerOutD2+0.18, uMaxR*0.96, 0.5+uPlumeAmt*0.5);
              float jetRad = smoothstep(outerOutD2-0.05, outerOutD2, r) * (1.0-smoothstep(jetOuter-0.05, jetOuter, r));
              vec3 jetCol = hueShift(vec3(1.0,0.95,0.6), uHueIris, 1.0);
              float jetVal = jetMask*jetRad;

              col += interior * fillMask * (0.55 + 0.4*cloudy);
              col += rimCol * rim * 1.5;
              col += jetCol * jetVal * 1.8;
              shellAlpha = clamp(fillMask*0.8 + rim*1.3 + jetVal, 0.0, 1.0);

            } else if(uDesign < 3.5){
              // ---- Species 3: broken, gapped loop — mostly hollow, energy
              // concentrated in a ring that doesn't fully close, one
              // hemisphere reading a different hue than the other, one
              // bright hot knot, and a faint diffuse halo cloud reaching
              // out past the ring itself.
              float ringMid = 0.64 + edgeLow*0.05;
              float ringHalfW = 0.15 + abs(edgeMid)*0.05;
              float ringShape = 1.0 - smoothstep(ringHalfW*0.55, ringHalfW, abs(r-ringMid));

              float dAngGap = ADIST(ang, uGapAngle);
              float gapFactor = smoothstep(uGapWidth*0.5, uGapWidth*0.5+0.35, dAngGap);
              ringShape *= mix(0.12, 1.0, gapFactor);

              float hemi = smoothstep(-0.2,0.6, sin(ang - uGapAngle - 1.2));
              vec3 ringColA = hueShift(vec3(0.25,0.55,1.0), uHueRing, uSatJitter);
              vec3 ringColB = hueShift(vec3(0.25,0.9,0.65), uHueIris, uSatJitter);
              vec3 ringCol = mix(ringColA, ringColB, hemi);
              ringCol = mix(ringCol, vec3(1.0), pow(texture1,5.0)*0.4);

              float dAngK = ADIST(ang, uKnotAngle);
              float knotMask = pow(max(0.0,1.0-dAngK/0.12),4.0) * pow(max(0.0,1.0-abs(r-ringMid)/(ringHalfW*1.3)),3.0);
              vec3 knotCol = hueShift(vec3(1.0,0.75,0.35), uHueOuter, 1.0);

              float haloMask = smoothstep(ringMid+ringHalfW*0.4, ringMid+ringHalfW*1.7, r) * (1.0-smoothstep(uMaxR*0.85,uMaxR,r));
              haloMask *= (0.2+0.5*cloudy);
              vec3 haloCol = hueShift(mix(vec3(0.5,0.08,0.05), vec3(0.7,0.15,0.05), cloudy), uHueOuter*0.6, 0.9);

              float interiorMask = (1.0-smoothstep(ringMid-ringHalfW*1.2, ringMid-ringHalfW*0.3, r)) * 0.16;
              vec3 interiorCol = hueShift(vec3(0.08,0.08,0.25), uHueRing*0.4, 0.8);

              col += ringCol * ringShape * (0.8+0.6*texture1);
              col += knotCol * knotMask * 1.7;
              col += haloCol * haloMask;
              col += interiorCol * interiorMask;
              shellAlpha = clamp(ringShape*0.95 + knotMask*1.2 + haloMask*0.7 + interiorMask, 0.0, 1.0);

            } else {
              // ---- Species 4: elongated barrel with red bracket filaments
              // on either side — a blue-green core cloud framed by thin red
              // arcs left and right, plus a handful of hot point knots.
              float outerOutD4 = clamp(uOuterOut*0.95 + edgeNoise*0.16, 0.4, uMaxR*0.95);
              float fillMask = 1.0 - smoothstep(outerOutD4-0.12, outerOutD4-0.02, r);
              fillMask *= mix(0.5, 1.0, smoothstep(0.15,0.6, fbm(vec2(ang*4.0+uSeed*3.3, r*3.0-uTime*0.02))));

              vec3 coreCol = mix(vec3(0.15,0.55,0.9), vec3(0.15,0.85,0.45), cloudy);
              coreCol = hueShift(coreCol, uHueIris, uSatJitter);

              float ringThin = (1.0-smoothstep(0.0,0.05,abs(r-outerOutD4))) * smoothstep(0.5,0.9,abs(cos(ang)));
              vec3 bracketCol = hueShift(vec3(0.9,0.2,0.15), uHueOuter, uSatJitter);

              col += coreCol * fillMask * (0.7+0.5*texture1);
              col += bracketCol * ringThin * 1.6;
              col += vec3(1.0,0.95,0.85) * speckle * fillMask * 0.9;
              shellAlpha = clamp(fillMask*0.9 + ringThin*1.3 + speckle*fillMask*0.6, 0.0, 1.0);
            }

            // ---- Blend ignition into the settled remnant, then fade the
            // whole thing in at birth and out at the end of its life.
            // Dust-lane occlusion — real remnants are never uniformly bright.
            // Cool, unshocked dust threaded through the shell blocks and dims
            // the glow behind it (JWST's Cas A imagery shows exactly this: a
            // dust-dominated sheet pockmarked with dark gaps over the bright
            // ejecta), so a coarse mask carves dark lanes and pockmarks
            // through the shell instead of leaving it smoothly lit everywhere.
            float dustLane = fbm(vec2(ang*5.0 + uSeed*4.4 + 200.0, r*3.4 - uTime*0.015));
            float dustMask = smoothstep(0.58,0.72,dustLane) * smoothstep(0.03, uMaxR*0.9, r);
            col *= mix(1.0, 0.12, dustMask);
            shellAlpha *= mix(1.0, 0.35, dustMask);

            vec3 col_final = mix(col, igniteCol, ignite);
            float alpha = mix(shellAlpha, igniteAlpha, ignite);
            alpha *= smoothstep(0.0, 0.03, uAge + 0.002);
            alpha *= (1.0 - smoothstep(0.6, 1.0, uAge));

            // ---- Rarefaction: a real blast shell conserves mass as it
            // expands, so its material thins out the further/longer it
            // travels — without this the shell just keeps looking like a
            // smooth, uniformly-dense skin inflating outward (a balloon)
            // instead of an expanding front that gets wispier with age.
            alpha *= mix(1.0, 0.45, smoothstep(0.12, 0.9, uAge));

            // ---- Porosity: punch actual holes through the shell wherever a
            // coarse, direction-consistent noise field dips low, so light
            // passes clean through in patches the way it does through a real
            // dust-and-debris cloud. Barely present during the initial flash
            // (kept solid and bright), then opens up steadily as the remnant
            // settles, so the transition itself reads as the skin breaking
            // apart into cloud rather than a balloon that simply grows.
            float porosity = fbm(vec2(ang*8.0 + uSeed*9.0 + dir.z*3.0, r*5.5 - uTime*0.04));
            float poreThresh = mix(-0.35, 0.4, smoothstep(0.1, 0.6, uAge));
            alpha *= smoothstep(poreThresh-0.16, poreThresh+0.16, porosity);

            gl_FragColor = vec4(col_final, alpha);
          }
        `,
      });
      if (irisMat.uniforms.uDesign.value === 3) irisMat.uniforms.uMaxR.value = 1.32;
      const iris = new THREE.Mesh(irisGeo, irisMat);
      iris.position.copy(pos);
      // Fixed random 3D orientation, set once and never touched again (no
      // billboarding to the camera) — this is what lets the shell's own
      // volume and lopsided silhouette show correctly as the viewer orbits.
      iris.rotation.set(Math.random() * Math.PI * 2, Math.random() * Math.PI * 2, Math.random() * Math.PI * 2);
      iris.scale.set(0.01, 0.01, 0.01);
      scene.add(iris);

      return {
        points, geo, mat, iris, irisMat, velocities, positions,
        birth: clock.elapsedTime, life: big ? 8.5 : 6.5, big,
        eventHue: [205, 268, 330][(Math.random() * 3) | 0],
        afterglowSent: false,
      };
    }

    function spawnSupernovaAtRay(clientX: number, clientY: number) {
      const ndc = new THREE.Vector2(
        (clientX / window.innerWidth) * 2 - 1,
        -(clientY / window.innerHeight) * 2 + 1,
      );
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(ndc, camera);

      // Click position drives how far along that ray the ignition sits: a
      // click near the center of the screen ignites relatively nearby, a
      // click out toward the edges sends it receding into the distance.
      // The shell itself grows to a ~27-34 unit radius at full size, and the
      // camera normally orbits only ~15 units from the black hole, so the
      // range has to be wide and the floor high enough that a "close" nova
      // doesn't just engulf the whole frame regardless of where you clicked
      // — otherwise every ignition reads as identically close-up.
      const centerDist = Math.min(1.5, Math.hypot(ndc.x, ndc.y));
      const edgeT = centerDist / 1.5; // 0 = dead-center, 1 = far corner
      const dist = 24 + Math.pow(edgeT, 1.15) * 170 + (Math.random() - 0.5) * 10;
      const pos = raycaster.ray.origin
        .clone()
        .add(raycaster.ray.direction.clone().multiplyScalar(dist));

      supernovae.push(makeSupernova(pos, false));
      // Close ignitions still punch the screen flash harder than distant ones.
      flashLevel = Math.min(1.0, flashLevel + 0.5 * (1.0 - edgeT) + 0.15);
      // Flood the palette with a white-gold flash pulse at the click point.
      emitCosmosEvent({
        type: "supernova", heat: 0.5, hue: 46, x: ndc.x, y: ndc.y,
      });
    }

    function spawnAmbientSupernova() {
      const r = 50 + Math.random() * 90;
      const theta = Math.acos(2 * Math.random() - 1);
      const phi = Math.random() * Math.PI * 2;
      const pos = new THREE.Vector3(
        r * Math.sin(theta) * Math.cos(phi),
        r * Math.cos(theta),
        r * Math.sin(theta) * Math.sin(phi),
      );
      supernovae.push(makeSupernova(pos, true));
      flashLevel = Math.min(1.0, flashLevel + 0.5);
      // Project to screen space so the UI pulse can originate near the blast.
      const ndc = pos.clone().project(camera);
      emitCosmosEvent({
        type: "supernova", heat: 0.75, hue: 46, x: ndc.x, y: ndc.y,
      });
    }

    function updateSupernovae(dt: number) {
      for (let i = supernovae.length - 1; i >= 0; i--) {
        const s = supernovae[i];
        const age = (clock.elapsedTime - s.birth) / s.life;
        if (age >= 1.0) {
          scene.remove(s.points);
          scene.remove(s.iris);
          s.geo.dispose();
          s.mat.dispose();
          s.irisMat.dispose();
          supernovae.splice(i, 1);
          continue;
        }
        const drag = Math.pow(0.985, dt * 60);
        const pos = s.positions, vel = s.velocities;
        for (let j = 0; j < pos.length; j += 3) {
          vel[j] *= drag;
          vel[j + 1] *= drag;
          vel[j + 2] *= drag;
          pos[j] += vel[j] * dt;
          pos[j + 1] += vel[j + 1] * dt;
          pos[j + 2] += vel[j + 2] * dt;
        }
        s.geo.attributes.position.needsUpdate = true;
        s.mat.uniforms.uAge.value = age;
        s.mat.uniforms.uTime.value = clock.elapsedTime;

        // The iris grows outward the way real ejecta expands — fast at first,
        // slowing as it goes. It's a real 3D shell now (see makeSupernova),
        // scaled uniformly in all three axes and left at the fixed random
        // orientation it was given at spawn — no camera-facing sync, so its
        // volume and lopsided silhouette are genuinely visible from any angle.
        const baseSize = s.big ? 34 : 27;
        const irisScale = baseSize * Math.pow(age + 0.015, 0.42);
        s.iris.scale.set(irisScale, irisScale, irisScale);
        s.irisMat.uniforms.uAge.value = age;
        s.irisMat.uniforms.uTime.value = clock.elapsedTime;
        s.irisMat.uniforms.uPalette.value = palette.value;

        // Once the ejecta cools into its coloured phase, fire a softer
        // afterglow pulse so the palette drifts teal/violet/magenta with it.
        if (age > 0.5 && !s.afterglowSent) {
          s.afterglowSent = true;
          const ndc = s.iris.position.clone().project(camera);
          emitCosmosEvent({
            type: "supernova", heat: s.big ? 0.4 : 0.28,
            hue: s.eventHue, x: ndc.x, y: ndc.y,
          });
        }
      }
    }

    // ---- post-process: fake gravitational lensing + cinematic grade ----
    let rt = new THREE.WebGLRenderTarget(window.innerWidth, window.innerHeight, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
    });

    const postScene = new THREE.Scene();
    const postCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const postGeo = new THREE.PlaneGeometry(2, 2);
    const MAX_NOVAE = 6;
    const postMat = new THREE.ShaderMaterial({
      uniforms: {
        tDiffuse: { value: rt.texture },
        uBHScreen: { value: new THREE.Vector2(0.5, 0.5) },
        uBHRadius: { value: 0.1 },
        uAspect: { value: window.innerWidth / window.innerHeight },
        uTime: { value: 0 },
        uFlash: { value: 0 },
        uPalette: { value: 0 },
        uTexel: { value: new THREE.Vector2(1 / window.innerWidth, 1 / window.innerHeight) },
        // Real geodesic ray tracing uniforms: the camera's exact world-space
        // basis + position (it's always orbiting/drifting) and the physical
        // black hole / disk parameters, so the fragment shader can integrate
        // true bent light paths instead of faking a 2D screen-space warp.
        uCamPos: { value: new THREE.Vector3() },
        uCamRight: { value: new THREE.Vector3(1, 0, 0) },
        uCamUp: { value: new THREE.Vector3(0, 1, 0) },
        uCamForward: { value: new THREE.Vector3(0, 0, -1) },
        uTanHalfFovY: { value: Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) },
        uRs: { value: R_EH },
        uDiskNormal: { value: new THREE.Vector3(0, 0, 1) },
        uDiskInner: { value: R_IN },
        uDiskOuter: { value: R_OUT },
        // Hypernovae are also real obstacles along the bent ray now — each
        // active explosion's current world position and current bounding
        // radius (it grows over its lifetime) gets fed in here every frame.
        uNovaCount: { value: 0 },
        uNovaPos: {
          value: Array.from({ length: MAX_NOVAE }, () => new THREE.Vector3()),
        },
        uNovaRadius: { value: new Float32Array(MAX_NOVAE) },
      },
      vertexShader: `
        varying vec2 vUv;
        void main(){
          vUv = uv;
          gl_Position = vec4(position.xy, 0.0, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D tDiffuse;
        uniform vec2 uBHScreen;
        uniform float uBHRadius;
        uniform float uAspect;
        uniform float uTime;
        uniform float uFlash;
        uniform float uPalette;
        uniform vec2 uTexel;
        uniform vec3 uCamPos, uCamRight, uCamUp, uCamForward;
        uniform float uTanHalfFovY;
        uniform float uRs;
        uniform vec3 uDiskNormal;
        uniform float uDiskInner, uDiskOuter;
        #define MAX_NOVAE 6
        uniform int uNovaCount;
        uniform vec3 uNovaPos[MAX_NOVAE];
        uniform float uNovaRadius[MAX_NOVAE];
        varying vec2 vUv;

        // ---- Real photon geodesic integration in Schwarzschild spacetime ----
        // Every pixel's camera ray is traced as an actual bent light path
        // rather than warped in flat 2D screen space. The exact equatorial
        // photon-orbit equation (du/dphi)^2 = 1/b^2 - u^2(1 - rs*u), with
        // u = 1/r, differentiates to u'' = -u + 1.5*rs*u^2; the equivalent
        // 3D vector form (conserving specific angular momentum h = p x v)
        // is a central "acceleration" of -1.5*rs*|h|^2/r^5 * p, which is
        // what's integrated below with RK4. Because central forces keep
        // motion planar automatically, this naturally reproduces the true
        // (larger-than-the-horizon) photon-capture shadow, the Einstein
        // ring as an emergent pile-up of directions rather than a drawn-on
        // circle, and correct multiple/higher-order images of the disk —
        // and, now, of any hypernova the bent path happens to graze too.
        vec2 reprojectDir(vec3 dirToHit, float tanX, float tanY, out bool ok){
          float dx = dot(dirToHit, uCamRight), dy = dot(dirToHit, uCamUp), dz = dot(dirToHit, uCamForward);
          ok = dz > 0.0001;
          return vec2((dx/dz/tanX)*0.5+0.5, (dy/dz/tanY)*0.5+0.5);
        }

        bool traceGeodesic(vec2 ndc, out vec2 hitUV, out bool captured){
          float tanY = uTanHalfFovY;
          float tanX = tanY * uAspect;
          vec3 dir = normalize(uCamForward + ndc.x*tanX*uCamRight + ndc.y*tanY*uCamUp);

          vec3 p = uCamPos;
          vec3 v = dir;
          vec3 h = cross(p, v);
          float h2 = dot(h,h);
          captured = false;

          float prevSide = dot(p, uDiskNormal);

          const int STEPS = 200;
          for(int i=0;i<STEPS;i++){
            float r = length(p);
            if(r < uRs*0.98){
              captured = true;
              return true;
            }
            if(r > 400.0){
              bool ok;
              hitUV = reprojectDir(normalize(v), tanX, tanY, ok);
              return ok;
            }

            float stepSize = clamp(r*0.045, 0.001, 2.0);

            vec3 k1p = v;
            vec3 k1v = -1.5*uRs*h2/pow(length(p),5.0) * p;
            vec3 p2 = p + 0.5*stepSize*k1p, v2 = v + 0.5*stepSize*k1v;
            vec3 k2p = v2;
            vec3 k2v = -1.5*uRs*h2/pow(length(p2),5.0) * p2;
            vec3 p3 = p + 0.5*stepSize*k2p, v3 = v + 0.5*stepSize*k2v;
            vec3 k3p = v3;
            vec3 k3v = -1.5*uRs*h2/pow(length(p3),5.0) * p3;
            vec3 p4 = p + stepSize*k3p, v4 = v + stepSize*k3v;
            vec3 k4p = v4;
            vec3 k4v = -1.5*uRs*h2/pow(length(p4),5.0) * p4;

            vec3 pNew = p + (stepSize/6.0)*(k1p + 2.0*k2p + 2.0*k3p + k4p);
            vec3 vNew = v + (stepSize/6.0)*(k1v + 2.0*k2v + 2.0*k3v + k4v);
            vec3 stepDir = pNew - p;

            // Nearest obstruction hit within this step, as a fraction 0..1
            // along p -> pNew. Checked against every active hypernova
            // sphere and the disk plane; whichever is actually reached
            // first along the bent path wins.
            float bestT = 2.0;
            vec3 bestHitPoint = vec3(0.0);
            bool foundHit = false;

            for(int n=0;n<MAX_NOVAE;n++){
              if(n >= uNovaCount) break;
              vec3 f = p - uNovaPos[n];
              float R = uNovaRadius[n];
              float a = dot(stepDir, stepDir);
              float b = 2.0*dot(f, stepDir);
              float c = dot(f,f) - R*R;
              float disc = b*b - 4.0*a*c;
              if(disc >= 0.0 && a > 1e-9){
                float sq = sqrt(disc);
                float t1 = (-b - sq)/(2.0*a);
                float t2 = (-b + sq)/(2.0*a);
                float tHit = (t1 >= 0.0 && t1 <= 1.0) ? t1 : ((t2 >= 0.0 && t2 <= 1.0) ? t2 : -1.0);
                if(tHit >= 0.0 && tHit < bestT){
                  bestT = tHit;
                  bestHitPoint = p + stepDir*tHit;
                  foundHit = true;
                }
              }
            }

            float newSide = dot(pNew, uDiskNormal);
            if(sign(newSide) != sign(prevSide) && abs(prevSide-newSide) > 1e-6){
              float tCross = prevSide/(prevSide-newSide);
              if(tCross >= 0.0 && tCross <= 1.0 && tCross < bestT){
                vec3 crossPt = mix(p, pNew, tCross);
                float rad = length(crossPt);
                if(rad > uDiskInner && rad < uDiskOuter){
                  bestT = tCross;
                  bestHitPoint = crossPt;
                  foundHit = true;
                }
              }
            }

            if(foundHit){
              bool ok;
              hitUV = reprojectDir(normalize(bestHitPoint - uCamPos), tanX, tanY, ok);
              return ok;
            }

            prevSide = newSide;
            p = pNew; v = vNew;
          }
          bool ok;
          hitUV = reprojectDir(normalize(v), tanX, tanY, ok);
          return ok;
        }

        void main(){
          vec2 uv = vUv;
          vec2 ndc = uv*2.0 - 1.0;

          vec2 sampleUV = uv;
          bool captured = false;
          bool hit = traceGeodesic(ndc, sampleUV, captured);
          sampleUV = clamp(sampleUV, 0.001, 0.999);

          // Kept purely as an art-directed proximity measure (extra glow +
          // chromatic spread right at the shadow edge); the ring's actual
          // existence and shape now come from the raytrace itself.
          vec2 p = uv - uBHScreen;
          p.x *= uAspect;
          float dist = length(p);
          vec2 dir = dist > 0.0001 ? p/dist : vec2(0.0);
          float horizon = max(uBHRadius, 0.001);

          vec3 ringTint  = mix(vec3(1.0,0.82,0.55), vec3(0.75,0.9,1.0), uPalette);
          vec3 coreTint  = mix(vec3(1.0,0.95,0.88), vec3(0.92,0.97,1.0), uPalette);
          vec3 caTint    = mix(vec3(1.0,0.6,0.35),  vec3(0.55,0.75,1.0), uPalette);

          vec3 color;
          if(!hit || captured){
            color = vec3(0.0);
          } else {
            color = texture2D(tDiffuse, sampleUV).rgb;

            // Thin, bright Einstein ring where lensed light from every side of
            // the disk piles up right at the shadow's edge — the defining
            // feature of a photorealistic black hole silhouette.
            float ringGlow = smoothstep(horizon*1.05, horizon*0.92, dist) * (1.0-smoothstep(horizon*0.92, horizon*0.82, dist));
            float ringCore = smoothstep(horizon*0.98, horizon*0.93, dist) * (1.0-smoothstep(horizon*0.93, horizon*0.88, dist));
            color += ringTint * ringGlow * 1.1;
            color += coreTint * ringCore * 1.4;

            float ca = smoothstep(horizon*4.5, horizon*0.9, dist) * 0.006;
            vec2 caUV1 = clamp(sampleUV + dir*ca, 0.001, 0.999);
            vec2 caUV2 = clamp(sampleUV - dir*ca, 0.001, 0.999);
            color.r = texture2D(tDiffuse, caUV1).r;
            color.b = texture2D(tDiffuse, caUV2).b;
            color += caTint * ringGlow * 0.4 * ca * 40.0;
          }

          // Cheap wide-radius bloom: pull in a handful of bright-pass taps at
          // increasing offsets and add them back additively, so the disk and
          // halo bands bleed soft light into the surrounding dark the way an
          // over-exposed, bloom-heavy render does in the reference footage.
          vec3 bloom = vec3(0.0);
          const int TAPS = 512;
          for(int i=0;i<TAPS;i++){
            float fi = float(i);
            float ang = fi*2.399963;
            float rad = (fi+1.0)*2.2;
            vec2 off = vec2(cos(ang), sin(ang)) * uTexel * rad;
            vec3 s = texture2D(tDiffuse, clamp(sampleUV+off,0.001,0.999)).rgb;
            float br = max(0.0, dot(s,vec3(0.299,0.587,0.114)) - 0.55);
            bloom += s * br;
          }
          bloom /= float(TAPS);
          color += bloom * 0.9;

          float vig = smoothstep(0.95, 0.22, length(uv-0.5));
          color *= mix(0.45, 1.0, vig);

          // Filmic-leaning tone shaping: crush the near-blacks a touch, roll
          // off the hot highlights so the disk doesn't clip to flat white.
          color = color / (1.0 + color*0.35);
          color = pow(max(color,0.0), vec3(0.92));

          float lum = dot(color, vec3(0.299,0.587,0.114));
          vec3 shadowTint = mix(vec3(0.0,0.05,0.09), vec3(0.0,0.03,0.11), uPalette);
          vec3 highlightTint = mix(vec3(0.07,0.02,-0.025), vec3(-0.02,0.02,0.06), uPalette);
          color += shadowTint * (1.0-smoothstep(0.0,0.35,lum));
          color += highlightTint * smoothstep(0.5,1.0,lum);

          // Slight saturation lift for the richly-colored, punchy look of the
          // reference photography.
          float g = dot(color, vec3(0.299,0.587,0.114));
          color = mix(vec3(g), color, 1.18);

          float grain = fract(sin(dot(uv*(uTime*60.0+1.0), vec2(12.9898,78.233)))*43758.5453);
          color += (grain-0.5)*0.015;

          vec3 flashTint = mix(vec3(1.0,0.9,0.75), vec3(0.85,0.92,1.0), uPalette);
          color += flashTint * uFlash * 0.25;
          color *= (1.0 + uFlash*0.3);

          gl_FragColor = vec4(color, 1.0);
        }
      `,
    });
    const postQuad = new THREE.Mesh(postGeo, postMat);
    postScene.add(postQuad);

    const bhCenterWorld = new THREE.Vector3(0, 0, 0);
    const bhEdgeWorld = new THREE.Vector3(R_EH, 0, 0);
    const camRightV = new THREE.Vector3();
    const camUpV = new THREE.Vector3();
    const camForwardV = new THREE.Vector3();
    function updateLensUniforms() {
      const c = bhCenterWorld.clone().project(camera);
      const e = bhEdgeWorld.clone().project(camera);
      const cUV = new THREE.Vector2((c.x + 1) / 2, (c.y + 1) / 2);
      const eUV = new THREE.Vector2((e.x + 1) / 2, (e.y + 1) / 2);
      postMat.uniforms.uBHScreen.value.copy(cUV);
      postMat.uniforms.uBHRadius.value = Math.max(cUV.distanceTo(eUV), 0.01);

      // Real per-pixel geodesic tracing needs the camera's exact current
      // world-space basis/position (it's always orbiting or auto-drifting)
      // and the disk's current world-space plane normal, since diskGroup
      // keeps precessing too.
      camera.updateMatrixWorld(true);
      camera.matrixWorld.extractBasis(camRightV, camUpV, camForwardV);
      camForwardV.negate(); // the camera looks down its local -Z axis
      postMat.uniforms.uCamPos.value.copy(camera.position);
      postMat.uniforms.uCamRight.value.copy(camRightV);
      postMat.uniforms.uCamUp.value.copy(camUpV);
      postMat.uniforms.uCamForward.value.copy(camForwardV);
      postMat.uniforms.uDiskNormal.value.copy(orbitAxisWorld);

      // Hypernovae move and grow constantly, so their sphere data for the
      // tracer has to be refreshed every frame too.
      const n = Math.min(supernovae.length, MAX_NOVAE);
      postMat.uniforms.uNovaCount.value = n;
      for (let i = 0; i < n; i++) {
        const s = supernovae[i];
        postMat.uniforms.uNovaPos.value[i].copy(s.iris.position);
        // Capped well below the disk's outer radius: this sphere is only meant
        // to catch the *ray tracer's* bent path so a lensed image of the nova
        // reprojects correctly near its own silhouette. Left uncapped, a fully
        // grown hypernova's true bounding sphere (up to ~34*1.9 units) can
        // exceed the distance between where it ignited and the black hole
        // itself, so the "obstacle" swallows the hole's own position and every
        // ray aimed anywhere near it hits the nova first — which is what was
        // reading as a growing orb smothering the horizon, and occasionally as
        // the whole hole falling back to flat, un-bent shading.
        postMat.uniforms.uNovaRadius.value[i] = Math.min(s.iris.scale.x * 1.9, 9.0);
      }
    }

    // ---- input: right-drag orbit, shift+scroll zoom, left-click supernova ----
    let orbitDragging = false, lastX = 0, lastY = 0, downX = 0, downY = 0, downTime = 0;

    const onPointerDown = (e: PointerEvent) => {
      downX = e.clientX;
      downY = e.clientY;
      downTime = performance.now();
      if (e.button === 2) {
        orbitDragging = true;
        lastX = e.clientX;
        lastY = e.clientY;
      }
    };
    const onPointerUp = (e: PointerEvent) => {
      orbitDragging = false;
      if (e.button !== 0) return; // only left click ignites supernovae
      const tgt = e.target as HTMLElement | null;
      if (
        tgt &&
        tgt.closest(
          "a,button,input,textarea,select,[role='button'],[contenteditable]",
        )
      ) {
        return;
      }
      const moved = Math.hypot(e.clientX - downX, e.clientY - downY);
      const dt = performance.now() - downTime;
      if (moved < 6 && dt < 400) {
        spawnSupernovaAtRay(e.clientX, e.clientY);
      }
    };
    const onPointerMove = (e: PointerEvent) => {
      if (orbitDragging && (e.buttons & 2) === 2) {
        const dx = e.clientX - lastX, dy = e.clientY - lastY;
        cam.targetAzimuth -= dx * 0.0045;
        cam.targetPolar -= dy * 0.0035;
        lastX = e.clientX;
        lastY = e.clientY;
      } else {
        const nx = (e.clientX / window.innerWidth) * 2 - 1;
        const ny = (e.clientY / window.innerHeight) * 2 - 1;
        cam.targetAzimuth = cam.azimuth + nx * 0.12 - mouseDriftX;
        cam.targetPolar = 1.25 + ny * 0.1;
        mouseDriftX = nx * 0.12;
      }
    };
    const onWheel = (e: WheelEvent) => {
      if (!e.shiftKey) return;
      e.preventDefault();
      cam.targetRadius = Math.max(3.5, Math.min(60, cam.targetRadius + e.deltaY * 0.01));
    };
    const onContextMenu = (e: MouseEvent) => {
      const tgt = e.target as HTMLElement | null;
      if (
        tgt &&
        tgt.closest(
          "a,button,input,textarea,select,[role='button'],[contenteditable]",
        )
      ) {
        return;
      }
      e.preventDefault();
    };
    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
      rt.setSize(window.innerWidth, window.innerHeight);
      postMat.uniforms.uAspect.value = window.innerWidth / window.innerHeight;
      postMat.uniforms.uTexel.value.set(1 / window.innerWidth, 1 / window.innerHeight);
    };

    window.addEventListener("resize", onResize);
    window.addEventListener("pointermove", onPointerMove, { passive: true });
    window.addEventListener("pointerdown", onPointerDown, { passive: true });
    window.addEventListener("pointerup", onPointerUp, { passive: true });
    window.addEventListener("pointercancel", () => { orbitDragging = false; });
    window.addEventListener("wheel", onWheel, { passive: false });
    window.addEventListener("contextmenu", onContextMenu);

    // ---- animate ----
    let ambientTimer = 0;
    let meteorTimer = 1.5 + Math.random() * 1.5;
    let cometTimer = 14 + Math.random() * 8;
    function animate() {
      raf = requestAnimationFrame(animate);
      const dt = Math.min(clock.getDelta(), 0.05);
      const t = clock.elapsedTime;

      updateCamera(dt);

      updatePalette(t);
      (stars.material as THREE.ShaderMaterial).uniforms.uTime.value = t;
      diskMats.forEach((m) => {
        m.uniforms.uTime.value = t;
        m.uniforms.uPalette.value = palette.value;
      });
      polarRingMats.forEach((m) => {
        m.uniforms.uTime.value = t;
        m.uniforms.uPalette.value = palette.value;
      });
      rimMat.uniforms.uPalette.value = palette.value;
      postMat.uniforms.uPalette.value = palette.value;
      haloMatB.opacity = 0.9 * palette.value;
      haloMatB2.opacity = 0.35 * palette.value;
      haloMat.opacity = 0.9 * (1.0 - palette.value * 0.65);
      haloMat2.opacity = 0.35 * (1.0 - palette.value * 0.65);

      // Real relative motion: the deep background slowly rotates independent
      // of the camera, so stars visibly sweep behind the hole and get bent.
      bgGroup.rotation.y += dt * 0.035;
      bgGroup.rotation.x = Math.sin(t * 0.05) * 0.08;

      // Frame-dragging style precession of the whole black-hole system.
      diskGroup.rotation.z += dt * 0.045;
      bhGroup.rotation.y += dt * 0.012;

      // Real Doppler beaming needs the disk's actual current world-space
      // rotation axis (not a fixed guess) since diskGroup/bhGroup keep
      // precessing — pulled fresh each frame and shared by the disk layers,
      // the lensed halo bands, and the photon ring so all three agree on
      // which side of the hole is currently "approaching".
      scene.updateMatrixWorld(true);
      orbitAxisWorld.set(0, 0, 1).transformDirection(diskGroup.matrixWorld).normalize();
      diskMats.forEach((m) => { m.uniforms.uOrbitAxis.value.copy(orbitAxisWorld); });
      polarRingMats.forEach((m) => { m.uniforms.uOrbitAxis.value.copy(orbitAxisWorld); });
      rimMat.uniforms.uOrbitAxis.value.copy(orbitAxisWorld);

      const haloPulse = 1.0 + Math.sin(t * 0.9) * 0.08;
      halo.scale.set(9 * haloPulse, 9 * haloPulse, 1);
      halo2.scale.set(13 / haloPulse, 13 / haloPulse, 1);
      halo2.material.rotation += dt * 0.05;
      haloB.scale.set(9 * haloPulse, 9 * haloPulse, 1);
      haloB2.scale.set(13 / haloPulse, 13 / haloPulse, 1);
      haloB2.material.rotation += dt * 0.05;

      nebulaGroup.children.forEach((spr) => {
        const s = spr as THREE.Sprite;
        s.position.x += Math.sin(t * 0.05 + s.id) * 0.002;
        s.material.rotation += s.userData.driftSpeed;
      });

      updateSupernovae(dt);
      flashLevel = Math.max(0, flashLevel - dt * 1.4);

      meteorTimer += dt;
      if (meteorTimer > 3.5 + Math.random() * 4) {
        meteorTimer = 0;
        spawnMeteorShower();
      }
      cometTimer += dt;
      if (cometTimer > 30 + Math.random() * 24) {
        cometTimer = 0;
        spawnComet();
      }
      updateMeteors(dt);
      updateComets(dt);

      ambientTimer += dt;
      if (ambientTimer > 9 + Math.random() * 6) {
        ambientTimer = 0;
        spawnAmbientSupernova();
      }

      updateLensUniforms();
      postMat.uniforms.uTime.value = t;
      postMat.uniforms.uFlash.value = flashLevel;

      renderer.setRenderTarget(rt);
      renderer.render(scene, camera);
      renderer.setRenderTarget(null);
      renderer.render(postScene, postCamera);
    }

    updateCamera(0);
    animate();

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("wheel", onWheel);
      window.removeEventListener("contextmenu", onContextMenu);

      for (const s of supernovae) {
        scene.remove(s.points);
        scene.remove(s.iris);
        s.geo.dispose();
        s.mat.dispose();
        s.irisMat.dispose();
      }
      for (const root of [scene, postScene]) {
        root.traverse((o) => {
          const obj = o as THREE.Mesh | THREE.Points | THREE.Sprite;
          if (obj.geometry) obj.geometry.dispose();
          const m = obj.material as THREE.Material | THREE.Material[] | undefined;
          const mats = Array.isArray(m) ? m : m ? [m] : [];
          for (const mat of mats) mat.dispose();
        });
      }
      irisGeo.dispose();
      for (const t of disposables) t.dispose();
      rt.dispose();
      renderer.forceContextLoss();
      renderer.dispose();
      if (cv.parentNode) cv.parentNode.removeChild(cv);
    };
  }, [broken]);

  if (!webglOk || broken) {
    return <CosmosFallback className={className} />;
  }

  return <div ref={hostRef} className={className} aria-hidden="true" />;
}
