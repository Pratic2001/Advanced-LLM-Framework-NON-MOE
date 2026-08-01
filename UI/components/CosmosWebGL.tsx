"use client";

/**
 * Deep-space background — a faithful port of the "Event Horizon" reference
 * (event_horizon.html), including its improvements:
 *
 *   · A black event-horizon sphere with a two-term fresnel photon-ring rim
 *     (pow 4.5 halo + pow 12 near-white core) and a dual-layer pulsing warm
 *     halo (a big out-of-phase glow that slowly rotates).
 *   · A tilted accretion disk displaced into a puffy turbulent torus — the
 *     vertex shader computes a real height field + normal (lambert shading,
 *     limb brightening), the fragment shader runs a 4-stop colour ramp with
 *     flicker, relativistic doppler beaming with blue/red-shift tinting.
 *   · A secondary "polar ring" (Gargantua-style halo silhouette) torus
 *     perpendicular to the disk, flowing independently.
 *   · 9000 twinkling stars + seven colourful nebulae in a slowly-rotating
 *     background group, so stars visibly sweep behind the hole and get bent.
 *   · Supernovae: particle sparks (with a sparkle term) PLUS a procedural
 *     "iris" billboard — a camera-facing quad of fine radial fibres, a
 *     collarette ring and a young→old colour morph, growing like real ejecta.
 *     One ambient (big) every 9–15 s, plus any number detonated by clicking
 *     empty sky. Each blast feeds a `uFlash` into the post pass.
 *   · A fake-lensing post pass: screen-space warp (with a breathing ripple),
 *     a thin bright Einstein ring + near-white core, radial chromatic
 *     aberration, stronger vignette, filmic tone shaping, dual-tone grade,
 *     saturation lift and film grain.
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

    // ---- camera orbit state (spherical, target = origin) ----
    const cam = {
      azimuth: 0.55, polar: 1.25, radius: 15,
      targetAzimuth: 0.55, targetPolar: 1.25, targetRadius: 15,
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

    // Shared unit quad for the supernova "iris" billboard — geometry only
    // carries UVs; every bit of shape comes from the fragment shader, so the
    // visible silhouette is always a soft circle, never a texture square.
    const irisGeo = new THREE.PlaneGeometry(1, 1);

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

    // ---- black hole: event horizon, photon rim, disk, halo ----
    const bhGroup = new THREE.Group();
    scene.add(bhGroup);

    const R_EH = 1.0;

    const horizonGeo = new THREE.SphereGeometry(R_EH, 64, 64);
    const horizonMat = new THREE.MeshBasicMaterial({ color: 0x000000 });
    const horizon = new THREE.Mesh(horizonGeo, horizonMat);
    bhGroup.add(horizon);

    // Photon-ring rim glow (fresnel, two terms: wide warm halo + bright core)
    const rimGeo = new THREE.SphereGeometry(R_EH * 1.035, 64, 64);
    const rimMat = new THREE.ShaderMaterial({
      uniforms: {},
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vViewPos;
        void main(){
          vNormal = normalize(normalMatrix*normal);
          vec4 mv = modelViewMatrix*vec4(position,1.0);
          vViewPos = mv.xyz;
          gl_Position = projectionMatrix*mv;
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        varying vec3 vViewPos;
        void main(){
          vec3 viewDir = normalize(-vViewPos);
          float f = max(dot(viewDir,vNormal),0.0);
          float fresnel = pow(1.0-f, 4.5);
          float core = pow(1.0-f, 12.0);
          vec3 glowColor = mix(vec3(1.0,0.86,0.62), vec3(1.0,0.97,0.92), core);
          float intensity = fresnel*2.0 + core*3.2;
          gl_FragColor = vec4(glowColor*intensity, min(1.0, fresnel*1.3 + core));
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

    // Accretion disk — a puffy torus of turbulent plasma.
    const R_IN = R_EH * 1.9, R_OUT = R_EH * 8.5;
    const diskGeo = new THREE.RingGeometry(R_IN, R_OUT, 200, 28);
    const diskMat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uInner: { value: R_IN },
        uOuter: { value: R_OUT },
      },
      vertexShader: `
        uniform float uTime;
        uniform float uInner;
        uniform float uOuter;
        varying vec3 vPos;
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
        // Vertical (out-of-plane) height of the disk surface — turns a flat
        // ring into a puffy torus of turbulent plasma.
        float heightAt(vec2 pos){
          float r = length(pos);
          float rn = clamp((r-uInner)/(uOuter-uInner), 0.0, 1.0);
          float angle = atan(pos.y, pos.x);
          vec2 c = vec2(cos(angle),sin(angle)) * (2.0+rn*3.0) + vec2(uTime*0.16, r*0.35 - uTime*mix(1.6,0.25,rn));
          float n = fbm(c*1.4);
          float profile = sin(clamp(rn,0.02,0.98)*3.14159265); // thin at both edges, puffed in the middle
          return (n-0.5) * mix(0.55, 0.16, rn) * profile;
        }

        void main(){
          vec2 xy = position.xy;
          float h = heightAt(xy);
          float eps = 0.08;
          float hx = heightAt(xy+vec2(eps,0.0));
          float hy = heightAt(xy+vec2(0.0,eps));
          vec3 n = normalize(vec3(-(hx-h)/eps, -(hy-h)/eps, 1.0));

          vec3 displaced = vec3(xy, h);
          vPos = displaced;

          vec3 lightDir = normalize(vec3(0.35, 0.55, 0.75));
          vShade = clamp(dot(n, lightDir), 0.12, 1.0);

          vec3 viewNormal = normalize(normalMatrix * n);
          vec4 mvPos = modelViewMatrix * vec4(displaced,1.0);
          vec3 viewDir = normalize(-mvPos.xyz);
          // Limb brightening: material seen edge-on through more of the puffy
          // disk's depth scatters more light back at us.
          vRim = pow(1.0-clamp(abs(dot(viewDir, viewNormal)),0.0,1.0), 2.4);

          gl_Position = projectionMatrix*mvPos;
        }
      `,
      fragmentShader: `
        uniform float uTime;
        uniform float uInner;
        uniform float uOuter;
        varying vec3 vPos;
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

          float flicker = 0.85 + 0.3*sin(uTime*3.0 + r*6.0 + turb*8.0) * (1.0-rn*0.5);

          vec3 hot  = vec3(1.0, 0.99, 0.95);
          vec3 mid  = vec3(1.0, 0.72, 0.32);
          vec3 amber= vec3(1.0, 0.45, 0.09);
          vec3 cool = vec3(0.62, 0.10, 0.03);
          vec3 col = mix(hot, mid, smoothstep(0.0,0.3,rn));
          col = mix(col, amber, smoothstep(0.25,0.65,rn));
          col = mix(col, cool, smoothstep(0.55,1.0,rn));
          col *= (0.4 + 1.3*turbulence) * flicker;

          // Relativistic beaming: material sweeping toward the camera is
          // Doppler-boosted and blue-shifted; the receding side dims/reddens.
          float doppler = 0.5 + 0.5*cos(angle - uTime*0.15 - 0.5);
          float dopplerSharp = pow(doppler, 1.6);
          col *= mix(0.32, 2.6, dopplerSharp);
          col = mix(col, vec3(0.75,0.85,1.05)*length(col), smoothstep(0.55,1.0,doppler)*0.28);
          col = mix(col, vec3(1.15,0.55,0.28)*length(col), smoothstep(0.45,0.0,doppler)*0.22);

          // Fake volumetric lighting from the puffy-surface normal + warm
          // limb-brightened rim — reads as an actual body of plasma.
          col *= mix(0.55, 1.4, vShade);
          col += vec3(1.0,0.82,0.55) * vRim * 0.6;

          float edgeFade = smoothstep(0.0,0.08,rn) * smoothstep(1.0,0.8,rn);
          float alpha = edgeFade * (0.6 + 0.6*turbulence) * flicker + vRim*0.25*edgeFade;

          gl_FragColor = vec4(col, clamp(alpha,0.0,1.4));
        }
      `,
      transparent: true, side: THREE.DoubleSide, depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const disk = new THREE.Mesh(diskGeo, diskMat);
    disk.rotation.x = Math.PI * 0.42;
    disk.rotation.z = 0.15;
    bhGroup.add(disk);

    // Secondary lensed ring — an approximation of the disk's light bent over
    // the poles of the hole (the classic Gargantua "halo" silhouette).
    const polarRingGeo = new THREE.TorusGeometry(R_EH * 1.28, R_EH * 0.16, 24, 200);
    const polarRingMat = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 } },
      vertexShader: `
        varying vec2 vUvV;
        void main(){
          vUvV = uv;
          gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0);
        }
      `,
      fragmentShader: `
        uniform float uTime;
        varying vec2 vUvV;
        float hash(float n){ return fract(sin(n)*43758.5453123); }
        void main(){
          float ang = vUvV.x*6.2831 + uTime*1.4;
          float flow = sin(ang*3.0)*0.5+0.5;
          float flow2 = sin(ang*7.0 - uTime*2.0)*0.5+0.5;
          float turb = mix(flow, flow2, 0.5);
          vec3 hot = vec3(1.0,0.95,0.85);
          vec3 warm = vec3(1.0,0.55,0.2);
          vec3 col = mix(warm, hot, turb) * (0.7+0.6*turb);
          float edge = smoothstep(0.0,0.25,vUvV.y) * smoothstep(1.0,0.75,vUvV.y);
          float alpha = edge * (0.35 + 0.35*turb);
          gl_FragColor = vec4(col, alpha);
        }
      `,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });
    const polarRing = new THREE.Mesh(polarRingGeo, polarRingMat);
    polarRing.rotation.y = Math.PI * 0.5;
    bhGroup.add(polarRing);

    // ---- supernovae: particle sparks + procedural "iris" billboard ----
    const supernovae: Supernova[] = [];
    let flashLevel = 0;

    function makeSupernova(pos: THREE.Vector3, big: boolean): Supernova {
      const count = big ? 900 : 420;
      const positions = new Float32Array(count * 3);
      const velocities = new Float32Array(count * 3);
      const sizes = new Float32Array(count);
      const rands = new Float32Array(count);

      for (let i = 0; i < count; i++) {
        const theta = Math.acos(2 * Math.random() - 1);
        const phi = Math.random() * Math.PI * 2;
        const speed = (2.0 + Math.random() * 6.0) * (big ? 1.6 : 1.0);
        const dx = Math.sin(theta) * Math.cos(phi);
        const dy = Math.cos(theta);
        const dz = Math.sin(theta) * Math.sin(phi);
        positions[i * 3] = pos.x;
        positions[i * 3 + 1] = pos.y;
        positions[i * 3 + 2] = pos.z;
        velocities[i * 3] = dx * speed;
        velocities[i * 3 + 1] = dy * speed;
        velocities[i * 3 + 2] = dz * speed;
        sizes[i] = Math.random() * 2.2 + 0.6;
        rands[i] = Math.random();
      }

      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geo.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
      geo.setAttribute("aRand", new THREE.BufferAttribute(rands, 1));

      const mat = new THREE.ShaderMaterial({
        uniforms: { uAge: { value: 0 }, uTime: { value: 0 } },
        vertexShader: `
          attribute float aSize;
          attribute float aRand;
          varying float vRand;
          uniform float uAge;
          void main(){
            vRand = aRand;
            vec4 mv = modelViewMatrix*vec4(position,1.0);
            float sizeFade = mix(1.4, 0.15, uAge);
            gl_PointSize = aSize*sizeFade*(420.0/-mv.z);
            gl_Position = projectionMatrix*mv;
          }
        `,
        fragmentShader: `
          varying float vRand;
          uniform float uAge;
          uniform float uTime;
          void main(){
            vec2 uv = gl_PointCoord-0.5;
            float d = length(uv);
            float alpha = smoothstep(0.5,0.0,d);
            vec3 hotc  = vec3(1.0,1.0,0.94);
            vec3 midc  = mix(vec3(1.0,0.5,0.14), vec3(0.9,0.2,0.6), vRand);
            vec3 coolc = mix(vec3(0.65,0.08,0.35), vec3(0.2,0.35,1.0), vRand);
            vec3 col = mix(hotc, midc, smoothstep(0.0,0.4,uAge));
            col = mix(col, coolc, smoothstep(0.35,1.0,uAge));
            float fade = 1.0-smoothstep(0.55,1.0,uAge);
            float sparkle = 0.75 + 0.35*sin(uTime*18.0 + vRand*60.0);
            gl_FragColor = vec4(col*sparkle, alpha*fade);
          }
        `,
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      });

      const points = new THREE.Points(geo, mat);
      scene.add(points);

      // The iris — a single always-camera-facing quad whose fragment shader
      // grows fine radial fibres out from a bright core, the way a real
      // supernova's ejecta forms filamentary structure. Everything is
      // procedural and the silhouette is forced to a soft circle in-shader.
      const irisMat = new THREE.ShaderMaterial({
        uniforms: {
          uAge: { value: 0 },
          uTime: { value: 0 },
          uSeed: { value: Math.random() * 1000 },
        },
        transparent: true, depthWrite: false, depthTest: true,
        blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
        vertexShader: `
          varying vec2 vUv;
          void main(){
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: `
          uniform float uAge, uTime, uSeed;
          varying vec2 vUv;

          float hash(vec2 p){ return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453123); }
          float noise(vec2 p){
            vec2 i=floor(p), f=fract(p);
            float a=hash(i), b=hash(i+vec2(1.0,0.0)), c=hash(i+vec2(0.0,1.0)), d=hash(i+vec2(1.0,1.0));
            vec2 u=f*f*(3.0-2.0*f);
            return mix(a,b,u.x) + (c-a)*u.y*(1.0-u.x) + (d-b)*u.x*u.y;
          }
          float fbm(vec2 p){
            float v=0.0, amp=0.5;
            for(int i=0;i<5;i++){ v += amp*noise(p); p *= 2.15; amp *= 0.55; }
            return v;
          }

          void main(){
            vec2 p = (vUv - 0.5) * 2.0;
            float r = length(p);
            if(r > 1.0) discard;
            float ang = atan(p.y, p.x);

            // Fine radial striations, like the fibrous structure of an iris
            // or of real filamentary ejecta.
            float fibers  = fbm(vec2(ang*46.0 + uSeed*11.0, r*3.2 - uTime*0.05));
            float fibers2 = fbm(vec2(ang*90.0 + uSeed*23.0, r*6.0 + uTime*0.08));
            float spiky   = pow(fbm(vec2(ang*30.0+uSeed*5.0, r*2.0)), 3.0);
            float cloudy  = fbm(vec2(ang*7.0 + uSeed*3.0 + uTime*0.04, r*4.0 - uTime*0.07));
            float texture1 = mix(fibers, fibers2, 0.5);

            // Young explosions are tight, spiky, radiant threads; aged ones
            // relax into a softer, cloudier filamentary shell.
            float pattern = mix(mix(texture1, spiky, 0.5), cloudy, smoothstep(0.15,0.7,uAge));

            // A collarette-like ring partway out, where density and color shift.
            float ringPos = mix(0.42, 0.6, fbm(vec2(uSeed, ang*2.0))*0.3+0.5);
            float collarette = smoothstep(ringPos-0.1, ringPos, r) * (1.0-smoothstep(ringPos, ringPos+0.16, r));

            float core = pow(1.0-clamp(r,0.0,1.0), 3.4);
            float edgeFade = 1.0 - smoothstep(0.72, 1.0, r);

            vec3 youngCore = vec3(1.0,0.98,0.9);
            vec3 youngMid  = vec3(1.0,0.62,0.22);
            vec3 youngEdge = vec3(0.85,0.2,0.06);
            vec3 oldCore   = vec3(0.7,0.82,1.0);
            vec3 oldMid    = vec3(0.75,0.22,0.82);
            vec3 oldEdge   = vec3(0.5,0.12,0.4);

            vec3 coreCol = mix(youngCore, oldCore, uAge);
            vec3 midCol  = mix(youngMid,  oldMid,  uAge);
            vec3 edgeCol = mix(youngEdge, oldEdge, uAge);

            vec3 col = mix(coreCol, midCol, smoothstep(0.0,0.55,r));
            col = mix(col, edgeCol, smoothstep(0.45,1.0,r));
            col *= (0.55 + 0.9*pattern);
            col += edgeCol * collarette * 0.5;

            float alpha = clamp(core*1.6 + pattern*edgeFade*0.85 + collarette*0.3*edgeFade, 0.0, 1.0);
            alpha *= smoothstep(0.0, 0.03, uAge + 0.002);
            alpha *= (1.0 - smoothstep(0.82, 1.0, uAge));

            gl_FragColor = vec4(col, alpha);
          }
        `,
      });
      const iris = new THREE.Mesh(irisGeo, irisMat);
      iris.position.copy(pos);
      iris.scale.set(0.01, 0.01, 1);
      scene.add(iris);

      return {
        points, geo, mat, iris, irisMat, velocities, positions,
        birth: clock.elapsedTime, life: big ? 7.5 : 4.5, big,
      };
    }

    function spawnSupernovaAtRay(clientX: number, clientY: number) {
      const ndc = new THREE.Vector2(
        (clientX / window.innerWidth) * 2 - 1,
        -(clientY / window.innerHeight) * 2 + 1,
      );
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(ndc, camera);
      const dist = 18 + Math.random() * 30;
      const pos = raycaster.ray.origin
        .clone()
        .add(raycaster.ray.direction.clone().multiplyScalar(dist));
      supernovae.push(makeSupernova(pos, false));
      flashLevel = Math.min(1.0, flashLevel + 0.35);
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

        // The iris grows the way real ejecta expands — fast at first, slowing
        // as it goes — and always faces the camera.
        const baseSize = s.big ? 34 : 22;
        const irisScale = baseSize * Math.pow(age + 0.015, 0.42);
        s.iris.scale.set(irisScale, irisScale, 1);
        s.iris.quaternion.copy(camera.quaternion);
        s.irisMat.uniforms.uAge.value = age;
        s.irisMat.uniforms.uTime.value = clock.elapsedTime;
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
    const postMat = new THREE.ShaderMaterial({
      uniforms: {
        tDiffuse: { value: rt.texture },
        uBHScreen: { value: new THREE.Vector2(0.5, 0.5) },
        uBHRadius: { value: 0.1 },
        uAspect: { value: window.innerWidth / window.innerHeight },
        uTime: { value: 0 },
        uFlash: { value: 0 },
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
        varying vec2 vUv;

        void main(){
          vec2 uv = vUv;
          vec2 p = uv - uBHScreen;
          p.x *= uAspect;
          float dist = length(p);
          vec2 dir = dist > 0.0001 ? p/dist : vec2(0.0);

          float horizon = max(uBHRadius, 0.001);
          float ripple = 1.0 + 0.08*sin(uTime*0.6) + 0.03*sin(uTime*2.3 + dist*40.0);
          float strength = (horizon*horizon*3.6*ripple) / max(dist*dist, horizon*horizon*0.3);
          strength = clamp(strength, 0.0, 2.8);

          vec2 bentP = p - dir*strength*horizon;
          vec2 sampleUV = vec2(bentP.x/uAspect, bentP.y) + uBHScreen;
          sampleUV = clamp(sampleUV, 0.001, 0.999);

          vec3 color;
          if(dist < horizon*0.9){
            color = vec3(0.0);
          } else {
            color = texture2D(tDiffuse, sampleUV).rgb;

            // Thin, bright Einstein ring where lensed light from every side of
            // the disk piles up right at the shadow's edge.
            float ringGlow = smoothstep(horizon*1.05, horizon*0.92, dist) * (1.0-smoothstep(horizon*0.92, horizon*0.82, dist));
            float ringCore = smoothstep(horizon*0.98, horizon*0.93, dist) * (1.0-smoothstep(horizon*0.93, horizon*0.88, dist));
            color += vec3(1.0,0.82,0.55) * ringGlow * 1.9;
            color += vec3(1.0,0.95,0.88) * ringCore * 2.4;

            float ca = smoothstep(horizon*4.5, horizon*0.9, dist) * 0.006;
            vec2 caUV1 = clamp(sampleUV + dir*ca, 0.001, 0.999);
            vec2 caUV2 = clamp(sampleUV - dir*ca, 0.001, 0.999);
            color.r = texture2D(tDiffuse, caUV1).r;
            color.b = texture2D(tDiffuse, caUV2).b;
            color += vec3(1.0,0.6,0.35) * ringGlow * 0.4 * ca * 40.0;
          }

          float vig = smoothstep(0.95, 0.22, length(uv-0.5));
          color *= mix(0.45, 1.0, vig);

          // Filmic-leaning tone shaping: crush near-blacks, roll off hot
          // highlights so the disk doesn't clip to flat white.
          color = color / (1.0 + color*0.35);
          color = pow(max(color,0.0), vec3(0.92));

          float lum = dot(color, vec3(0.299,0.587,0.114));
          color += vec3(0.0,0.05,0.09) * (1.0-smoothstep(0.0,0.35,lum));
          color += vec3(0.07,0.02,-0.025) * smoothstep(0.5,1.0,lum);

          // Slight saturation lift for the punchy look of the reference.
          float g = dot(color, vec3(0.299,0.587,0.114));
          color = mix(vec3(g), color, 1.18);

          float grain = fract(sin(dot(uv*(uTime*60.0+1.0), vec2(12.9898,78.233)))*43758.5453);
          color += (grain-0.5)*0.015;

          color += vec3(1.0,0.9,0.75) * uFlash * 0.25;
          color *= (1.0 + uFlash*0.3);

          gl_FragColor = vec4(color, 1.0);
        }
      `,
    });
    const postQuad = new THREE.Mesh(postGeo, postMat);
    postScene.add(postQuad);

    const bhCenterWorld = new THREE.Vector3(0, 0, 0);
    const bhEdgeWorld = new THREE.Vector3(R_EH, 0, 0);
    function updateLensUniforms() {
      const c = bhCenterWorld.clone().project(camera);
      const e = bhEdgeWorld.clone().project(camera);
      const cUV = new THREE.Vector2((c.x + 1) / 2, (c.y + 1) / 2);
      const eUV = new THREE.Vector2((e.x + 1) / 2, (e.y + 1) / 2);
      postMat.uniforms.uBHScreen.value.copy(cUV);
      postMat.uniforms.uBHRadius.value = Math.max(cUV.distanceTo(eUV), 0.01);
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
    function animate() {
      raf = requestAnimationFrame(animate);
      const dt = Math.min(clock.getDelta(), 0.05);
      const t = clock.elapsedTime;

      updateCamera(dt);

      (stars.material as THREE.ShaderMaterial).uniforms.uTime.value = t;
      diskMat.uniforms.uTime.value = t;
      polarRingMat.uniforms.uTime.value = t;

      // Real relative motion: the deep background slowly rotates independent
      // of the camera, so stars visibly sweep behind the hole and get bent.
      bgGroup.rotation.y += dt * 0.035;
      bgGroup.rotation.x = Math.sin(t * 0.05) * 0.08;

      // Frame-dragging style precession of the whole black-hole system.
      disk.rotation.z += dt * 0.045;
      bhGroup.rotation.y += dt * 0.012;

      const haloPulse = 1.0 + Math.sin(t * 0.9) * 0.08;
      halo.scale.set(9 * haloPulse, 9 * haloPulse, 1);
      halo2.scale.set(13 / haloPulse, 13 / haloPulse, 1);
      halo2.material.rotation += dt * 0.05;

      nebulaGroup.children.forEach((spr) => {
        const s = spr as THREE.Sprite;
        s.position.x += Math.sin(t * 0.05 + s.id) * 0.002;
        s.material.rotation += s.userData.driftSpeed;
      });

      updateSupernovae(dt);
      flashLevel = Math.max(0, flashLevel - dt * 1.4);

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
