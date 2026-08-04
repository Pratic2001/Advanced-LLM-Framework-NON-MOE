// Hardware audit for the render router.
//
// Runs entirely in the browser. Produces a verdict of "heavy" (full WebGL UI)
// or "lite" (canvas-2D UI) from:
//   1. WebGL2 / WebGL1 context availability
//   2. GPU renderer string (detects software rasterizers like SwiftShader)
//   3. navigator.deviceMemory + hardwareConcurrency
//   4. A short WebGL draw-call stress benchmark (2D-canvas fallback)
//
// Thresholds are deliberately conservative: a capable machine landing on the
// lite UI is a minor inconvenience, while a weak machine landing on the heavy
// WebGL UI is a bad experience.

export type Verdict = "heavy" | "lite";

export interface AuditResult {
  webgl2: boolean;
  webgl1: boolean;
  renderer: string;
  softwareRenderer: boolean;
  deviceMemory: number | null; // GB, null when the browser doesn't expose it
  cores: number | null;
  dpr: number;
  viewport: string;
  coarsePointer: boolean;
  webglDraws: number | null; // draws/sec from the WebGL stress loop
  canvas2D: number | null; // ops/sec from the 2D fallback loop
  score: number;
  verdict: Verdict;
  reasons: string[];
}

function getGL(): {
  gl: WebGLRenderingContext | WebGL2RenderingContext | null;
  is2: boolean;
} {
  try {
    const canvas = document.createElement("canvas");
    // failIfMajorPerformanceCaveat: false so software renderers still hand
    // back a context — we detect them via the renderer string instead.
    const gl2 = canvas.getContext("webgl2", {
      failIfMajorPerformanceCaveat: false,
    });
    if (gl2) return { gl: gl2 as WebGL2RenderingContext, is2: true };
    const gl1 = canvas.getContext("webgl", {
      failIfMajorPerformanceCaveat: false,
    });
    if (gl1) return { gl: gl1 as WebGLRenderingContext, is2: false };
  } catch {
    /* ignore */
  }
  return { gl: null, is2: false };
}

function getRenderer(gl: WebGLRenderingContext | WebGL2RenderingContext): string {
  try {
    const dbg = gl.getExtension("WEBGL_debug_renderer_info");
    if (dbg) {
      const unmasked = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL);
      if (unmasked) return String(unmasked);
    }
    const masked = gl.getParameter(gl.RENDERER);
    return masked ? String(masked) : "unknown";
  } catch {
    return "unknown";
  }
}

const SOFTWARE_RENDERER_RE =
  /swiftshader|llvmpipe|software|basic render|microsoft basic|google.*sw/i;

// Stress the WebGL pipeline: draw a full ring of triangles in a tight loop for
// a fixed window and report draws/second. Frames are rendered synchronously, so
// the number is not vsync-limited and spreads fast vs. slow GPUs widely.
function benchmarkWebGL(
  gl: WebGLRenderingContext | WebGL2RenderingContext
): number | null {
  try {
    const compile = (type: number, src: string) => {
      const s = gl.createShader(type);
      if (!s) throw new Error("shader alloc");
      gl.shaderSource(s, src);
      gl.compileShader(s);
      return s;
    };
    const vs = compile(
      gl.VERTEX_SHADER,
      "attribute vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }"
    );
    const fs = compile(
      gl.FRAGMENT_SHADER,
      "precision mediump float; void main(){ gl_FragColor = vec4(0.5,0.3,0.9,1.0); }"
    );
    const prog = gl.createProgram();
    if (!prog) return null;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return null;
    gl.useProgram(prog);

    const COUNT = 3000;
    const data = new Float32Array(COUNT * 2);
    for (let i = 0; i < COUNT; i++) {
      const a = (i / COUNT) * Math.PI * 2;
      data[i * 2] = Math.cos(a);
      data[i * 2 + 1] = Math.sin(a);
    }
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "p");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    const WINDOW = 500;
    const start = performance.now();
    let draws = 0;
    while (performance.now() - start < WINDOW) {
      gl.clearColor(0.02, 0.02, 0.08, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLES, 0, COUNT);
      draws++;
    }
    return Math.round((draws * 1000) / WINDOW);
  } catch {
    return null;
  }
}

// Fallback when WebGL is unavailable: timed 2D arc fills on a small canvas.
function benchmark2D(): number | null {
  try {
    const canvas = document.createElement("canvas");
    canvas.width = 400;
    canvas.height = 300;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    const WINDOW = 300;
    const start = performance.now();
    let ops = 0;
    while (performance.now() - start < WINDOW) {
      for (let i = 0; i < 200; i++) {
        ctx.fillStyle = `hsl(${(i * 37) % 360} 80% 60%)`;
        ctx.beginPath();
        ctx.arc(Math.random() * 400, Math.random() * 300, 6, 0, Math.PI * 2);
        ctx.fill();
      }
      ops += 200;
    }
    return Math.round((ops * 1000) / WINDOW);
  } catch {
    return null;
  }
}

export async function runAudit(): Promise<AuditResult> {
  const { gl, is2 } = getGL();
  const renderer = gl ? getRenderer(gl) : "none";
  const softwareRenderer = SOFTWARE_RENDERER_RE.test(renderer);
  const nav = navigator as Navigator & { deviceMemory?: number };
  const deviceMemory =
    typeof nav.deviceMemory === "number" && nav.deviceMemory > 0
      ? nav.deviceMemory
      : null;
  const cores =
    typeof navigator.hardwareConcurrency === "number"
      ? navigator.hardwareConcurrency
      : null;
  const dpr = window.devicePixelRatio || 1;
  const coarsePointer =
    typeof window.matchMedia === "function"
      ? window.matchMedia("(pointer: coarse)").matches
      : false;

  let webglDraws: number | null = null;
  let canvas2D: number | null = null;
  if (gl) webglDraws = benchmarkWebGL(gl);
  else canvas2D = benchmark2D();

  const reasons: string[] = [];
  let score = 0;
  let verdict: Verdict = "heavy";

  if (is2) {
    score += 40;
    reasons.push("WebGL2 available");
  } else if (gl) {
    score += 20;
    reasons.push("WebGL1 only");
  } else {
    reasons.push("No WebGL — canvas 2D only");
  }

  if (softwareRenderer) {
    reasons.push(`Software renderer: ${renderer}`);
  } else {
    score += 25;
    reasons.push(
      renderer && renderer !== "unknown" ? `GPU: ${renderer}` : "GPU renderer detected"
    );
  }

  if (deviceMemory != null) {
    if (deviceMemory >= 8) {
      score += 15;
      reasons.push(`${deviceMemory} GB memory`);
    } else if (deviceMemory >= 4) {
      score += 8;
      reasons.push(`${deviceMemory} GB memory`);
    } else {
      reasons.push(`${deviceMemory} GB memory (low)`);
    }
  } else {
    reasons.push("Memory: unknown");
  }

  if (cores != null) {
    if (cores >= 8) {
      score += 10;
      reasons.push(`${cores} CPU threads`);
    } else if (cores >= 4) {
      score += 6;
      reasons.push(`${cores} CPU threads`);
    } else {
      reasons.push(`${cores} CPU threads (low)`);
    }
  } else {
    reasons.push("CPU: unknown");
  }

  if (gl && webglDraws != null) {
    if (webglDraws >= 2000) {
      score += 10;
      reasons.push(`WebGL throughput ~${webglDraws}/s (fast)`);
    } else if (webglDraws >= 500) {
      score += 6;
      reasons.push(`WebGL throughput ~${webglDraws}/s (ok)`);
    } else {
      score -= 15;
      reasons.push(`WebGL throughput ~${webglDraws}/s (slow)`);
    }
  } else if (canvas2D != null) {
    reasons.push(`2D canvas ~${canvas2D} ops/s`);
  }

  // Hard rules: any of these overrides the score.
  if (!gl) {
    verdict = "lite";
    reasons.push("No WebGL context — routed to Lite");
  } else if (softwareRenderer) {
    verdict = "lite";
    reasons.push("Software-rendered GPU — routed to Lite");
  } else if (deviceMemory != null && deviceMemory < 4) {
    verdict = "lite";
    reasons.push("Under 4 GB memory — routed to Lite");
  } else if (cores != null && cores < 4) {
    verdict = "lite";
    reasons.push("Fewer than 4 CPU threads — routed to Lite");
  } else if (webglDraws != null && webglDraws < 500) {
    verdict = "lite";
    reasons.push("Very low WebGL throughput — routed to Lite");
  } else {
    verdict = score >= 55 ? "heavy" : "lite";
  }

  return {
    webgl2: is2,
    webgl1: !!gl && !is2,
    renderer,
    softwareRenderer,
    deviceMemory,
    cores,
    dpr,
    viewport: `${window.innerWidth}×${window.innerHeight}`,
    coarsePointer,
    webglDraws,
    canvas2D,
    score,
    verdict,
    reasons,
  };
}
