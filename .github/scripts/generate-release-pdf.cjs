/**
 * generate-release-pdf.cjs
 *
 * Renders a single-file PDF summarizing what was deployed, what's new, and
 * what's changed for a containerized llmforge release.
 *
 * Reads context from env vars (all set by the GitHub Actions job):
 *   RELEASE_TAG                    — e.g. v0.1.0
 *   RELEASE_DATE                   — ISO-8601 UTC
 *   RELEASE_REPO_URL               — e.g. https://github.com/Pratic2001/Advanced-LLM-Framework-NON-MOE
 *   RELEASE_COMMIT                 — short SHA that triggered the tag
 *   PREVIOUS_TAG                   — the tag before this one, or empty if first
 *   ROUTER_REF, UI_REF, TRAINER_REF — image refs that were pushed
 *   HEAVY_DIGEST, LITE_DIGEST, ROUTER_DIGEST, TRAINER_DIGEST — sha256:... digests
 *   PUBLIC_HOSTNAME                — Tailscale MagicDNS hostname
 *
 * Reads commits since PREVIOUS_TAG from the local git checkout and embeds
 * them as the "What's changed" section.
 *
 * Output: release.pdf in the cwd.
 *
 * Uses `pdfkit` (npm install --no-save pdfkit). Pure JS, no browser, no
 * chrome dependency — works on ubuntu-latest runners with `npm ci`.
 */

const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

// We require pdfkit lazily so the failure mode is clearer if npm install
// didn't include it.
let PDFDocument, doc;
try {
  PDFDocument = require("pdfkit");
} catch (e) {
  console.error("[generate-release-pdf] pdfkit is not installed.");
  console.error("Run `npm install pdfkit` in the workflow step first.");
  process.exit(1);
}

function env(name, fallback = "") {
  return process.env[name] !== undefined ? process.env[name] : fallback;
}

const TAG = env("RELEASE_TAG", "v?.?.?");
const DATE = env("RELEASE_DATE", new Date().toUTCString());
const REPO = env("RELEASE_REPO_URL");
const COMMIT = env("RELEASE_COMMIT", "????????");
const PREV = env("PREVIOUS_TAG", "");

const ROUTER_REF = env("ROUTER_REF", "pratic2001/llmforge-ui-router:latest");
const UI_REF = env("UI_REF", "pratic2001/llmforge-ui:latest");
const TRAINER_REF = env("TRAINER_REF", "pratic2001/llmforge-trainer:latest");

const ROUTER_DIGEST = env("ROUTER_DIGEST", "—");
const UI_DIGEST = env("UI_DIGEST", "—");
const TRAINER_DIGEST = env("TRAINER_DIGEST", "—");

const HOSTNAME = env("PUBLIC_HOSTNAME", "your-host.tailnet.ts.net");

// Pull the commit log from git. PREVIOUS_TAG may be empty (first release) —
// in that case we list the last 25 commits overall so the PDF isn't blank.
let commits = [];
try {
  const range = PREV ? `${PREV}..HEAD` : "HEAD~25..HEAD";
  const log = execSync(`git log --no-merges --pretty=format:"%h%x09%an%x09%s" ${range}`, {
    encoding: "utf8",
    maxBuffer: 8 * 1024 * 1024,
    stdio: ["ignore", "pipe", "ignore"],
  });
  commits = log
    .split("\n")
    .filter(Boolean)
    .map((line) => {
      const [sha, author, subject] = line.split("\t");
      return { sha, author, subject };
    });
} catch (e) {
  // OUTSIDE_CWD or not-a-repo — silently fall back to "(no commits
  // available)" rather than crash. The PDF is still useful.
  commits = [];
}

// ── Document setup ────────────────────────────────────────────────────────
doc = new PDFDocument({
  size: "LETTER",
  margins: { top: 72, bottom: 72, left: 72, right: 72 },
  info: {
    Title: `llmforge ${TAG} release notes`,
    Author: "release.yml",
    Subject: "Containerized deployment summary",
    CreationDate: new Date(DATE),
  },
});

doc.pipe(fs.createWriteStream(path.join(process.cwd(), "release.pdf")));

// ── Cover header ──────────────────────────────────────────────────────────
doc
  .fontSize(28)
  .fillColor("#0f172a")
  .text(`llmforge ${TAG}`, { align: "left" })
  .moveDown(0.3);

doc
  .fontSize(11)
  .fillColor("#475569")
  .text(`Released ${DATE}`)
  .text(`Commit: ${COMMIT}`)
  .text(`Repo: ${REPO || "—"}`)
  .moveDown(1);

doc
  .moveTo(72, doc.y)
  .lineTo(540, doc.y)
  .strokeColor("#cbd5e1")
  .stroke()
  .moveDown(0.5);

doc.fontSize(13).fillColor("#0f172a").text("What was deployed", { underline: true });
doc.moveDown(0.4);

doc.fontSize(10).fillColor("#0f172a");

const images = [
  { name: "ui-router",  ref: ROUTER_REF,  digest: ROUTER_DIGEST,  role: "Landing page + iframe mounts (:3002)" },
  { name: "ui",         ref: UI_REF,      digest: UI_DIGEST,      role: "Heavy UI (:3000, /heavy) + UI_lite (:3001, /lite) under one supervisor" },
  { name: "trainer",    ref: TRAINER_REF, digest: TRAINER_DIGEST, role: "PyTorch + SSH + framework code (GPU, :22)" },
];

for (const img of images) {
  doc.font("Courier-Bold").fontSize(10).text(`pratic2001/llmforge-${img.name}`);
  doc.font("Courier").fontSize(9).fillColor("#334155");
  doc.text(`  ref:    ${img.ref}`);
  doc.text(`  digest: ${img.digest}`);
  doc.text(`  role:   ${img.role}`);
  doc.moveDown(0.6);
  doc.fillColor("#0f172a");
}

// ── Public URLs ───────────────────────────────────────────────────────────
doc.moveDown(0.5);
doc.fontSize(13).fillColor("#0f172a").text("Public URLs", { underline: true });
doc.moveDown(0.4);

doc.font("Courier").fontSize(10).fillColor("#0f172a");
const urls = [
  `https://${HOSTNAME}/`,
  `https://${HOSTNAME}/heavy/`,
  `https://${HOSTNAME}/lite/`,
];
for (const u of urls) doc.text(u);
doc.moveDown(0.8);

// ── What's new ────────────────────────────────────────────────────────────
doc.fillColor("#0f172a").font("Helvetica-Bold").fontSize(13).text("What's new", { underline: true });
doc.moveDown(0.4);

doc.font("Helvetica").fontSize(10).fillColor("#0f172a");

const newItems = [
  "Containerized deployment: three production images on Docker Hub (pratic2001/llmforge-{ui-router,ui,trainer}).",
  "Tailscale sidecar pattern: ui-router, ui, and trainer each get a MagicDNS hostname on your tailnet; ui-router and ui publish :443 via tailscale serve / funnel.",
  "GPU trainer is now a separate container reachable over SSH at hostname `trainer`; the UI's existing node-ssh remote-node path handles it without code changes.",
  "Release workflow (.github/workflows/release.yml) builds and pushes all three images on git tags (v*); a downloadable env-bundle artifact collects the runtime secrets.",
  "Smoke test in CI: compose-up the UI containers on a fresh ubuntu runner and curl the auth endpoints to catch regressions before a tag is pushed.",
];

for (const item of newItems) {
  doc.text(`• ${item}`, { align: "left", indent: 12 });
  doc.moveDown(0.2);
}

doc.moveDown(0.6);

// ── What's changed (commits) ─────────────────────────────────────────────
doc.fillColor("#0f172a").font("Helvetica-Bold").fontSize(13).text("What's changed", { underline: true });
doc.moveDown(0.3);

doc.font("Helvetica").fontSize(10).fillColor("#475569").text(
  PREV
    ? `Commits between ${PREV} and ${TAG} (${commits.length} total):`
    : `First release — listing the most recent ${commits.length} commits overall:`
);
doc.moveDown(0.4);

if (commits.length === 0) {
  doc.fontSize(10).fillColor("#475569").text("(no commits available — checkout was shallow)");
} else {
  doc.font("Courier").fontSize(8.5).fillColor("#0f172a");
  for (const c of commits) {
    // Subject truncated to keep each line on one PDF line.
    const subject = c.subject.length > 90 ? c.subject.slice(0, 87) + "…" : c.subject;
    doc.text(`${c.sha}  ${c.author.padEnd(20, " ")}  ${subject}`);
  }
}

doc.moveDown(0.8);

// ── Deployment instructions ───────────────────────────────────────────────
doc.fillColor("#0f172a").font("Helvetica-Bold").fontSize(13).text("Deployment", { underline: true });
doc.moveDown(0.4);

doc.font("Helvetica").fontSize(10).fillColor("#0f172a");
const steps = [
  "1. Pull the env-bundle artifact from this workflow run and extract it:",
  "      mkdir -p secrets && tar -xzf env-bundle.tar.gz -C secrets",
  "",
  "2. Set TS_AUTHKEY in your shell (or write ./secrets/ts-authkey):",
  "      export TS_AUTHKEY=tskey-xxxxxxxx",
  "",
  "3. Pull the new images and bring the stack up:",
  "      docker compose pull && docker compose up -d",
  "",
  "4. Verify:",
  "      docker compose ps                       # all 6 containers healthy",
  "      docker exec trainer nvidia-smi          # GPU visible",
  `      curl -fsS https://${HOSTNAME}/api/auth/providers | grep -q credentials`,
];

for (const s of steps) doc.font(s.startsWith("      ") ? "Courier" : "Helvetica").text(s);

doc.moveDown(1);

// ── Footer ────────────────────────────────────────────────────────────────
doc
  .fontSize(8)
  .fillColor("#94a3b8")
  .text(
    "Generated by .github/workflows/release.yml — see the run for full logs.",
    72,
    doc.page.height - 56,
    { align: "center", width: doc.page.width - 144 }
  );

doc.end();