/**
 * Email Notification Service
 * Sends email alerts when training jobs fail, using SMTP via nodemailer.
 *
 * Configure via environment variables (see .env.example):
 *   SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, NOTIFICATION_EMAIL
 */

import nodemailer from "nodemailer";
import { env } from "./env";

// ── Config ────────────────────────────────────────────────────────────

function getMailConfig() {
  return {
    host: env.SMTP_HOST || "",
    port: env.SMTP_PORT || 587,
    user: env.SMTP_USER || "",
    pass: env.SMTP_PASS || "",
    from: env.SMTP_FROM || env.SMTP_USER || "noreply@llmforge.local",
    to: env.NOTIFICATION_EMAIL || "",
  };
}

function isConfigured(): boolean {
  const cfg = getMailConfig();
  return Boolean(cfg.host && cfg.user && cfg.pass && cfg.to);
}

// ── Transporter (lazy singleton) ──────────────────────────────────────

let _transporter: nodemailer.Transporter | null = null;

function getTransporter(): nodemailer.Transporter | null {
  if (_transporter) return _transporter;
  const cfg = getMailConfig();
  if (!cfg.host || !cfg.user || !cfg.pass) return null;

  _transporter = nodemailer.createTransport({
    host: cfg.host,
    port: cfg.port,
    secure: cfg.port === 465,
    auth: { user: cfg.user, pass: cfg.pass },
  });

  return _transporter;
}

// ── Helpers ───────────────────────────────────────────────────────────

function formatTimestamp(ts: Date | string): string {
  const d = typeof ts === "string" ? new Date(ts) : ts;
  return d.toLocaleString("en-US", {
    dateStyle: "full",
    timeStyle: "medium",
    timeZone: "UTC",
  }) + " UTC";
}

function truncateLog(log: string, maxLines = 80): string {
  const lines = log.split("\n");
  if (lines.length <= maxLines) return log;
  const tail = lines.slice(-maxLines);
  return `[... ${lines.length - maxLines} lines suppressed ...]\n\n${tail.join("\n")}`;
}

// ── Template ──────────────────────────────────────────────────────────

function buildFailureEmail(params: {
  jobId: string;
  jobType: string;
  backend: string;
  status: string;
  exitCode: number | null | undefined;
  errorMessage: string | null;
  logTail: string;
  createdAt: Date;
  pipelineName?: string | null;
  userEmail?: string | null;
}) {
  const subject = `[LLMForge] ❌ Job Failed — ${params.jobType}${params.pipelineName ? ` (${params.pipelineName})` : ""}`;

  const logSection = params.logTail
    ? `\n## Log Tail (last 80 lines)\n\n\`\`\`\n${truncateLog(params.logTail)}\n\`\`\``
    : "\n_No log output captured._";

  const errorSection = params.errorMessage
    ? `\n**Error:** ${params.errorMessage}`
    : params.exitCode != null
      ? `\n**Exit Code:** ${params.exitCode}`
      : "";

  const pipelineSection = params.pipelineName
    ? `\n**Pipeline:** ${params.pipelineName}`
    : "";

  const text = `# Job Failure Notification

**Job ID:** ${params.jobId}
**Type:** ${params.jobType}
**Backend:** ${params.backend}
**Status:** ${params.status}
**Created:** ${formatTimestamp(params.createdAt)}
**Failed:** ${formatTimestamp(new Date())}${pipelineSection}${errorSection}
${logSection}

---
This is an automated notification from LLMForge. To manage notification settings, update your SMTP configuration in .env.local.`;

  return { subject, text };
}

// ── Public API ────────────────────────────────────────────────────────

export async function sendJobFailureAlert(params: {
  jobId: string;
  jobType: string;
  backend: string;
  status: string;
  exitCode: number | null | undefined;
  errorMessage: string | null;
  logTail: string;
  createdAt: Date;
  pipelineName?: string | null;
  userEmail?: string | null;
}): Promise<boolean> {
  if (!isConfigured()) {
    // Not configured — silently skip (no spam)
    return false;
  }

  const transporter = getTransporter();
  if (!transporter) return false;

  const cfg = getMailConfig();
  const { subject, text } = buildFailureEmail(params);

  try {
    await transporter.sendMail({
      from: cfg.from,
      to: params.userEmail || cfg.to,
      subject,
      text,
    });
    console.log(`[Mail] Job failure alert sent for ${params.jobId}`);
    return true;
  } catch (err) {
    console.error("[Mail] Failed to send alert:", err);
    return false;
  }
}

export async function sendPipelineFailureAlert(params: {
  pipelineId: string;
  pipelineName: string | null;
  failedStage: string;
  failedJobId: string;
  errorMessage: string | null;
  logTail: string;
  createdAt: Date;
  userEmail?: string | null;
}): Promise<boolean> {
  if (!isConfigured()) return false;

  const transporter = getTransporter();
  if (!transporter) return false;

  const cfg = getMailConfig();

  const logSection = params.logTail
    ? `\n## Log Tail (last 80 lines)\n\n\`\`\`\n${truncateLog(params.logTail)}\n\`\`\``
    : "\n_No log output captured._";

  const errorSection = params.errorMessage
    ? `\n**Error:** ${params.errorMessage}`
    : "";

  const text = `# Pipeline Failure Notification

**Pipeline:** ${params.pipelineName || "(unnamed)"}
**Pipeline ID:** ${params.pipelineId}
**Failed Stage:** ${params.failedStage}
**Failed Job ID:** ${params.failedJobId}
**Created:** ${formatTimestamp(params.createdAt)}
**Failed:** ${formatTimestamp(new Date())}${errorSection}
${logSection}

---
This is an automated notification from LLMForge. To manage notification settings, update your SMTP configuration in .env.local.`;

  try {
    await transporter.sendMail({
      from: cfg.from,
      to: params.userEmail || cfg.to,
      subject: `[LLMForge] ❌ Pipeline Failed — ${params.pipelineName || params.pipelineId}`,
      text,
    });
    console.log(`[Mail] Pipeline failure alert sent for ${params.pipelineId}`);
    return true;
  } catch (err) {
    console.error("[Mail] Failed to send pipeline alert:", err);
    return false;
  }
}
