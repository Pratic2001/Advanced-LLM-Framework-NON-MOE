/**
 * Zod validation schemas for all API request bodies.
 * Centralizes runtime input validation for every route handler.
 */

import { z } from "zod";

// ── Auth ────────────────────────────────────────────────────────────────────

export const registerSchema = z.object({
  email: z.string().email("Invalid email address"),
  password: z
    .string()
    .min(8, "Password must be at least 8 characters")
    .max(128, "Password must be at most 128 characters"),
  name: z.string().min(1).max(255).optional(),
});

// ── Settings ─────────────────────────────────────────────────────────────────

export const updateSettingsSchema = z.object({
  // Empty string clears the setting (stored as null).
  pythonBin: z.string().trim().max(1024, "Interpreter path is too long").optional(),
});

// ── Jobs ─────────────────────────────────────────────────────────────────────

export const backendEnum = z.enum(["torch", "deepspeed", "hivemind"]);

const scriptNameEnum = z.enum([
  "train_pretrain.py",
  "train_sft.py",
  "train_grpo.py",
  "train_dpo.py",
  "tokenizer_train.py",
  "hf_to_packed.py",
]);

export const createJobSchema = z.object({
  type: scriptNameEnum,
  backend: backendEnum,
  config: z.record(z.string(), z.unknown()),
  nodeIds: z.array(z.string().min(1)).optional(),
  extraArgs: z.string().optional(),
});

export const createMetricSchema = z.object({
  step: z.number().int().positive("Step must be a positive integer"),
  loss: z.number().optional(),
  vramUsed: z.number().optional(),
  tokensPerSec: z.number().optional(),
  lr: z.number().optional(),
  gradNorm: z.number().optional(),
  throughput: z.number().optional(),
  nodeName: z.string().optional(),
});

const stageSchema = z.object({
  type: z.string().min(1),
  label: z.string().min(1),
  config: z.record(z.string(), z.unknown()).default({}),
  dependsOn: z.array(z.string()).optional(),
  extraArgs: z.string().optional(),
});

export const pipelineSchema = z.object({
  name: z.string().min(1, "Pipeline name is required").max(255),
  backend: backendEnum,
  stages: z.array(stageSchema).min(1, "At least one stage is required"),
  nodeIds: z.array(z.string().min(1)).optional(),
});

// ── Nodes ────────────────────────────────────────────────────────────────────

export const createNodeSchema = z.object({
  name: z.string().min(1, "Node name is required").max(255),
  host: z.string().min(1, "Host/IP is required"),
  port: z.number().int().positive().optional(),
  username: z.string().min(1, "Username is required"),
  role: z.enum(["HEAD", "WORKER"]).optional(),
  sshKeyId: z.string().optional(),
});

// ── Configs ──────────────────────────────────────────────────────────────────

export const createConfigSchema = z.object({
  name: z.string().min(1, "Config name is required").max(255),
  description: z.string().optional(),
  backend: backendEnum,
  config: z.record(z.string(), z.unknown()),
});

export const updateConfigSchema = createConfigSchema.partial();

// ── NFS Mount ────────────────────────────────────────────────────────────────

export const mountNfsSchema = z.object({
  nfsServer: z.string().min(1, "NFS server is required"),
  exportPath: z.string().min(1, "Export path is required"),
  mountPoint: z.string().optional(),
});

// ── Helper ───────────────────────────────────────────────────────────────────

/**
 * Extract a clean, flat list of field errors from a Zod error.
 */
export function formatZodErrors(
  error: z.ZodError
): Record<string, string[]> {
  return error.flatten().fieldErrors as Record<string, string[]>;
}
