/**
 * Environment variable validation.
 * Uses Zod to validate all env vars at module load time.
 * Import this first in server.ts to fail fast on misconfiguration.
 */

import { z } from "zod";

const envSchema = z.object({
  // Required
  DATABASE_URL: z.string().url("DATABASE_URL must be a valid URL").refine(
    (v) => v.startsWith("postgresql") || v.startsWith("postgres"),
    { message: "DATABASE_URL must start with postgresql://" }
  ),
  NEXTAUTH_URL: z.string().url("NEXTAUTH_URL must be a valid URL"),
  NEXTAUTH_SECRET: z
    .string()
    .min(16, "NEXTAUTH_SECRET must be at least 16 characters"),

  // Optional with defaults
  NODE_ENV: z
    .enum(["development", "production", "test"])
    .default("development"),
  PORT: z.coerce.number().int().positive().default(3000),
  HOSTNAME: z.string().default("localhost"),
  REPO_ROOT: z.string().default("../"),
  SSH_KEY_ENCRYPTION_KEY: z.string().length(32).optional(),

  // SMTP (optional — email notifications are disabled when not configured)
  SMTP_HOST: z.string().optional(),
  SMTP_PORT: z.coerce.number().int().positive().optional(),
  SMTP_USER: z.string().optional(),
  SMTP_PASS: z.string().optional(),
  SMTP_FROM: z.string().optional(),
  NOTIFICATION_EMAIL: z.string().email().optional(),
});

const parsed = envSchema.safeParse(process.env);

if (!parsed.success) {
  console.error("❌ Invalid environment variables:");
  const { fieldErrors } = parsed.error.flatten();
  for (const [key, errors] of Object.entries(fieldErrors)) {
    if (errors) {
      for (const msg of errors) {
        console.error(`   ${key}: ${msg}`);
      }
    }
  }
  process.exit(1);
}

export const env = parsed.data;
