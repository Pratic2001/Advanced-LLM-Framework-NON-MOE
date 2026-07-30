/**
 * CLI Command Builder
 * Converts config objects into CLI command strings for each training script
 */

import type { ScriptName, BackendType } from "./schema";

export interface BuildCommandInput {
  script: ScriptName;
  config: Record<string, any>;
  backend: BackendType;
  nodes?: string[];
  nodeCount?: number;
  gpuCount?: number;
  pythonBin?: string;
  /** Raw CLI arguments appended verbatim after generated flags */
  extraArgs?: string;
}

/**
 * Build the CLI command from a typed config object
 */
export function buildCommand(input: BuildCommandInput): string {
  const {
    script,
    config,
    backend,
    nodes = [],
    nodeCount = 1,
    gpuCount = 1,
    pythonBin = "python3",
    extraArgs,
  } = input;

  // Base command
  let cmd = "";

  // Handle backend launchers
  if (backend === "deepspeed") {
    const numGpus = gpuCount * Math.max(nodeCount, 1);
    cmd = `deepspeed --num_gpus ${gpuCount} --num_nodes ${nodeCount}`;

    // Add hostfile if multi-node
    if (nodes.length > 1 && nodeCount > 1) {
      cmd += ` --hostfile /mnt/training/hostfile`;
    }

    cmd += ` ${script}`;
  } else if (backend === "torch") {
    const nprocPerNode = gpuCount;
    const nnodes = Math.max(nodeCount, 1);
    cmd = `torchrun --nproc_per_node=${nprocPerNode} --nnodes=${nnodes}`;

    if (nodes.length > 0 && nodeCount > 1) {
      const masterAddr = nodes[0];
      cmd += ` --rdzv_endpoint=${masterAddr}:29500 --rdzv_backend=c10d`;
    }

    cmd += ` ${script}`;
  } else if (backend === "hivemind") {
    // Hivemind uses the script directly with --bootstrap_peer etc
    cmd = `${pythonBin} ${script}`;
  } else {
    cmd = `${pythonBin} ${script}`;
  }

  // Append flags from config
  const flags = buildFlags(config);
  cmd += ` ${flags}`;

  // Append extra CLI args (pip install through raw text)
  if (extraArgs && extraArgs.trim()) {
    cmd += ` ${extraArgs.trim()}`;
  }

  return cmd;
}

/**
 * Build CLI flag string from config object
 */
function buildFlags(config: Record<string, any>): string {
  const parts: string[] = [];

  for (const [key, value] of Object.entries(config)) {
    // Skip null/undefined
    if (value === null || value === undefined) continue;

    const flagName = `--${key}`;

    if (typeof value === "boolean") {
      // Only add boolean flags if they're true (false = default)
      if (value) {
        parts.push(flagName);
      }
    } else if (Array.isArray(value)) {
      // Arrays become multiple --flag values
      for (const item of value) {
        parts.push(`${flagName} ${String(item)}`);
      }
    } else if (typeof value === "number" && !Number.isInteger(value)) {
      // Float — format without unnecessary trailing zeros
      parts.push(`${flagName} ${value}`);
    } else {
      parts.push(`${flagName} ${String(value)}`);
    }
  }

  return parts.join(" ");
}

/**
 * Build the hostfile content for DeepSpeed multi-node
 */
export function buildHostfile(nodes: { host: string; gpuCount: number }[]): string {
  return nodes.map((n) => `${n.host} slots=${n.gpuCount}`).join("\n");
}

/**
 * Get the effective batch size across all devices
 */
export function calcEffectiveBatchSize(
  perDeviceBatchSize: number,
  gradientAccumulation: number,
  gpuCount: number,
  nodeCount: number
): number {
  return perDeviceBatchSize * gradientAccumulation * gpuCount * Math.max(nodeCount, 1);
}

/**
 * Estimate training time (rough) based on config
 * Returns estimated time in seconds
 */
export function estimateTrainingTime(
  totalTokens: number,
  tokensPerSecond: number,
  gpuCount: number,
  nodeCount: number
): number {
  const effectiveThroughput = tokensPerSecond * gpuCount * Math.max(nodeCount, 1);
  return totalTokens / effectiveThroughput;
}
