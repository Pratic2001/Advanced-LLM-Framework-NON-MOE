/**
 * Terminal helpers — shared between the IntegratedTerminal component and the
 * config pages that mount it. Provides:
 *   - EXAMPLE_EXTRA_ARGS: per-stage prefilled CLI snippets so users can run
 *     train_tokenizer / pack_dataset / train_grpo / train_dpo without leaving
 *     the website.
 *   - formatStatusBadge: maps a JobStatus to a label + tailwind class for the
 *     status pill in the terminal header.
 *   - isTerminalStatus: true when a job has reached a final state (used to
 *     disable the Stop button).
 */

export type JobStatus =
  | "PENDING"
  | "QUEUED"
  | "SETUP"
  | "RUNNING"
  | "COMPLETED"
  | "FAILED"
  | "CANCELLED";

/**
 * Example `extraArgs` prefill for stages whose flags are not exposed in the
 * config form. These are deliberately conservative — they assume a
 * /mnt/training layout and end with `--dry-run` so a smoke-test of the script
 * doesn't accidentally train on real data. Users can edit before launching.
 */
export const EXAMPLE_EXTRA_ARGS: Record<string, string> = {
  tokenizer:
    "--data_path /mnt/training/data\n" +
    "--vocab_size 100352\n" +
    "--tokenizer_type bpe\n" +
    "--output_dir /mnt/training/tokenizer\n" +
    "--min_frequency 2\n" +
    "--dry-run",

  packing:
    "--input_path /mnt/training/data\n" +
    "--output_path /mnt/training/data/packed\n" +
    "--tokenizer_path /mnt/training/tokenizer\n" +
    "--max_length 4096\n" +
    "--packing_strategy balanced\n" +
    "--num_workers 8\n" +
    "--dry-run",

  grpo:
    "--model_name_or_path /mnt/training/output/pretrain\n" +
    "--reward_model_path /mnt/training/reward-model\n" +
    "--data_path /mnt/training/data/grpo\n" +
    "--output_dir /mnt/training/output/grpo\n" +
    "--learning_rate 5e-6\n" +
    "--num_epochs 1\n" +
    "--kl_coef 0.05\n" +
    "--rollouts_per_step 64\n" +
    "--dry-run",

  dpo:
    "--model_name_or_path /mnt/training/output/sft\n" +
    "--data_path /mnt/training/data/dpo\n" +
    "--output_dir /mnt/training/output/dpo\n" +
    "--learning_rate 5e-7\n" +
    "--num_epochs 1\n" +
    "--dpo_beta 0.1\n" +
    "--dry-run",
};

export function getExampleExtraArgs(stageId: string): string | undefined {
  return EXAMPLE_EXTRA_ARGS[stageId];
}

export function formatStatusBadge(status: JobStatus | string): {
  label: string;
  className: string;
  pulsing?: boolean;
} {
  switch (status) {
    case "PENDING":
    case "QUEUED":
    case "SETUP":
      return {
        label: status,
        className: "bg-amber-500/10 text-amber-400 border border-amber-500/20",
      };
    case "RUNNING":
      return {
        label: "RUNNING",
        className:
          "bg-neon-cyan/10 text-neon-cyan border border-neon-cyan/30 status-running",
        pulsing: true,
      };
    case "COMPLETED":
      return {
        label: "COMPLETED",
        className: "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20",
      };
    case "FAILED":
      return {
        label: "FAILED",
        className: "bg-red-500/10 text-red-400 border border-red-500/20",
      };
    case "CANCELLED":
      return {
        label: "CANCELLED",
        className: "bg-orange-500/10 text-orange-400 border border-orange-500/20",
      };
    default:
      return {
        label: status || "UNKNOWN",
        className: "bg-accent/30 text-muted-foreground border border-border",
      };
  }
}

export function isTerminalStatus(status: JobStatus | string): boolean {
  return status === "COMPLETED" || status === "FAILED" || status === "CANCELLED";
}