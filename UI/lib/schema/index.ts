/**
 * Flag Schema Index
 * Single source of truth for all CLI flags across all training scripts
 */

export type { FlagDefinition, FlagGroup, FlagOption, ScriptSchema } from "./types";

export { commonFlags, optimizerFlags } from "./common";
export { architectureFlags } from "./architecture";
export { deepspeedFlags, deepspeedSchema } from "./deepspeed";
export { hivemindFlags, hivemindSchema } from "./hivemind";

export { pretrainSchema } from "./pretrain";
export { sftSchema } from "./sft";
export { grpoSchema } from "./grpo";
export { dpoSchema } from "./dpo";
export { tokenizerSchema } from "./tokenizer";
export { packingSchema } from "./packing";

/**
 * All schemas indexed by script name
 */
export const allSchemas = {
  "tokenizer_train.py": tokenizerSchema,
  "hf_to_packed.py": packingSchema,
  "train_pretrain.py": pretrainSchema,
  "train_sft.py": sftSchema,
  "train_grpo.py": grpoSchema,
  "train_dpo.py": dpoSchema,
} as const;

export type ScriptName = keyof typeof allSchemas;
export type BackendType = "torch" | "deepspeed" | "hivemind";

/**
 * Get flags for a specific backend + script combination
 */
export function getFlags(script: ScriptName, backend?: BackendType) {
  const schema = allSchemas[script];
  if (!schema) return [];

  return schema.groups
    .flatMap((g) => g.flags)
    .filter((f) => {
      if (!backend) return true;
      if (f.backend && f.backend !== backend) return false;
      return true;
    });
}
