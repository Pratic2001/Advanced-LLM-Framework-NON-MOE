/**
 * Type definitions for CLI flag schemas
 */

export interface FlagOption {
  label: string;
  value: string;
}

export interface FlagDefinition {
  /** CLI flag name (e.g., "model_name" becomes "--model_name") */
  key: string;
  /** Human-readable label for UI forms */
  label: string;
  /** Input type */
  type: "string" | "number" | "boolean" | "select" | "multiselect";
  /** Default value */
  default?: string | number | boolean | string[];
  /** Select options */
  options?: FlagOption[];
  /** Description shown as help text */
  description: string;
  /** Whether this flag is required */
  required?: boolean;
  /** Section/group name for UI organization */
  group: string;
  /** Category: which scripts use this flag */
  appliesTo: ("pretrain" | "sft" | "grpo" | "dpo" | "tokenizer" | "packing")[];
  /** CLI argument prefix (default: "--") */
  prefix?: string;
  /** If true, this flag is only shown on the specific backend tab */
  backend?: "torch" | "deepspeed" | "hivemind";
  /** Validation constraints */
  validation?: {
    min?: number;
    max?: number;
    pattern?: string;
    message?: string;
  };
}

export interface FlagGroup {
  label: string;
  flags: FlagDefinition[];
}

export interface ScriptSchema {
  script: string;
  description: string;
  groups: FlagGroup[];
}
