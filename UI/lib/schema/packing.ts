import type { FlagDefinition } from "./types";

/**
 * Data packing/preprocessing flags
 */
const packingFlags: FlagDefinition[] = [
  {
    key: "input_path",
    label: "Input Directory",
    type: "string",
    description: "Path to raw text/JSONL files for preprocessing",
    group: "Data",
    required: true,
    appliesTo: ["packing"],
  },
  {
    key: "output_path",
    label: "Output Directory",
    type: "string",
    default: "/mnt/training/data/packed",
    description: "Where to save packed .pt files",
    group: "Data",
    appliesTo: ["packing"],
  },
  {
    key: "tokenizer_path",
    label: "Tokenizer Path",
    type: "string",
    description: "Path to trained tokenizer (model file or HuggingFace ID)",
    group: "Tokenizer",
    required: true,
    appliesTo: ["packing"],
  },
  {
    key: "max_length",
    label: "Max Sequence Length",
    type: "number",
    default: 4096,
    description: "Maximum sequence length for packed samples",
    group: "Data",
    appliesTo: ["packing"],
  },
  {
    key: "packing_strategy",
    label: "Packing Strategy",
    type: "select",
    default: "fill",
    options: [
      { label: "Fill (greedy)", value: "fill" },
      { label: "Balanced", value: "balanced" },
    ],
    description: "Strategy for packing sequences into fixed-length chunks",
    group: "Data",
    appliesTo: ["packing"],
  },
  {
    key: "dataset_config",
    label: "Dataset Config (JSON)",
    type: "string",
    description: "Optional JSON configuration for dataset loading",
    group: "Data",
    appliesTo: ["packing"],
  },
  {
    key: "num_workers",
    label: "Dataloader Workers",
    type: "number",
    default: 4,
    description: "Number of worker processes for preprocessing",
    group: "Performance",
    appliesTo: ["packing"],
  },
  {
    key: "shuffle",
    label: "Shuffle Data",
    type: "boolean",
    default: true,
    description: "Shuffle dataset before packing",
    group: "Data",
    appliesTo: ["packing"],
  },
];

export const packingSchema = {
  script: "hf_to_packed.py",
  description: "Tokenize and pack raw text data into pre-tokenized PyTorch tensors (.pt) for efficient training.",
  groups: [
    { label: "Data", flags: packingFlags.filter(f => f.group === "Data") },
    { label: "Tokenizer", flags: packingFlags.filter(f => f.group === "Tokenizer") },
    { label: "Performance", flags: packingFlags.filter(f => f.group === "Performance") },
  ],
};
