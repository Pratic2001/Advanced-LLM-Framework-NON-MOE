import type { FlagDefinition } from "./types";

/**
 * Tokenizer training-specific flags
 */
const tokenizerFlags: FlagDefinition[] = [
  {
    key: "data_path",
    label: "Data Path",
    type: "string",
    description: "Path to text corpus for tokenizer training",
    group: "Data",
    required: true,
    appliesTo: ["tokenizer"],
  },
  {
    key: "vocab_size",
    label: "Vocabulary Size",
    type: "number",
    default: 100352,
    description: "Target vocabulary size for the tokenizer",
    group: "Tokenizer",
    required: true,
    appliesTo: ["tokenizer"],
  },
  {
    key: "tokenizer_type",
    label: "Tokenizer Type",
    type: "select",
    default: "bpe",
    options: [
      { label: "BPE", value: "bpe" },
      { label: "WordPiece", value: "wordpiece" },
      { label: "Unigram", value: "unigram" },
      { label: "SentencePiece (BPE)", value: "sentencepiece_bpe" },
      { label: "SentencePiece (Unigram)", value: "sentencepiece_unigram" },
    ],
    description: "Tokenizer algorithm type",
    group: "Tokenizer",
    appliesTo: ["tokenizer"],
  },
  {
    key: "output_dir",
    label: "Output Directory",
    type: "string",
    default: "/mnt/training/tokenizer",
    description: "Where to save the trained tokenizer files",
    group: "General",
    appliesTo: ["tokenizer"],
  },
  {
    key: "min_frequency",
    label: "Min Frequency",
    type: "number",
    default: 2,
    description: "Minimum token frequency for inclusion in vocabulary",
    group: "Tokenizer",
    appliesTo: ["tokenizer"],
  },
  {
    key: "special_tokens",
    label: "Special Tokens (JSON)",
    type: "string",
    default: '{"pad_token":"<pad>","bos_token":"<s>","eos_token":"</s>","unk_token":"<unk>"}',
    description: "JSON map of special tokens",
    group: "Tokenizer",
    appliesTo: ["tokenizer"],
  },
];

export const tokenizerSchema = {
  script: "tokenizer_train.py",
  description: "Train a tokenizer (BPE/WordPiece/Unigram) on a text corpus for LLM training.",
  groups: [
    { label: "Tokenizer", flags: tokenizerFlags.filter(f => f.group === "Tokenizer") },
    { label: "Data", flags: tokenizerFlags.filter(f => f.group === "Data") },
    { label: "General", flags: tokenizerFlags.filter(f => f.group === "General") },
  ],
};
