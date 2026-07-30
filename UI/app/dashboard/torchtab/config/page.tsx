"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  PlayCircle,
  Save,
  Settings2,
  ChevronDown,
  ChevronRight,
  Loader2,
  BookTemplate,
  Terminal,
} from "lucide-react";

interface FlagField {
  key: string;
  label: string;
  type: "string" | "number" | "boolean" | "select" | "multiselect";
  default?: string | number | boolean;
  options?: string[];
  description: string;
  required?: boolean;
  group: string;
}

const pretrainFlags: FlagField[] = [
  { key: "model_name", label: "Model Name", type: "string", default: "LLM", description: "Name identifier for the model", required: true, group: "Model" },
  { key: "model_type", label: "Architecture", type: "select", default: "dense", options: ["dense", "jamba", "parallel", "mla", "mamba", "mod", "mtp"], description: "Model architecture variant", required: true, group: "Model" },
  { key: "vocab_size", label: "Vocab Size", type: "number", default: 100352, description: "Vocabulary size", required: true, group: "Model" },
  { key: "hidden_dim", label: "Hidden Dim", type: "number", default: 768, description: "Hidden dimension size", group: "Model" },
  { key: "num_layers", label: "Layers", type: "number", default: 12, description: "Number of transformer layers", group: "Model" },
  { key: "num_heads", label: "Attention Heads", type: "number", default: 12, description: "Number of attention heads", group: "Model" },
  { key: "max_seq_len", label: "Max Sequence Length", type: "number", default: 4096, description: "Maximum sequence length", group: "Model" },
  { key: "batch_size", label: "Batch Size", type: "number", default: 8, description: "Per-device batch size", required: true, group: "Training" },
  { key: "gradient_accumulation", label: "Gradient Accumulation", type: "number", default: 1, description: "Gradient accumulation steps", group: "Training" },
  { key: "learning_rate", label: "Learning Rate", type: "number", default: 3e-4, description: "Peak learning rate", group: "Training" },
  { key: "num_epochs", label: "Epochs", type: "number", default: 1, description: "Number of training epochs", group: "Training" },
  { key: "warmup_steps", label: "Warmup Steps", type: "number", default: 100, description: "LR warmup steps", group: "Training" },
  { key: "weight_decay", label: "Weight Decay", type: "number", default: 0.1, description: "AdamW weight decay", group: "Training" },
  { key: "data_path", label: "Data Path", type: "string", default: "/mnt/training/data", description: "Path to training data", required: true, group: "Data" },
  { key: "output_dir", label: "Output Directory", type: "string", default: "/mnt/training/output", description: "Output directory for checkpoints", group: "Data" },
  { key: "lr_scheduler", label: "LR Scheduler", type: "select", default: "cosine", options: ["cosine", "linear", "warmup_stable_decay"], description: "Learning rate scheduler type", group: "Training" },
];

const sftFlags: FlagField[] = [
  { key: "model_name_or_path", label: "Model Path", type: "string", description: "Path to pretrained model", required: true, group: "Model" },
  { key: "batch_size", label: "Batch Size", type: "number", default: 4, description: "Per-device batch size", required: true, group: "Training" },
  { key: "learning_rate", label: "Learning Rate", type: "number", default: 1e-5, description: "Learning rate for SFT", group: "Training" },
  { key: "num_epochs", label: "Epochs", type: "number", default: 3, description: "Fine-tuning epochs", group: "Training" },
  { key: "data_path", label: "Data Path", type: "string", default: "/mnt/training/data/sft", description: "SFT dataset path", required: true, group: "Data" },
];

const stageScriptMap: Record<string, string> = {
  tokenizer: "tokenizer_train.py",
  packing: "hf_to_packed.py",
  pretrain: "train_pretrain.py",
  sft: "train_sft.py",
  grpo: "train_grpo.py",
  dpo: "train_dpo.py",
};

const allStages = [
  { id: "tokenizer", label: "Tokenizer Train", flags: [] as FlagField[] },
  { id: "packing", label: "Data Packing", flags: [] as FlagField[] },
  { id: "pretrain", label: "Pretrain", flags: pretrainFlags },
  { id: "sft", label: "SFT", flags: sftFlags },
  { id: "grpo", label: "GRPO", flags: [] as FlagField[] },
  { id: "dpo", label: "DPO", flags: [] as FlagField[] },
];

export default function TorchtabConfigPage() {
  const router = useRouter();
  const [config, setConfig] = useState<Record<string, any>>({});
  const [selectedStages, setSelectedStages] = useState<string[]>(["pretrain"]);
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({});
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [extraArgs, setExtraArgs] = useState("");

  const setFlag = (key: string, value: any) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const toggleStage = (id: string) => {
    setSelectedStages((prev) =>
      prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]
    );
  };

  const toggleGroup = (group: string) => {
    setExpandedGroups((prev) => ({ ...prev, [group]: !prev[group] }));
  };

  const activeFlags = allStages
    .filter((s) => selectedStages.includes(s.id))
    .flatMap((s) => s.flags);

  const groups = [...new Set(activeFlags.map((f) => f.group))];

  // Use the first selected stage to determine the script type
  const primaryStage = selectedStages[0] || "pretrain";

  const launchAll = async () => {
    setLaunching(true);
    setError(null);

    try {
      const res = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: stageScriptMap[primaryStage],
          backend: "torch",
          config,
          extraArgs: extraArgs.trim() || undefined,
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to launch job");

      // Navigate to jobs page on success
      router.push("/dashboard/torchtab/jobs");
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLaunching(false);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Configure Pipeline</h1>
          <p className="text-muted-foreground text-sm mt-1">
            Select stages and configure flags for Torch DDP
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button className="inline-flex items-center gap-2 px-3 py-2 rounded-lg glass text-sm hover:bg-accent/50 transition-all">
            <Save className="w-4 h-4" />
            Save Preset
          </button>
          <button className="inline-flex items-center gap-2 px-3 py-2 rounded-lg glass text-sm hover:bg-accent/50 transition-all">
            <BookTemplate className="w-4 h-4" />
            Load Preset
          </button>
        </div>
      </div>

      {/* Pipeline Stages */}
      <div className="glass rounded-xl p-5 border border-border/50">
        <h3 className="font-semibold mb-3">Pipeline Stages</h3>
        <div className="flex flex-wrap gap-2">
          {allStages.map((stage) => {
            const active = selectedStages.includes(stage.id);
            return (
              <button
                key={stage.id}
                onClick={() => toggleStage(stage.id)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  active
                    ? "bg-neon-cyan/10 text-neon-cyan border border-neon-cyan/20"
                    : "bg-accent/30 text-muted-foreground border border-transparent hover:text-foreground"
                }`}
              >
                {stage.label}
              </button>
            );
          })}
        </div>
        <p className="text-xs text-muted-foreground mt-2">
          Select the stages you want to run.
        </p>
      </div>

      {/* Config form */}
      {groups.length === 0 ? (
        <div className="glass rounded-xl p-10 border border-border/50 text-center">
          <Settings2 className="w-6 h-6 text-muted-foreground mx-auto mb-3" />
          <h3 className="font-semibold mb-1">No Configuration Available</h3>
          <p className="text-sm text-muted-foreground">
            Select pipeline stages above to configure their flags.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {groups.map((group) => {
            const groupFlags = activeFlags.filter((f) => f.group === group);
            const expanded = expandedGroups[group] !== false;

            return (
              <motion.div
                key={group}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass rounded-xl border border-border/50 overflow-hidden"
              >
                <button
                  onClick={() => toggleGroup(group)}
                  className="w-full flex items-center gap-2 px-5 py-3 text-sm font-semibold hover:bg-accent/20 transition-all"
                >
                  {expanded ? (
                    <ChevronDown className="w-4 h-4 text-muted-foreground" />
                  ) : (
                    <ChevronRight className="w-4 h-4 text-muted-foreground" />
                  )}
                  {group}
                </button>
                {expanded && (
                  <div className="px-5 pb-5 grid md:grid-cols-2 gap-4">
                    {groupFlags.map((flag) => (
                      <div key={flag.key}>
                        <label className="block text-xs font-medium mb-1.5">
                          {flag.label}
                          {flag.required && <span className="text-red-400 ml-1">*</span>}
                        </label>
                        {flag.type === "boolean" ? (
                          <button
                            onClick={() => setFlag(flag.key, !config[flag.key])}
                            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                              config[flag.key]
                                ? "bg-neon-cyan/10 text-neon-cyan border border-neon-cyan/20"
                                : "bg-accent/30 text-muted-foreground border border-border"
                            }`}
                          >
                            {config[flag.key] ? "Enabled" : "Disabled"}
                          </button>
                        ) : flag.type === "select" ? (
                          <select
                            value={config[flag.key] ?? flag.default ?? ""}
                            onChange={(e) => setFlag(flag.key, e.target.value)}
                            className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-neon-cyan focus:ring-1 focus:ring-neon-cyan outline-none text-sm"
                          >
                            {flag.options?.map((opt) => (
                              <option key={opt} value={opt}>
                                {opt}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <input
                            type={flag.type === "number" ? "number" : "text"}
                            value={config[flag.key] ?? flag.default ?? ""}
                            onChange={(e) =>
                              setFlag(
                                flag.key,
                                flag.type === "number"
                                  ? parseFloat(e.target.value)
                                  : e.target.value
                              )
                            }
                            placeholder={flag.description}
                            className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-neon-cyan focus:ring-1 focus:ring-neon-cyan outline-none text-sm"
                          />
                        )}
                        <p className="text-xs text-muted-foreground mt-1">
                          {flag.description}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </motion.div>
            );
          })}
        </div>
      )}

      {/* Extra CLI Arguments */}
      <div className="glass rounded-xl border border-border/50 overflow-hidden">
        <div className="px-5 py-3 text-sm font-semibold flex items-center gap-2">
          <Terminal className="w-4 h-4 text-muted-foreground" />
          Extra CLI Arguments
        </div>
        <div className="px-5 pb-5">
          <textarea
            value={extraArgs}
            onChange={(e) => setExtraArgs(e.target.value)}
            placeholder={`--gradient_checkpointing
--use_flash_attn_2
--optimizer adamw`}
            rows={4}
            className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-neon-cyan focus:ring-1 focus:ring-neon-cyan outline-none text-sm font-mono resize-y"
          />
          <p className="text-xs text-muted-foreground mt-1">
            These arguments are appended verbatim to the CLI command after the generated flags.
          </p>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="glass rounded-xl p-4 border border-red-500/30 bg-red-500/5">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {/* Run button */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="glass rounded-xl p-6 border border-border/50 text-center"
      >
        <button
          onClick={launchAll}
          disabled={launching}
          className="inline-flex items-center gap-3 px-8 py-3 rounded-xl bg-gradient-to-r from-neon-cyan to-neon-blue text-black font-bold text-lg hover:opacity-90 transition-all disabled:opacity-50 shadow-[0_0_30px_rgba(0,240,255,0.2)]"
        >
          {launching ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Launching...
            </>
          ) : (
            <>
              <PlayCircle className="w-5 h-5" />
              Run Training
            </>
          )}
        </button>
        <p className="text-xs text-muted-foreground mt-3">
          Launches {stageScriptMap[primaryStage]} with Torch DDP backend
          {extraArgs.trim() && " + custom CLI args"}
        </p>
      </motion.div>
    </div>
  );
}
