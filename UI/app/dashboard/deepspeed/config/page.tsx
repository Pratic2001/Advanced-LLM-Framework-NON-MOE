"use client";

import { useState } from "react";
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
import { IntegratedTerminal } from "@/components/IntegratedTerminal";
import { InteractiveShell } from "@/components/InteractiveShell";
import { buildCommand } from "@/lib/command-builder";

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

const deepspeedFlags: FlagField[] = [
  { key: "zero_stage", label: "ZeRO Stage", type: "select", default: "2", options: ["0", "1", "2", "3"], description: "ZeRO optimization stage", required: true, group: "DeepSpeed" },
  { key: "offload_optimizer", label: "Offload Optimizer", type: "boolean", default: false, description: "Offload optimizer states to CPU/NVMe", group: "DeepSpeed" },
  { key: "offload_param", label: "Offload Parameters", type: "boolean", default: false, description: "Offload parameters to CPU/NVMe (ZeRO-3)", group: "DeepSpeed" },
  { key: "gradient_accumulation", label: "Gradient Accumulation", type: "number", default: 4, description: "Gradient accumulation steps", group: "DeepSpeed" },
  { key: "gradient_clipping", label: "Gradient Clipping", type: "number", default: 1.0, description: "Max gradient norm for clipping", group: "DeepSpeed" },
  { key: "fp16", label: "FP16", type: "boolean", default: true, description: "Enable FP16 mixed precision", group: "DeepSpeed" },
  { key: "bf16", label: "BF16", type: "boolean", default: false, description: "Enable BF16 mixed precision", group: "DeepSpeed" },
  { key: "train_batch_size", label: "Train Batch Size", type: "number", default: 32, description: "Effective training batch size (across all devices)", group: "DeepSpeed" },
  { key: "train_micro_batch_size", label: "Micro Batch Size", type: "number", default: 4, description: "Per-device micro batch size", group: "DeepSpeed" },
  { key: "zero_optimization_stage3_param_persistence_threshold", label: "Param Persistence Threshold", type: "number", default: 0, description: "ZeRO-3 threshold for parameter offload persistence", group: "DeepSpeed" },
];

const modelFlags: FlagField[] = [
  { key: "model_name", label: "Model Name", type: "string", default: "LLM-DS", description: "Name identifier", required: true, group: "Model" },
  { key: "model_type", label: "Architecture", type: "select", default: "dense", options: ["dense", "jamba", "parallel", "mla", "mamba", "mod", "mtp"], description: "Model architecture", required: true, group: "Model" },
  { key: "vocab_size", label: "Vocab Size", type: "number", default: 100352, description: "Vocabulary size", required: true, group: "Model" },
  { key: "hidden_dim", label: "Hidden Dim", type: "number", default: 4096, description: "Hidden dimension", group: "Model" },
  { key: "num_layers", label: "Layers", type: "number", default: 32, description: "Number of transformer layers", group: "Model" },
  { key: "num_heads", label: "Attention Heads", type: "number", default: 32, description: "Number of attention heads", group: "Model" },
  { key: "max_seq_len", label: "Max Sequence Length", type: "number", default: 8192, description: "Maximum sequence length", group: "Model" },
];

const trainingFlags: FlagField[] = [
  { key: "learning_rate", label: "Learning Rate", type: "number", default: 2e-4, description: "Peak learning rate", group: "Training" },
  { key: "num_epochs", label: "Epochs", type: "number", default: 1, description: "Number of training epochs", group: "Training" },
  { key: "warmup_steps", label: "Warmup Steps", type: "number", default: 500, description: "LR warmup steps", group: "Training" },
  { key: "weight_decay", label: "Weight Decay", type: "number", default: 0.1, description: "AdamW weight decay", group: "Training" },
  { key: "lr_scheduler", label: "LR Scheduler", type: "select", default: "cosine", options: ["cosine", "linear", "warmup_stable_decay", "constant"], description: "Learning rate scheduler", group: "Training" },
  { key: "data_path", label: "Data Path", type: "string", default: "/mnt/training/data", description: "Path to training data", required: true, group: "Data" },
  { key: "output_dir", label: "Output Directory", type: "string", default: "/mnt/training/output", description: "Checkpoint output path", group: "Data" },
];

const allGroups = [
  { label: "DeepSpeed Settings", flags: deepspeedFlags },
  { label: "Model Architecture", flags: modelFlags },
  { label: "Training", flags: trainingFlags },
];

export default function DeepSpeedConfigPage() {
  const [config, setConfig] = useState<Record<string, any>>({});
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({});
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [extraArgs, setExtraArgs] = useState("");
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  const setFlag = (key: string, value: any) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const toggleGroup = (group: string) => {
    setExpandedGroups((prev) => ({ ...prev, [group]: !prev[group] }));
  };

  const launchAll = async () => {
    setLaunching(true);
    setError(null);

    try {
      const res = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "train_pretrain.py",
          backend: "deepspeed",
          config,
          extraArgs: extraArgs.trim() || undefined,
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to launch job");

      // Render the integrated terminal inline (no redirect).
      setActiveJobId(data.id);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLaunching(false);
    }
  };

  // Command preview — exactly what the server will run. Updates whenever
  // the form changes so the terminal header is always accurate.
  const commandPreview = buildCommand({
    script: "train_pretrain.py" as any,
    config,
    backend: "deepspeed",
    extraArgs: extraArgs.trim() || undefined,
  });

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Interactive bash shell — always visible at the top of the config page
          so you can run commands (inspect files, kick off ad-hoc tooling, etc.)
          before or while configuring the job. Click "Connect" to spawn a PTY
          session scoped to REPO_ROOT. */}
      <InteractiveShell tone="blue" />

      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Configure DeepSpeed</h1>
          <p className="text-muted-foreground text-sm mt-1">
            ZeRO settings, model architecture, and training hyperparameters
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button className="inline-flex items-center gap-2 px-3 py-2 rounded-lg glass text-sm hover:bg-accent/50 transition-all">
            <Save className="w-4 h-4" />
            Save
          </button>
          <button className="inline-flex items-center gap-2 px-3 py-2 rounded-lg glass text-sm hover:bg-accent/50 transition-all">
            <BookTemplate className="w-4 h-4" />
            Presets
          </button>
        </div>
      </div>

      <div className="space-y-4">
        {allGroups.map((group) => {
          const expanded = expandedGroups[group.label] !== false;
          return (
            <motion.div
              key={group.label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass rounded-xl border border-border/50 overflow-hidden"
            >
              <button
                onClick={() => toggleGroup(group.label)}
                className="w-full flex items-center gap-2 px-5 py-3 text-sm font-semibold hover:bg-accent/20 transition-all"
              >
                {expanded ? (
                  <ChevronDown className="w-4 h-4 text-muted-foreground" />
                ) : (
                  <ChevronRight className="w-4 h-4 text-muted-foreground" />
                )}
                {group.label}
              </button>
              {expanded && (
                <div className="px-5 pb-5 grid md:grid-cols-2 gap-4">
                  {group.flags.map((flag) => (
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
                              ? "bg-neon-blue/10 text-neon-blue border border-neon-blue/20"
                              : "bg-accent/30 text-muted-foreground border border-border"
                          }`}
                        >
                          {config[flag.key] ? "Enabled" : "Disabled"}
                        </button>
                      ) : flag.type === "select" ? (
                        <select
                          value={config[flag.key] ?? flag.default ?? ""}
                          onChange={(e) => setFlag(flag.key, e.target.value)}
                          className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-neon-blue focus:ring-1 focus:ring-neon-blue outline-none text-sm"
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
                          className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-neon-blue focus:ring-1 focus:ring-neon-blue outline-none text-sm"
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
--deepspeed_config ds_config.json`}
            rows={4}
            className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-neon-blue focus:ring-1 focus:ring-neon-blue outline-none text-sm font-mono resize-y"
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
          className="inline-flex items-center gap-3 px-8 py-3 rounded-xl bg-gradient-to-r from-neon-blue to-blue-500 text-white font-bold text-lg hover:opacity-90 transition-all disabled:opacity-50"
          style={{ boxShadow: "0 0 30px rgba(0,136,255,0.2)" }}
        >
          {launching ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Launching...
            </>
          ) : (
            <>
              <PlayCircle className="w-5 h-5" />
              Run DeepSpeed Training
            </>
          )}
        </button>
        <p className="text-xs text-muted-foreground mt-3">
          Launches train_pretrain.py with DeepSpeed launcher
          {extraArgs.trim() && " + custom CLI args"}
        </p>
      </motion.div>

      {/* Integrated terminal — appears below the form once a job is launched. */}
      {activeJobId && (
        <IntegratedTerminal
          key={activeJobId}
          jobId={activeJobId}
          backend="deepspeed"
          commandPreview={commandPreview}
        />
      )}
    </div>
  );
}
