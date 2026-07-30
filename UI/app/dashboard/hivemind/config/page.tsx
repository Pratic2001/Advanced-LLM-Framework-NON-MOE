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
} from "lucide-react";

interface FlagField {
  key: string;
  label: string;
  type: "string" | "number" | "boolean" | "select";
  default?: string | number | boolean;
  options?: string[];
  description: string;
  required?: boolean;
  group: string;
}

const hivemindFlags: FlagField[] = [
  { key: "bootstrap_peer", label: "Bootstrap Peer", type: "string", default: "/ip4/0.0.0.0/tcp/31337/p2p/...", description: "Multi-address of the bootstrap peer", required: true, group: "Hivemind" },
  { key: "host_maddrs", label: "Host Address", type: "string", default: "/ip4/0.0.0.0/tcp/31337", description: "Local multi-address to listen on", group: "Hivemind" },
  { key: "announce_maddrs", label: "Announce Address", type: "string", description: "Public multi-address to announce to peers", group: "Hivemind" },
  { key: "dht_port", label: "DHT Port", type: "number", default: 31337, description: "Port for DHT peer discovery", group: "Hivemind" },
  { key: "allreduce_port", label: "AllReduce Port", type: "number", default: 31338, description: "Port for gradient averaging", group: "Hivemind" },
  { key: "compression", label: "Compression", type: "select", default: "float16", options: ["none", "float16", "quantile_8bit", "min_max_8bit", "uniform_8bit"], description: "Gradient compression type for all-reduce", group: "Hivemind" },
  { key: "max_peers", label: "Max Peers", type: "number", default: 32, description: "Maximum number of connected peers", group: "Hivemind" },
  { key: "target_batch_size", label: "Target Batch Size", type: "number", default: 2048, description: "Target batch size across all peers", group: "Hivemind" },
  { key: "gradient_averaging_timeout", label: "AllReduce Timeout", type: "number", default: 30, description: "Timeout (s) for gradient averaging rounds", group: "Hivemind" },
  { key: "matchmaking_time", label: "Matchmaking Time", type: "number", default: 5, description: "Time (s) to wait for peers before averaging", group: "Hivemind" },
];

const modelFlags: FlagField[] = [
  { key: "model_name", label: "Model Name", type: "string", default: "LLM-HIVEMIND", description: "Name identifier", required: true, group: "Model" },
  { key: "model_type", label: "Architecture", type: "select", default: "dense", options: ["dense", "jamba", "parallel", "mla", "mamba", "mod", "mtp"], description: "Model architecture", required: true, group: "Model" },
  { key: "vocab_size", label: "Vocab Size", type: "number", default: 100352, description: "Vocabulary size", required: true, group: "Model" },
  { key: "hidden_dim", label: "Hidden Dim", type: "number", default: 2048, description: "Hidden dimension (smaller for hetero)", group: "Model" },
  { key: "num_layers", label: "Layers", type: "number", default: 16, description: "Number of transformer layers", group: "Model" },
  { key: "num_heads", label: "Attention Heads", type: "number", default: 16, description: "Number of attention heads", group: "Model" },
  { key: "max_seq_len", label: "Max Sequence Length", type: "number", default: 4096, description: "Maximum sequence length", group: "Model" },
];

const trainingFlags: FlagField[] = [
  { key: "learning_rate", label: "Learning Rate", type: "number", default: 1e-4, description: "Learning rate", group: "Training" },
  { key: "num_epochs", label: "Epochs", type: "number", default: 1, description: "Number of training epochs", group: "Training" },
  { key: "warmup_steps", label: "Warmup Steps", type: "number", default: 100, description: "LR warmup steps", group: "Training" },
  { key: "batch_size_per_peer", label: "Batch Size Per Peer", type: "number", default: 4, description: "Local batch size per peer", group: "Training" },
  { key: "data_path", label: "Data Path", type: "string", default: "/mnt/training/data", description: "Dataset path", required: true, group: "Data" },
  { key: "output_dir", label: "Output Directory", type: "string", default: "/mnt/training/output", description: "Output path", group: "Data" },
];

const allGroups = [
  { label: "Hivemind Network", flags: hivemindFlags },
  { label: "Model Architecture", flags: modelFlags },
  { label: "Training", flags: trainingFlags },
];

export default function HivemindConfigPage() {
  const [config, setConfig] = useState<Record<string, any>>({});
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({});
  const [launching, setLaunching] = useState(false);

  const setFlag = (key: string, value: any) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const toggleGroup = (group: string) => {
    setExpandedGroups((prev) => ({ ...prev, [group]: !prev[group] }));
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Configure Hivemind</h1>
          <p className="text-muted-foreground text-sm mt-1">
            Decentralized training settings, peer networking, and model configuration
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
                              ? "bg-neon-purple/10 text-neon-purple border border-neon-purple/20"
                              : "bg-accent/30 text-muted-foreground border border-border"
                          }`}
                        >
                          {config[flag.key] ? "Enabled" : "Disabled"}
                        </button>
                      ) : flag.type === "select" ? (
                        <select
                          value={config[flag.key] ?? flag.default ?? ""}
                          onChange={(e) => setFlag(flag.key, e.target.value)}
                          className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-neon-purple focus:ring-1 focus:ring-neon-purple outline-none text-sm"
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
                          className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-neon-purple focus:ring-1 focus:ring-neon-purple outline-none text-sm"
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

      {/* Decentralized architecture note */}
      <div className="glass rounded-xl p-5 border border-border/50">
        <h3 className="font-semibold text-sm mb-2">How Hivemind Training Works</h3>
        <p className="text-xs text-muted-foreground leading-relaxed">
          Hivemind uses a decentralized architecture where each peer independently computes gradients
          on its local data shard. Periodically, peers synchronize via distributed all-reduce, averaging
          their model parameters. No single node orchestrates the training — the group collectively
          maintains the model state through the DHT. This enables heterogeneous multi-node training
          where peers can join and leave dynamically.
        </p>
      </div>

      {/* Run button */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="glass rounded-xl p-6 border border-border/50 text-center"
      >
        <button
          onClick={() => setLaunching(true)}
          disabled={launching}
          className="inline-flex items-center gap-3 px-8 py-3 rounded-xl bg-gradient-to-r from-neon-purple to-purple-600 text-white font-bold text-lg hover:opacity-90 transition-all disabled:opacity-50"
          style={{ boxShadow: "0 0 30px rgba(124,58,237,0.2)" }}
        >
          {launching ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Launching...
            </>
          ) : (
            <>
              <PlayCircle className="w-5 h-5" />
              Run Hivemind Training
            </>
          )}
        </button>
        <p className="text-xs text-muted-foreground mt-3">
          Launches decentralized training across all configured peers.
        </p>
      </motion.div>
    </div>
  );
}
