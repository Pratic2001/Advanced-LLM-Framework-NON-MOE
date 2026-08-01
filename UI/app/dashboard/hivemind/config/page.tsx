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
  Plus,
  Trash2,
  Radio,
  Monitor,
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

interface PeerEntry {
  id: string;
  name: string;
  host: string;
  role: "BOOTSTRAP" | "PEER";
  extraArgs: string;
}

const hivemindFlags: FlagField[] = [
  { key: "bootstrap_peer", label: "Bootstrap Peer Multi-Address", type: "string", default: "/ip4/0.0.0.0/tcp/31337/p2p/...", description: "Multi-address of the bootstrap peer for DHT discovery", required: true, group: "Hivemind" },
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
  const router = useRouter();
  const [config, setConfig] = useState<Record<string, any>>({});
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({});
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [extraArgs, setExtraArgs] = useState("");
  const [peers, setPeers] = useState<PeerEntry[]>([]);
  const [showAddPeer, setShowAddPeer] = useState(false);
  const [newPeer, setNewPeer] = useState({ name: "", host: "", role: "PEER" as "BOOTSTRAP" | "PEER" });

  const setFlag = (key: string, value: any) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const toggleGroup = (group: string) => {
    setExpandedGroups((prev) => ({ ...prev, [group]: !prev[group] }));
  };

  const addPeer = () => {
    if (!newPeer.name || !newPeer.host) return;
    const peer: PeerEntry = {
      id: Date.now().toString(),
      name: newPeer.name,
      host: newPeer.host,
      role: newPeer.role,
      extraArgs: newPeer.role === "BOOTSTRAP"
        ? "--host_maddrs /ip4/0.0.0.0/tcp/31337"
        : `--bootstrap_peer /ip4/${newPeer.host}/tcp/31337`,
    };
    setPeers((prev) => [...prev, peer]);
    setNewPeer({ name: "", host: "", role: "PEER" });
    setShowAddPeer(false);
  };

  const removePeer = (id: string) => {
    setPeers((prev) => prev.filter((p) => p.id !== id));
  };

  const updatePeerExtraArgs = (id: string, args: string) => {
    setPeers((prev) => prev.map((p) => (p.id === id ? { ...p, extraArgs: args } : p)));
  };

  const launchAll = async () => {
    if (peers.length === 0) {
      setError("Add at least one peer before launching.");
      return;
    }

    setLaunching(true);
    setError(null);

    const globalExtra = extraArgs.trim();
    const results: { name: string; success: boolean; error?: string }[] = [];

    for (const peer of peers) {
      // Combine global extra args with per-peer extra args
      const peerExtra = [globalExtra, peer.extraArgs.trim()].filter(Boolean).join("\n");

      try {
        const res = await fetch("/api/jobs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            type: "train_pretrain.py",
            backend: "hivemind",
            config: {
              ...config,
              peer_name: peer.name,
              peer_role: peer.role,
              peer_host: peer.host,
            },
            extraArgs: peerExtra || undefined,
          }),
        });

        const data = await res.json();
        if (res.ok) {
          results.push({ name: peer.name, success: true });
        } else {
          results.push({ name: peer.name, success: false, error: data.error || "Unknown error" });
        }
      } catch (err: any) {
        results.push({ name: peer.name, success: false, error: err.message });
      }
    }

    const allSucceeded = results.every((r) => r.success);
    if (allSucceeded) {
      router.push("/dashboard/hivemind/jobs");
    } else {
      const failed = results.filter((r) => !r.success).map((r) => `${r.name}: ${r.error}`);
      setError(`Failed to launch some peers:\n${failed.join("\n")}`);
    }

    setLaunching(false);
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

      {/* Config groups */}
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
                          className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-[hsl(var(--palette-primary))] focus:ring-1 focus:ring-[hsl(var(--palette-primary))] outline-none text-sm"
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
                          className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-[hsl(var(--palette-primary))] focus:ring-1 focus:ring-[hsl(var(--palette-primary))] outline-none text-sm"
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

      {/* Global Extra CLI Arguments */}
      <div className="glass rounded-xl border border-border/50 overflow-hidden">
        <div className="px-5 py-3 text-sm font-semibold flex items-center gap-2">
          <Terminal className="w-4 h-4 text-muted-foreground" />
          Global Extra CLI Arguments
          <span className="text-xs text-muted-foreground font-normal">(applied to all peers)</span>
        </div>
        <div className="px-5 pb-5">
          <textarea
            value={extraArgs}
            onChange={(e) => setExtraArgs(e.target.value)}
            placeholder={`--compression float16
--max_peers 32
--verbose`}
            rows={3}
            className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-[hsl(var(--palette-primary))] focus:ring-1 focus:ring-[hsl(var(--palette-primary))] outline-none text-sm font-mono resize-y"
          />
          <p className="text-xs text-muted-foreground mt-1">
            These arguments are appended to every peer's CLI command.
          </p>
        </div>
      </div>

      {/* Per-Peer Configuration */}
      <div className="glass rounded-xl border border-border/50 overflow-hidden">
        <div className="px-5 py-3 text-sm font-semibold flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Radio className="w-4 h-4 text-muted-foreground" />
            Peer Configuration
          </div>
          <button
            onClick={() => setShowAddPeer(true)}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gradient-to-r from-[hsl(var(--palette-primary))] to-[hsl(var(--palette-secondary))] text-white text-xs font-medium hover:opacity-90 transition-all glow-primary"
          >
            <Plus className="w-3.5 h-3.5" />
            Add Peer
          </button>
        </div>

        {/* Add peer form */}
        {showAddPeer && (
          <div className="px-5 pb-3">
            <div className="grid md:grid-cols-4 gap-3 mb-3">
              <div>
                <label className="block text-xs font-medium mb-1">Name</label>
                <input
                  type="text"
                  value={newPeer.name}
                  onChange={(e) => setNewPeer({ ...newPeer, name: e.target.value })}
                  placeholder="peer-1"
                  className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-[hsl(var(--palette-primary))] focus:ring-1 focus:ring-[hsl(var(--palette-primary))] outline-none text-sm"
                />
              </div>
              <div>
                <label className="block text-xs font-medium mb-1">Host / IP</label>
                <input
                  type="text"
                  value={newPeer.host}
                  onChange={(e) => setNewPeer({ ...newPeer, host: e.target.value })}
                  placeholder="192.168.1.100"
                  className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-[hsl(var(--palette-primary))] focus:ring-1 focus:ring-[hsl(var(--palette-primary))] outline-none text-sm"
                />
              </div>
              <div>
                <label className="block text-xs font-medium mb-1">Role</label>
                <select
                  value={newPeer.role}
                  onChange={(e) => setNewPeer({ ...newPeer, role: e.target.value as "BOOTSTRAP" | "PEER" })}
                  className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-[hsl(var(--palette-primary))] focus:ring-1 focus:ring-[hsl(var(--palette-primary))] outline-none text-sm"
                >
                  <option value="PEER">Worker Peer</option>
                  <option value="BOOTSTRAP">Bootstrap Peer</option>
                </select>
              </div>
              <div className="flex items-end gap-2">
                <button
                  onClick={addPeer}
                  disabled={!newPeer.name || !newPeer.host}
                  className="px-4 py-2 rounded-lg bg-gradient-to-r from-[hsl(var(--palette-primary))] to-[hsl(var(--palette-secondary))] text-white text-sm font-medium hover:opacity-90 transition-all disabled:opacity-50 glow-primary"
                >
                  Add
                </button>
                <button
                  onClick={() => setShowAddPeer(false)}
                  className="px-4 py-2 rounded-lg glass text-sm font-medium hover:bg-accent/50 transition-all"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Peer list */}
        <div className="px-5 pb-5 space-y-3">
          {peers.length === 0 ? (
            <div className="text-center py-6">
              <Monitor className="w-6 h-6 text-muted-foreground mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">
                No peers configured. Add peers with their specific CLI arguments.
              </p>
            </div>
          ) : (
            peers.map((peer) => (
              <div
                key={peer.id}
                className="rounded-lg border border-border/50 bg-accent/20 p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {peer.role === "BOOTSTRAP" ? (
                      <Radio className="w-4 h-4 text-neon-purple" />
                    ) : (
                      <Monitor className="w-4 h-4 text-muted-foreground" />
                    )}
                    <span className="font-medium text-sm">{peer.name}</span>
                    <span className="text-xs font-mono text-muted-foreground">{peer.host}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full ${
                      peer.role === "BOOTSTRAP"
                        ? "bg-neon-purple/10 text-neon-purple"
                        : "bg-accent/30 text-muted-foreground"
                    }`}>
                      {peer.role}
                    </span>
                  </div>
                  <button
                    onClick={() => removePeer(peer.id)}
                    className="text-muted-foreground hover:text-red-400 transition-all"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
                <label className="block text-xs font-medium text-muted-foreground mb-1">
                  Peer-specific CLI Arguments
                </label>
                <textarea
                  value={peer.extraArgs}
                  onChange={(e) => updatePeerExtraArgs(peer.id, e.target.value)}
                  placeholder={peer.role === "BOOTSTRAP"
                    ? "--host_maddrs /ip4/0.0.0.0/tcp/31337"
                    : `--bootstrap_peer /ip4/${peer.host}/tcp/31337/p2p/...`}
                  rows={2}
                  className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-[hsl(var(--palette-primary))] focus:ring-1 focus:ring-[hsl(var(--palette-primary))] outline-none text-xs font-mono resize-y"
                />
              </div>
            ))
          )}
        </div>
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

      {/* Error display */}
      {error && (
        <div className="glass rounded-xl p-4 border border-red-500/30 bg-red-500/5">
          <p className="text-sm text-red-400 whitespace-pre-line">{error}</p>
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
          className="inline-flex items-center gap-3 px-8 py-3 rounded-xl bg-gradient-to-r from-[hsl(var(--palette-primary))] to-[hsl(var(--palette-secondary))] text-white font-bold text-lg hover:opacity-90 transition-all disabled:opacity-50 glow-primary"
        >
          {launching ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Launching {peers.length} Peer{peers.length > 1 ? "s" : ""}...
            </>
          ) : (
            <>
              <PlayCircle className="w-5 h-5" />
              Run Hivemind Training
            </>
          )}
        </button>
        <p className="text-xs text-muted-foreground mt-3">
          {peers.length > 0
            ? `Launches training across ${peers.length} peer${peers.length > 1 ? "s" : ""}`
            : "Add at least one peer to launch"}
        </p>
      </motion.div>
    </div>
  );
}
