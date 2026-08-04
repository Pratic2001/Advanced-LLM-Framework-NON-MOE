"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  CheckCircle2,
  ChevronRight,
  Terminal,
  Key,
  FolderTree,
  FlaskConical,
  ArrowRight,
  Copy,
  Check,
} from "lucide-react";

const setupSteps = [
  {
    id: "ssh",
    title: "Passwordless SSH",
    icon: Key,
    description: "Generate SSH keys and distribute them to your worker nodes.",
    commands: [
      '# Generate SSH key pair (if you don\'t have one)',
      'ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""',
      '',
      '# Copy to each worker node',
      'ssh-copy-id user@<worker-ip>',
      '',
      '# Verify passwordless login',
      'ssh user@<worker-ip> "echo OK"',
    ],
  },
  {
    id: "nfs",
    title: "Shared Storage (NFS)",
    icon: FolderTree,
    description: "Set up a shared NFS directory for checkpoints, data, and logs.",
    commands: [
      '# On the head node (server):',
      'sudo apt-get install -y nfs-kernel-server',
      'sudo mkdir -p /srv/nfs/training',
      'sudo chown -R $(whoami):$(whoami) /srv/nfs/training',
      'echo "/srv/nfs/training <worker-ip>(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports',
      'sudo exportfs -ra',
      '',
      '# On every node (client):',
      'sudo apt-get install -y nfs-common',
      'sudo mkdir -p /mnt/training',
      'sudo mount -t nfs <head-ip>:/srv/nfs/training /mnt/training',
      'echo "<head-ip>:/srv/nfs/training /mnt/training nfs rw,defaults 0 0" | sudo tee -a /etc/fstab',
    ],
  },
  {
    id: "env",
    title: "Python Environment",
    icon: FlaskConical,
    description: "Set up the Python environment with PyTorch and dependencies.",
    commands: [
      '# Install Python 3.11+ if not available',
      '# Then create a virtual environment:',
      'python3 -m venv .venv',
      'source .venv/bin/activate',
      '',
      '# Install PyTorch with CUDA',
      'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124',
      '',
      '# Install project dependencies',
      'pip install -r requirements.txt',
    ],
  },
];

export default function TorchtabSetupPage() {
  const [completed, setCompleted] = useState<Record<string, boolean>>({});
  const [copiedIndex, setCopiedIndex] = useState<string | null>(null);

  const copyCommands = async (commands: string[], stepId: string) => {
    const text = commands.filter((l) => !l.startsWith("#")).join("\n");
    await navigator.clipboard.writeText(text);
    setCopiedIndex(stepId);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const toggleStep = (id: string) => {
    setCompleted((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const allDone = Object.keys(completed).length === setupSteps.length;

  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      <div>
        <h1 className="text-2xl font-bold">Torch / DDP Setup</h1>
        <p className="text-muted-foreground mt-1 text-sm">
          Follow these steps to prepare your environment for distributed training with PyTorch DDP.
        </p>
      </div>

      {/* Steps */}
      <div className="space-y-4">
        {setupSteps.map((step, index) => {
          const StepIcon = step.icon;
          const done = completed[step.id];
          return (
            <motion.div
              key={step.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`glass rounded-xl border transition-all ${
                done ? "border-green-500/30" : "border-border/50"
              }`}
            >
              <div className="p-5">
                {/* Header */}
                <div className="flex items-start gap-4">
                  <div
                    className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
                      done
                        ? "bg-green-500/20 text-green-400"
                        : "bg-neon-cyan/10 text-neon-cyan"
                    }`}
                  >
                    {done ? (
                      <CheckCircle2 className="w-4 h-4" />
                    ) : (
                      <StepIcon className="w-4 h-4" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <h3 className="font-semibold">{step.title}</h3>
                      <button
                        onClick={() => toggleStep(step.id)}
                        className={`text-xs font-medium px-3 py-1 rounded-full transition-all ${
                          done
                            ? "bg-green-500/10 text-green-400"
                            : "bg-accent/30 text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        {done ? "Done" : "Mark done"}
                      </button>
                    </div>
                    <p className="text-sm text-muted-foreground mt-1">
                      {step.description}
                    </p>
                  </div>
                </div>

                {/* Terminal commands */}
                <div className="mt-4 relative group">
                  <div className="rounded-lg bg-black/50 border border-border/50 overflow-hidden">
                    {/* Terminal header */}
                    <div className="flex items-center justify-between px-4 py-2 bg-accent/30 border-b border-border/50">
                      <div className="flex items-center gap-2">
                        <Terminal className="w-3.5 h-3.5 text-muted-foreground" />
                        <span className="text-xs text-muted-foreground font-mono">
                          {step.id === "ssh"
                            ? "ssh-setup"
                            : step.id === "nfs"
                            ? "nfs-setup"
                            : "env-setup"}
                        </span>
                      </div>
                      <button
                        onClick={() => copyCommands(step.commands, step.id)}
                        className="text-muted-foreground hover:text-foreground transition-all"
                        title="Copy commands"
                      >
                        {copiedIndex === step.id ? (
                          <Check className="w-3.5 h-3.5 text-green-400" />
                        ) : (
                          <Copy className="w-3.5 h-3.5" />
                        )}
                      </button>
                    </div>
                    {/* Commands */}
                    <pre className="p-4 text-xs font-mono leading-relaxed overflow-x-auto">
                      {step.commands.map((line, i) => (
                        <div
                          key={i}
                          className={
                            line.startsWith("#")
                              ? "text-muted-foreground italic"
                              : "text-foreground"
                          }
                        >
                          {line || " "}
                        </div>
                      ))}
                    </pre>
                  </div>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Next steps */}
      {allDone && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass rounded-xl p-6 border border-neon-cyan/30 neon-glow-cyan"
        >
          <div className="flex items-center gap-3 mb-4">
            <CheckCircle2 className="w-5 h-5 text-green-400" />
            <h3 className="font-semibold">All Setup Steps Complete</h3>
          </div>
          <p className="text-sm text-muted-foreground mb-4">
            Your environment is ready. Next steps:
          </p>
          <div className="flex flex-wrap gap-3">
            <a
              href="/dashboard/torchtab/nodes"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg glass text-sm font-medium hover:bg-accent/50 transition-all"
            >
              Add Nodes <ArrowRight className="w-3.5 h-3.5" />
            </a>
            <a
              href="/dashboard/torchtab/config"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg glass text-sm font-medium hover:bg-accent/50 transition-all"
            >
              Configure Pipeline <ArrowRight className="w-3.5 h-3.5" />
            </a>
          </div>
        </motion.div>
      )}
    </div>
  );
}
