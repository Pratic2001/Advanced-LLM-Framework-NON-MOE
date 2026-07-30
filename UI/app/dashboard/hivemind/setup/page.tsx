"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  CheckCircle2,
  Terminal,
  Key,
  FolderTree,
  FlaskConical,
  ArrowRight,
  Copy,
  Check,
  Globe,
} from "lucide-react";

const setupSteps = [
  {
    id: "ssh",
    title: "Passwordless SSH",
    icon: Key,
    description: "Generate SSH keys and distribute them to your peer nodes.",
    commands: [
      '# Generate SSH key pair (if you don\'t have one)',
      'ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""',
      '',
      '# Copy to each peer node',
      'ssh-copy-id user@<peer-ip>',
      '',
      '# Verify passwordless login',
      'ssh user@<peer-ip> "echo OK"',
    ],
  },
  {
    id: "nfs",
    title: "Shared Storage (NFS)",
    icon: FolderTree,
    description: "Shared storage accessible by all Hivemind peers.",
    commands: [
      '# On the server node:',
      'sudo apt-get install -y nfs-kernel-server',
      'sudo mkdir -p /srv/nfs/training',
      'sudo chown -R $(whoami):$(whoami) /srv/nfs/training',
      'echo "/srv/nfs/training <peer-subnet>(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports',
      'sudo exportfs -ra',
      '',
      '# On every peer node:',
      'sudo apt-get install -y nfs-common',
      'sudo mkdir -p /mnt/training',
      'sudo mount -t nfs <server-ip>:/srv/nfs/training /mnt/training',
    ],
  },
  {
    id: "hivemind",
    title: "Hivemind Setup",
    icon: Globe,
    description: "Install Hivemind and understand peer-to-peer training architecture.",
    commands: [
      '# Activate your Python environment',
      'source .venv/bin/activate',
      '',
      '# Install Hivemind',
      'pip install hivemind',
      '',
      '# Verify installation',
      'python -c "import hivemind; print(hivemind.__version__)"',
      '',
      '# Hivemind uses a decentralized architecture:',
      '# - One node acts as the initial peer (bootstrap)',
      '# - Other nodes connect to the bootstrap peer',
      '# - Once connected, all peers form a DHT network',
      '# - Training state is averaged via all-reduce across peers',
    ],
  },
  {
    id: "ports",
    title: "Network Configuration",
    icon: Terminal,
    description: "Open required ports for Hivemind peer-to-peer communication.",
    commands: [
      '# Hivemind requires the following ports to be open:',
      '#  - 31337: DHT (Distributed Hash Table) for peer discovery',
      '#  - 31338: Gradient averaging via AllReduce',
      '#  - 31339: Training data exchange',
      '',
      '# Open ports with ufw:',
      'sudo ufw allow 31337/tcp',
      'sudo ufw allow 31338/tcp',
      'sudo ufw allow 31339/tcp',
      '',
      '# Or with iptables:',
      'sudo iptables -A INPUT -p tcp --dport 31337 -j ACCEPT',
      'sudo iptables -A INPUT -p tcp --dport 31338 -j ACCEPT',
      'sudo iptables -A INPUT -p tcp --dport 31339 -j ACCEPT',
      '',
      '# Test connectivity between nodes:',
      'nc -zv <peer-ip> 31337',
    ],
  },
  {
    id: "env",
    title: "Python Environment",
    icon: FlaskConical,
    description: "Set up the Python environment with PyTorch and dependencies.",
    commands: [
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

export default function HivemindSetupPage() {
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
        <h1 className="text-2xl font-bold">Hivemind Setup</h1>
        <p className="text-muted-foreground mt-1 text-sm">
          Prepare for decentralized peer-to-peer training with Hivemind. Unlike traditional master-worker
          setups, Hivemind uses a distributed hash table (DHT) for peer discovery and decentralized all-reduce.
        </p>
      </div>

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
                <div className="flex items-start gap-4">
                  <div
                    className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
                      done
                        ? "bg-green-500/20 text-green-400"
                        : "bg-neon-purple/10 text-neon-purple"
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

                <div className="mt-4 relative group">
                  <div className="rounded-lg bg-black/50 border border-border/50 overflow-hidden">
                    <div className="flex items-center justify-between px-4 py-2 bg-accent/30 border-b border-border/50">
                      <div className="flex items-center gap-2">
                        <Terminal className="w-3.5 h-3.5 text-muted-foreground" />
                        <span className="text-xs text-muted-foreground font-mono">
                          {step.id}
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

      {allDone && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass rounded-xl p-6 border border-neon-purple/30"
          style={{ boxShadow: "0 0 20px rgba(124,58,237,0.1)" }}
        >
          <div className="flex items-center gap-3 mb-4">
            <CheckCircle2 className="w-5 h-5 text-green-400" />
            <h3 className="font-semibold">All Setup Steps Complete</h3>
          </div>
          <p className="text-sm text-muted-foreground mb-4">
            Your Hivemind environment is ready. Next step:
          </p>
          <div className="flex flex-wrap gap-3">
            <a
              href="/dashboard/hivemind/nodes"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg glass text-sm font-medium hover:bg-accent/50 transition-all"
            >
              Add Bootstrap & Peers <ArrowRight className="w-3.5 h-3.5" />
            </a>
            <a
              href="/dashboard/hivemind/config"
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
