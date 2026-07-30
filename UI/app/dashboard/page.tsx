"use client";

import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  Cpu,
  Waves,
  Globe,
  Activity,
  PlayCircle,
  Server,
  BarChart3,
  AlertTriangle,
  CheckCircle2,
  Clock,
} from "lucide-react";

const quickLinks = [
  {
    label: "Torch / DDP",
    href: "/dashboard/torchtab",
    icon: Cpu,
    color: "#00f0ff",
    gradient: "from-neon-cyan to-cyan-500",
  },
  {
    label: "DeepSpeed",
    href: "/dashboard/deepspeed",
    icon: Waves,
    color: "#0088ff",
    gradient: "from-neon-blue to-blue-500",
  },
  {
    label: "Hivemind",
    href: "/dashboard/hivemind",
    icon: Globe,
    color: "#7c3aed",
    gradient: "from-neon-purple to-purple-500",
  },
];

const statsCards = [
  { label: "Total Jobs", value: "0", icon: Activity, color: "text-neon-cyan" },
  { label: "Active Runs", value: "0", icon: PlayCircle, color: "text-green-400" },
  { label: "Connected Nodes", value: "0", icon: Server, color: "text-neon-blue" },
  { label: "Config Presets", value: "0", icon: BarChart3, color: "text-neon-purple" },
];

export default function DashboardHome() {
  const { data: session } = useSession();
  const [gpuInfo, setGpuInfo] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/system/audit")
      .then((r) => r.json())
      .then((d) => setGpuInfo(d.gpu || null))
      .catch(() => setGpuInfo(null));
  }, []);

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Welcome banner */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass rounded-2xl p-6 md:p-8 neon-glow-cyan"
      >
        <h1 className="text-2xl md:text-3xl font-bold mb-2">
          Welcome{session?.user?.name ? `, ${session.user.name}` : " back"}
        </h1>
        <p className="text-muted-foreground max-w-2xl">
          Launch and monitor your LLM training jobs across Torch DDP, DeepSpeed, and Hivemind backends.
          Configure flags, add nodes, and orchestrate the full pipeline from one place.
        </p>
        <div className="flex flex-wrap gap-3 mt-6">
          <Link
            href="/dashboard/torchtab/config"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-neon-cyan to-neon-blue text-black font-semibold text-sm hover:opacity-90 transition-all"
          >
            <PlayCircle className="w-4 h-4" />
            Launch Training
          </Link>
          <Link
            href="/dashboard/torchtab/setup"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg glass text-sm font-medium hover:bg-accent/50 transition-all"
          >
            <Server className="w-4 h-4" />
            Setup Nodes
          </Link>
        </div>
      </motion.div>

      {/* Stats cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {statsCards.map((card, i) => {
          const Icon = card.icon;
          return (
            <motion.div
              key={card.label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              className="glass rounded-xl p-4 border border-border/50"
            >
              <div className="flex items-center justify-between mb-3">
                <Icon className={`w-4 h-4 ${card.color}`} />
              </div>
              <p className="text-2xl font-bold">{card.value}</p>
              <p className="text-xs text-muted-foreground mt-1">{card.label}</p>
            </motion.div>
          );
        })}
      </div>

      {/* Quick start */}
      <div>
        <h2 className="text-lg font-semibold mb-4">Quick Start</h2>
        <div className="grid md:grid-cols-3 gap-4">
          {quickLinks.map((link, i) => {
            const Icon = link.icon;
            return (
              <motion.div
                key={link.label}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 + i * 0.05 }}
              >
                <Link href={link.href}>
                  <div
                    className="glass rounded-xl p-5 border border-border/50 hover:scale-[1.02] transition-all cursor-pointer group"
                  >
                    <div
                      className={`w-10 h-10 rounded-lg bg-gradient-to-br ${link.gradient} flex items-center justify-center mb-4`}
                    >
                      <Icon className="w-5 h-5 text-white" />
                    </div>
                    <h3 className="font-semibold mb-1 group-hover:text-neon-cyan transition-colors">
                      {link.label}
                    </h3>
                    <p className="text-sm text-muted-foreground">
                      Configure and launch training jobs
                    </p>
                  </div>
                </Link>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Local GPU status */}
      {gpuInfo && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="glass rounded-xl p-5 border border-border/50"
        >
          <h3 className="font-semibold mb-3 flex items-center gap-2">
            <Cpu className="w-4 h-4 text-neon-cyan" />
            Local GPU
          </h3>
          <pre className="text-xs text-muted-foreground font-mono">{gpuInfo}</pre>
        </motion.div>
      )}

      {/* Empty state */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        className="glass rounded-xl p-10 border border-border/50 text-center"
      >
        <div className="w-16 h-16 rounded-full bg-accent/30 flex items-center justify-center mx-auto mb-4">
          <AlertTriangle className="w-6 h-6 text-muted-foreground" />
        </div>
        <h3 className="font-semibold mb-2">No Active Jobs</h3>
        <p className="text-sm text-muted-foreground max-w-md mx-auto">
          You haven&apos;t launched any training jobs yet. Start by selecting a backend and configuring your pipeline.
        </p>
        <div className="flex items-center justify-center gap-3 mt-6">
          {quickLinks.map((link) => {
            const Icon = link.icon;
            return (
              <Link
                key={link.label}
                href={link.href}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg glass text-sm hover:bg-accent/50 transition-all"
              >
                <Icon className="w-3.5 h-3.5" />
                {link.label}
              </Link>
            );
          })}
        </div>
      </motion.div>
    </div>
  );
}
