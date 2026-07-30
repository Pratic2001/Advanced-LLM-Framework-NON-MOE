"use client";

import { useEffect, useRef } from "react";
import Link from "next/link";
import { ArrowRight, Cpu, Network, Zap, BarChart3, Server, GitBranch } from "lucide-react";
import { motion } from "framer-motion";

function ParticleBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    // Particle background is handled by the global canvas
    const canvas = document.getElementById("particle-canvas") as HTMLCanvasElement;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particles: { x: number; y: number; vx: number; vy: number; size: number }[] = [];
    const particleCount = 80;

    for (let i = 0; i < particleCount; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        size: Math.random() * 2 + 1,
      });
    }

    function animate() {
      ctx!.clearRect(0, 0, canvas!.width, canvas!.height);
      ctx!.fillStyle = "rgba(0, 240, 255, 0.4)";

      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0 || p.x > canvas!.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas!.height) p.vy *= -1;

        ctx!.beginPath();
        ctx!.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx!.fill();
      }

      // Draw connections
      ctx!.strokeStyle = "rgba(0, 240, 255, 0.08)";
      ctx!.lineWidth = 0.5;
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 200) {
            ctx!.beginPath();
            ctx!.moveTo(particles[i].x, particles[i].y);
            ctx!.lineTo(particles[j].x, particles[j].y);
            ctx!.stroke();
          }
        }
      }

      requestAnimationFrame(animate);
    }

    animate();

    const handleResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    window.addEventListener("resize", handleResize);

    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return null;
}

const features = [
  {
    icon: Cpu,
    title: "Multi-Backend Training",
    description: "Torch DDP, DeepSpeed, or Hivemind — choose your distributed backend",
    color: "rgba(0, 240, 255, 0.1)",
    border: "rgba(0, 240, 255, 0.3)",
    glow: "neon-glow-cyan",
  },
  {
    icon: GitBranch,
    title: "Full Pipeline",
    description: "Tokenizer → Data Packing → Pretrain → SFT → GRPO/DPO in one click",
    color: "rgba(0, 136, 255, 0.1)",
    border: "rgba(0, 136, 255, 0.3)",
    glow: "neon-glow-blue",
  },
  {
    icon: Server,
    title: "Multi-Node Orchestration",
    description: "Add nodes, auto-audit hardware, mount NFS, launch distributed training",
    color: "rgba(124, 58, 237, 0.1)",
    border: "rgba(124, 58, 237, 0.3)",
    glow: "neon-glow-purple",
  },
  {
    icon: BarChart3,
    title: "Live Monitoring",
    description: "Real-time loss curves, VRAM usage, throughput, and log streaming",
    color: "rgba(0, 240, 255, 0.1)",
    border: "rgba(0, 240, 255, 0.3)",
    glow: "neon-glow-cyan",
  },
];

export default function HomePage() {
  return (
    <>
      <ParticleBackground />
      <main className="relative min-h-screen flex flex-col">
        {/* Navigation */}
        <nav className="glass border-b border-border/50">
          <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-neon-cyan to-neon-blue flex items-center justify-center">
                <Zap className="w-4 h-4 text-white" />
              </div>
              <span className="font-semibold text-lg tracking-tight">
                LLM<span className="text-neon-cyan">Forge</span>
              </span>
            </div>
            <div className="flex items-center gap-4">
              <Link
                href="/login"
                className="text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                Sign In
              </Link>
              <Link
                href="/signup"
                className="text-sm px-4 py-2 rounded-lg bg-neon-cyan/10 text-neon-cyan border border-neon-cyan/30 hover:bg-neon-cyan/20 transition-all neon-glow-cyan"
              >
                Get Started
              </Link>
            </div>
          </div>
        </nav>

        {/* Hero */}
        <section className="flex-1 flex flex-col items-center justify-center px-6 py-24 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="max-w-4xl"
          >
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-neon-cyan/10 border border-neon-cyan/20 text-neon-cyan text-xs font-medium mb-8">
              <Network className="w-3 h-3" />
              Advanced LLM Training Framework
            </div>

            <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-6">
              Train LLMs from
              <br />
              <span className="bg-gradient-to-r from-neon-cyan via-neon-blue to-neon-purple bg-clip-text text-transparent">
                Browser to Cluster
              </span>
            </h1>

            <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-10 leading-relaxed">
              Orchestrate the full LLM training pipeline — from tokenizer training
              through pretraining, SFT, and RL — across single or multi-node clusters
              with real-time monitoring and one-click deployment.
            </p>

            <div className="flex items-center justify-center gap-4">
              <Link
                href="/signup"
                className="group flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-neon-cyan to-neon-blue text-black font-semibold hover:opacity-90 transition-all neon-glow-cyan"
              >
                Start Building
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link
                href="/login"
                className="px-6 py-3 rounded-xl glass-strong text-foreground font-medium hover:bg-accent/50 transition-all"
              >
                Sign In
              </Link>
            </div>
          </motion.div>

          {/* Floating status bar */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="mt-16 glass rounded-2xl px-8 py-4 flex items-center gap-8 text-sm"
          >
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              <span className="text-muted-foreground">3 Backends</span>
            </div>
            <div className="w-px h-6 bg-border" />
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-neon-cyan animate-pulse" />
              <span className="text-muted-foreground">4 Training Stages</span>
            </div>
            <div className="w-px h-6 bg-border" />
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-neon-purple animate-pulse" />
              <span className="text-muted-foreground">Multi-Node</span>
            </div>
          </motion.div>
        </section>

        {/* Features */}
        <section className="max-w-7xl mx-auto px-6 pb-24">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {features.map((feature, i) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: 0.4 + i * 0.1 }}
                  className={`glass rounded-xl p-6 ${feature.glow} hover:scale-[1.02] transition-all cursor-default`}
                >
                  <div
                    className="w-10 h-10 rounded-lg flex items-center justify-center mb-4"
                    style={{ background: feature.color, border: `1px solid ${feature.border}` }}
                  >
                    <Icon className="w-5 h-5" style={{ color: feature.border }} />
                  </div>
                  <h3 className="font-semibold mb-2">{feature.title}</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {feature.description}
                  </p>
                </motion.div>
              );
            })}
          </div>
        </section>

        {/* Footer */}
        <footer className="border-t border-border/50 py-6">
          <div className="max-w-7xl mx-auto px-6 flex items-center justify-between text-xs text-muted-foreground">
            <span>LLM Training Pipeline UI</span>
            <span>Built with Next.js • Prisma • PostgreSQL</span>
          </div>
        </footer>
      </main>
    </>
  );
}
