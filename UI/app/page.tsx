"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import Link from "next/link";
import {
  ArrowRight,
  Cpu,
  Network,
  Zap,
  BarChart3,
  Server,
  GitBranch,
  MousePointer2,
} from "lucide-react";
import { motion, useMotionValue, useSpring, useTransform } from "framer-motion";
import { usePalette, PaletteSelector } from "@/components/PaletteProvider";

const features = [
  {
    icon: Cpu,
    title: "Multi-Backend Training",
    description: "Torch DDP, DeepSpeed, or Hivemind — choose your distributed backend",
    glow: "glow-primary",
    accent: "primary" as const,
  },
  {
    icon: GitBranch,
    title: "Full Pipeline",
    description: "Tokenizer → Data Packing → Pretrain → SFT → GRPO/DPO in one click",
    glow: "glow-secondary",
    accent: "secondary" as const,
  },
  {
    icon: Server,
    title: "Multi-Node Orchestration",
    description: "Add nodes, auto-audit hardware, mount NFS, launch distributed training",
    glow: "glow-tertiary",
    accent: "tertiary" as const,
  },
  {
    icon: BarChart3,
    title: "Live Monitoring",
    description: "Real-time loss curves, VRAM usage, throughput, and log streaming",
    glow: "glow-accent",
    accent: "accent" as const,
  },
];

function CursorGlow() {
  const { currentPalette } = usePalette();
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  // Spring-smoothed cursor position for soft trailing glow
  const springConfig = { damping: 25, stiffness: 200, mass: 0.5 };
  const sx = useSpring(mouseX, springConfig);
  const sy = useSpring(mouseY, springConfig);

  useEffect(() => {
    const handleMove = (e: MouseEvent) => {
      mouseX.set(e.clientX);
      mouseY.set(e.clientY);
    };
    window.addEventListener("mousemove", handleMove, { passive: true });
    return () => window.removeEventListener("mousemove", handleMove);
  }, [mouseX, mouseY]);

  return (
    <motion.div
      className="pointer-events-none fixed inset-0 z-0"
      style={{ x: sx, y: sy }}
      aria-hidden="true"
    >
      {/* Soft outer glow that follows the cursor */}
      <div
        className="absolute -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full blur-3xl opacity-40"
        style={{
          background: `radial-gradient(circle, hsl(${currentPalette.primaryGlow} / 0.35) 0%, hsl(${currentPalette.secondary} / 0.15) 40%, transparent 70%)`,
        }}
      />
      {/* Tight inner highlight */}
      <div
        className="absolute -translate-x-1/2 -translate-y-1/2 w-32 h-32 rounded-full blur-2xl"
        style={{
          background: `radial-gradient(circle, hsl(${currentPalette.primaryGlow} / 0.6) 0%, transparent 70%)`,
        }}
      />
    </motion.div>
  );
}

function HeroParallax({ children }: { children: React.ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const x = useMotionValue(0);
  const y = useMotionValue(0);

  // Heavier spring = more weight, slightly delayed, reads as "depth"
  const config = { damping: 20, stiffness: 80, mass: 1 };
  const sx = useSpring(x, config);
  const sy = useSpring(y, config);

  // Map cursor offset to small translations per child depth
  const tx1 = useTransform(sx, (v) => v * 0.02);
  const ty1 = useTransform(sy, (v) => v * 0.02);
  const tx2 = useTransform(sx, (v) => v * -0.04);
  const ty2 = useTransform(sy, (v) => v * -0.04);
  const tx3 = useTransform(sx, (v) => v * 0.08);
  const ty3 = useTransform(sy, (v) => v * 0.08);

  const handleMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const rect = ref.current?.getBoundingClientRect();
      if (!rect) return;
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      x.set(e.clientX - cx);
      y.set(e.clientY - cy);
    },
    [x, y]
  );

  const handleLeave = useCallback(() => {
    x.set(0);
    y.set(0);
  }, [x, y]);

  return (
    <div
      ref={ref}
      onMouseMove={handleMove}
      onMouseLeave={handleLeave}
      className="relative"
    >
      <motion.div style={{ x: tx3, y: ty3 }} className="absolute inset-0 pointer-events-none">
        {/* Far back ambient orbs */}
        <div className="absolute top-1/4 left-1/4 w-72 h-72 rounded-full blur-3xl opacity-20"
             style={{ background: `hsl(var(--palette-primary))` }} />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 rounded-full blur-3xl opacity-15"
             style={{ background: `hsl(var(--palette-tertiary))` }} />
      </motion.div>
      <motion.div style={{ x: tx1, y: ty1 }}>{children}</motion.div>
      <motion.div
        style={{ x: tx2, y: ty2 }}
        className="absolute inset-0 pointer-events-none"
        aria-hidden="true"
      >
        {/* Mid-layer floating chips */}
        <div className="absolute top-20 right-10 w-20 h-20 glass-subtle rounded-xl rotate-12 opacity-30" />
        <div className="absolute bottom-32 left-16 w-16 h-16 glass-subtle rounded-full opacity-25" />
      </motion.div>
    </div>
  );
}

export default function HomePage() {
  const { currentPalette } = usePalette();
  const [tilt, setTilt] = useState<Record<string, { x: number; y: number }>>({});

  // Track mouse position per-card for 3D tilt + spotlight effect
  const handleCardMove = useCallback(
    (id: string) => (e: React.MouseEvent<HTMLDivElement>) => {
      const rect = e.currentTarget.getBoundingClientRect();
      const px = (e.clientX - rect.left) / rect.width;
      const py = (e.clientY - rect.top) / rect.height;
      // -1 .. 1
      setTilt((t) => ({ ...t, [id]: { x: (px - 0.5) * 2, y: (py - 0.5) * 2 } }));
    },
    []
  );

  const handleCardLeave = useCallback((id: string) => {
    setTilt((t) => ({ ...t, [id]: { x: 0, y: 0 } }));
  }, []);

  return (
    <>
      <main className="relative min-h-screen flex flex-col overflow-x-hidden">
        {/* Cursor-following glow */}
        <CursorGlow />

        {/* Navigation */}
        <nav className="glass-strong border-b border-border/50 sticky top-0 z-40">
          <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <motion.div
                whileHover={{ rotate: 12, scale: 1.08 }}
                transition={{ type: "spring", stiffness: 300 }}
                className="w-9 h-9 rounded-lg bg-gradient-to-br from-[hsl(var(--palette-primary))] to-[hsl(var(--palette-secondary))] flex items-center justify-center glow-primary"
              >
                <Zap className="w-4 h-4 text-white" />
              </motion.div>
              <span className="font-semibold text-lg tracking-tight">
                LLM<span className="text-glow-primary">Forge</span>
              </span>
            </div>
            <div className="flex items-center gap-3">
              <PaletteSelector />
              <Link
                href="/login"
                className="text-sm text-muted-foreground hover:text-foreground transition-colors hidden sm:inline"
              >
                Sign In
              </Link>
              <Link
                href="/signup"
                className="text-sm px-4 py-2 rounded-lg bg-gradient-to-r from-[hsl(var(--palette-primary))] to-[hsl(var(--palette-secondary))] text-white font-semibold hover:opacity-90 transition-all glow-primary"
              >
                Get Started
              </Link>
            </div>
          </div>
        </nav>

        {/* Hero */}
        <section className="flex-1 flex flex-col items-center justify-center px-6 py-20 md:py-28 text-center relative">
          <HeroParallax>
            <motion.div
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, ease: "easeOut" }}
              className="max-w-4xl mx-auto"
            >
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full glass-strong border border-border/50 text-glow-primary text-xs font-medium mb-8">
                <motion.span
                  animate={{ rotate: [0, 8, -8, 0] }}
                  transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                  className="inline-flex"
                >
                  <Network className="w-3 h-3" />
                </motion.span>
                Advanced LLM Training Framework
              </div>

              <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-6 leading-[1.05]">
                Train LLMs from
                <br />
                <span className="relative inline-block bg-gradient-to-r from-[hsl(var(--palette-primary))] via-[hsl(var(--palette-secondary))] to-[hsl(var(--palette-tertiary))] bg-clip-text text-transparent">
                  Browser to Cluster
                  <motion.span
                    className="absolute -bottom-2 left-0 h-1 rounded-full bg-gradient-to-r from-[hsl(var(--palette-primary))] to-[hsl(var(--palette-secondary))]"
                    initial={{ width: 0 }}
                    animate={{ width: "100%" }}
                    transition={{ duration: 1.2, delay: 0.6, ease: "easeOut" }}
                  />
                </span>
              </h1>

              <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-10 leading-relaxed">
                Orchestrate the full LLM training pipeline — from tokenizer training
                through pretraining, SFT, and RL — across single or multi-node clusters
                with real-time monitoring and one-click deployment.
              </p>

              <div className="flex items-center justify-center gap-4 flex-wrap">
                <Link
                  href="/signup"
                  className="group flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-[hsl(var(--palette-primary))] to-[hsl(var(--palette-secondary))] text-white font-semibold hover:opacity-90 transition-all glow-primary"
                >
                  Start Building
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </Link>
                <Link
                  href="/login"
                  className="px-6 py-3 rounded-xl glass-strong text-foreground font-medium hover:bg-accent/50 transition-all border border-border/50"
                >
                  Sign In
                </Link>
              </div>
            </motion.div>
          </HeroParallax>

          {/* Floating status bar with mouse-following cursor hint */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.4 }}
            className="mt-16 glass-strong rounded-2xl px-8 py-4 flex items-center gap-8 text-sm border border-border/40"
          >
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              <span className="text-muted-foreground">3 Backends</span>
            </div>
            <div className="w-px h-6 bg-border/50" />
            <div className="flex items-center gap-2">
              <motion.div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: `hsl(${currentPalette.primary})` }}
                animate={{ scale: [1, 1.4, 1], opacity: [1, 0.6, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
              <span className="text-muted-foreground">4 Training Stages</span>
            </div>
            <div className="w-px h-6 bg-border/50" />
            <div className="flex items-center gap-2">
              <motion.div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: `hsl(${currentPalette.tertiary})` }}
                animate={{ scale: [1, 1.4, 1], opacity: [1, 0.6, 1] }}
                transition={{ duration: 2, repeat: Infinity, delay: 0.4 }}
              />
              <span className="text-muted-foreground">Multi-Node</span>
            </div>
            <div className="w-px h-6 bg-border/50" />
            <div className="hidden md:flex items-center gap-2 text-glow-primary">
              <MousePointer2 className="w-3 h-3" />
              <span className="text-xs">Move your mouse</span>
            </div>
          </motion.div>
        </section>

        {/* Features */}
        <section className="max-w-7xl mx-auto px-6 pb-24 relative z-10">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {features.map((feature, i) => {
              const Icon = feature.icon;
              const t = tilt[feature.title] || { x: 0, y: 0 };
              return (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.5 + i * 0.1 }}
                  onMouseMove={handleCardMove(feature.title)}
                  onMouseLeave={() => handleCardLeave(feature.title)}
                  className={`glass-strong rounded-xl p-6 ${feature.glow} border border-border/40 cursor-default relative overflow-hidden group`}
                  style={{
                    transform: `perspective(800px) rotateX(${-t.y * 6}deg) rotateY(${t.x * 6}deg)`,
                    transformStyle: "preserve-3d",
                    transition: "transform 0.15s ease-out",
                  }}
                  whileHover={{ scale: 1.03 }}
                >
                  {/* Cursor-tracked spotlight */}
                  <div
                    className="pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                    style={{
                      background: `radial-gradient(circle 200px at ${(t.x + 1) * 50}% ${(t.y + 1) * 50}%, hsla(var(--palette-${feature.accent}) / 0.22), transparent 70%)`,
                    }}
                  />
                  <div
                    className="w-10 h-10 rounded-lg flex items-center justify-center mb-4 relative"
                    style={{
                      background: `hsla(var(--palette-${feature.accent}) / 0.15)`,
                      border: `1px solid hsla(var(--palette-${feature.accent}) / 0.3)`,
                      boxShadow: `0 0 20px hsla(var(--palette-${feature.accent}) / 0.2)`,
                    }}
                  >
                    <Icon
                      className="w-5 h-5"
                      style={{ color: `hsl(var(--palette-${feature.accent}))` }}
                    />
                  </div>
                  <h3 className="font-semibold mb-2 relative">{feature.title}</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed relative">
                    {feature.description}
                  </p>
                </motion.div>
              );
            })}
          </div>
        </section>

        {/* Footer */}
        <footer className="border-t border-border/50 py-6 glass-subtle">
          <div className="max-w-7xl mx-auto px-6 flex items-center justify-between text-xs text-muted-foreground">
            <span>LLM Training Pipeline UI</span>
            <span>Built with Next.js • Prisma • PostgreSQL</span>
          </div>
        </footer>
      </main>
    </>
  );
}