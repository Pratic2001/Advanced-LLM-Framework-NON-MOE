"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { motion } from "framer-motion";

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  color: string;
  baseSize: number;
  phase: number;
  hue?: number;
}

interface MousePosition {
  x: number;
  y: number;
  velocityX: number;
  velocityY: number;
  lastX: number;
  lastY: number;
}

export function ParticleBackground({
  className = "",
  palette = "neon-cyber",
  interactive = true,
}: {
  className?: string;
  palette?: string;
  interactive?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const particlesRef = useRef<Particle[]>([]);
  const mouseRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const [initialized, setInitialized] = useState(false);

  // Palette-specific particle colors
  const paletteColors: Record<string, string[]> = {
    "neon-cyber": ["rgba(0, 240, 255, 0.6)", "rgba(0, 136, 255, 0.5)", "rgba(124, 58, 237, 0.5)", "rgba(236, 72, 153, 0.4)"],
    "aurora-borealis": ["rgba(14, 199, 167, 0.6)", "rgba(0, 240, 255, 0.5)", "rgba(0, 136, 255, 0.5)", "rgba(34, 197, 94, 0.4)"],
    "solar-flare": ["rgba(249, 115, 22, 0.6)", "rgba(239, 68, 68, 0.5)", "rgba(234, 179, 8, 0.5)", "rgba(245, 158, 11, 0.4)"],
    "cosmic-void": ["rgba(192, 132, 252, 0.6)", "rgba(124, 58, 237, 0.5)", "rgba(99, 102, 241, 0.5)", "rgba(217, 70, 239, 0.4)"],
    "matrix-green": ["rgba(34, 197, 94, 0.6)", "rgba(0, 153, 0, 0.5)", "rgba(0, 128, 0, 0.5)", "rgba(234, 179, 8, 0.4)"],
    "ocean-depths": ["rgba(6, 182, 212, 0.6)", "rgba(0, 240, 255, 0.5)", "rgba(14, 199, 167, 0.5)", "rgba(13, 148, 136, 0.4)"],
    "rose-quartz": ["rgba(236, 72, 153, 0.6)", "rgba(244, 63, 94, 0.5)", "rgba(217, 70, 239, 0.5)", "rgba(248, 113, 113, 0.4)"],
    "golden-hour": ["rgba(245, 158, 11, 0.6)", "rgba(249, 115, 22, 0.5)", "rgba(234, 179, 8, 0.5)", "rgba(217, 119, 6, 0.4)"],
  };

  const colors = paletteColors[palette] || paletteColors["neon-cyber"];

  const initParticles = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particleCount = Math.min(120, Math.floor((canvas.width * canvas.height) / 15000));
    const newParticles: Particle[] = [];

    for (let i = 0; i < particleCount; i++) {
      newParticles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.8,
        vy: (Math.random() - 0.5) * 0.8,
        size: Math.random() * 2.5 + 0.5,
        color: colors[Math.floor(Math.random() * colors.length)],
        baseSize: Math.random() * 2.5 + 0.5,
        phase: Math.random() * Math.PI * 2,
      });
    }

    particlesRef.current = newParticles;
  }, [colors]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!interactive) return;
    mouseRef.current = { x: e.clientX, y: e.clientY };
  }, [interactive]);

  const handleTouchMove = useCallback((e: TouchEvent) => {
    if (!interactive) return;
    const touch = e.touches[0];
    if (touch) {
      mouseRef.current = { x: touch.clientX, y: touch.clientY };
    }
  }, [interactive]);

  const animate = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const particles = particlesRef.current;
    const { x: mouseX, y: mouseY } = mouseRef.current;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Update and draw particles
    for (const p of particles) {
      // Mouse attraction/repulsion
      if (interactive) {
        const dx = mouseX - p.x;
        const dy = mouseY - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const maxDist = 180;

        if (dist < maxDist && dist > 0) {
          const force = (1 - dist / maxDist) * 0.08;
          const angle = Math.atan2(dy, dx);
          p.vx += Math.cos(angle) * force;
          p.vy += Math.sin(angle) * force;
        }
      }

      // Apply velocity with damping
      p.vx *= 0.98;
      p.vy *= 0.98;
      p.x += p.vx;
      p.y += p.vy;

      // Breathing animation
      p.phase += 0.015;
      p.size = p.baseSize * (0.85 + Math.sin(p.phase) * 0.15);

      // Boundary wrapping with margin
      const margin = 50;
      if (p.x < -margin) p.x = canvas.width + margin;
      if (p.x > canvas.width + margin) p.x = -margin;
      if (p.y < -margin) p.y = canvas.height + margin;
      if (p.y > canvas.height + margin) p.y = -margin;

      // Draw particle
      ctx.beginPath();
      const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.size * 2);
      gradient.addColorStop(0, p.color);
      gradient.addColorStop(1, "rgba(0, 0, 0, 0)");
      ctx.fillStyle = gradient;
      ctx.arc(p.x, p.y, p.size * 2, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw connections
    ctx.strokeStyle = colors[0].replace(/[\d.]+\)$/, "0.06)");
    ctx.lineWidth = 0.4;

    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 140) {
          const opacity = (1 - dist / 140) * 0.15;
          ctx.strokeStyle = colors[0].replace(/[\d.]+\)$/, `${opacity})`);
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }

    animationRef.current = requestAnimationFrame(animate);
  }, [colors, interactive]);

  useEffect(() => {
    initParticles();
    setInitialized(true);

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("touchmove", handleTouchMove, { passive: true });

    const handleResize = () => initParticles();
    window.addEventListener("resize", handleResize);

    animate();

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("touchmove", handleTouchMove);
      window.removeEventListener("resize", handleResize);
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [initParticles, handleMouseMove, handleTouchMove, animate, interactive]);

  if (!initialized) {
    return (
      <canvas
        ref={canvasRef}
        className={`fixed inset-0 -z-10 pointer-events-none ${className}`}
        style={{ width: "100%", height: "100%" }}
        aria-hidden="true"
      />
    );
  }

  return (
    <canvas
      ref={canvasRef}
      className={`fixed inset-0 -z-10 pointer-events-none ${className}`}
      style={{ width: "100%", height: "100%" }}
      aria-hidden="true"
    />
  );
}

// Alternative: Grid overlay with mouse distortion
export function GridBackground({
  className = "",
  palette = "neon-cyber",
  interactive = true,
}: {
  className?: string;
  palette?: string;
  interactive?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const mouseRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const timeRef = useRef(0);
  const [initialized, setInitialized] = useState(false);

  const paletteColors: Record<string, { primary: string; secondary: string }> = {
    "neon-cyber": { primary: "0, 240, 255", secondary: "0, 136, 255" },
    "aurora-borealis": { primary: "14, 199, 167", secondary: "34, 197, 94" },
    "solar-flare": { primary: "249, 115, 22", secondary: "239, 68, 68" },
    "cosmic-void": { primary: "192, 132, 252", secondary: "124, 58, 237" },
    "matrix-green": { primary: "34, 197, 94", secondary: "0, 153, 0" },
    "ocean-depths": { primary: "6, 182, 212", secondary: "14, 199, 167" },
    "rose-quartz": { primary: "236, 72, 153", secondary: "244, 63, 94" },
    "golden-hour": { primary: "245, 158, 11", secondary: "249, 115, 22" },
  };

  const { primary, secondary } = paletteColors[palette] || paletteColors["neon-cyber"];

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!interactive) return;
    mouseRef.current = { x: e.clientX, y: e.clientY };
  }, [interactive]);

  const animate = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    timeRef.current += 0.003;
    const { x: mouseX, y: mouseY } = mouseRef.current;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const gridSize = 50;
    const distortionRadius = interactive ? 200 : 0;
    const distortionStrength = 15;

    // Draw vertical lines
    ctx.strokeStyle = `rgba(${primary}, 0.04)`;
    ctx.lineWidth = 0.5;

    for (let x = 0; x <= canvas.width; x += gridSize) {
      ctx.beginPath();
      for (let y = 0; y <= canvas.height; y += 5) {
        let offsetX = 0;
        if (interactive) {
          const dx = x - mouseX;
          const dy = y - mouseY;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < distortionRadius) {
            const factor = (1 - dist / distortionRadius) * distortionStrength;
            offsetX = Math.sin(timeRef.current * 2 + y * 0.01) * factor;
          }
        }
        const waveOffset = Math.sin(timeRef.current + y * 0.02) * 2;
        if (y === 0) ctx.moveTo(x + offsetX + waveOffset, y);
        else ctx.lineTo(x + offsetX + waveOffset, y);
      }
      ctx.stroke();
    }

    // Draw horizontal lines
    for (let y = 0; y <= canvas.height; y += gridSize) {
      ctx.beginPath();
      for (let x = 0; x <= canvas.width; x += 5) {
        let offsetY = 0;
        if (interactive) {
          const dx = x - mouseX;
          const dy = y - mouseY;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < distortionRadius) {
            const factor = (1 - dist / distortionRadius) * distortionStrength;
            offsetY = Math.cos(timeRef.current * 2 + x * 0.01) * factor;
          }
        }
        const waveOffset = Math.cos(timeRef.current + x * 0.02) * 2;
        if (x === 0) ctx.moveTo(x, y + offsetY + waveOffset);
        else ctx.lineTo(x, y + offsetY + waveOffset);
      }
      ctx.stroke();
    }

    // Glowing intersection points
    ctx.fillStyle = `rgba(${primary}, 0.15)`;
    for (let x = 0; x <= canvas.width; x += gridSize) {
      for (let y = 0; y <= canvas.height; y += gridSize) {
        let px = x, py = y;
        if (interactive) {
          const dx = x - mouseX;
          const dy = y - mouseY;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < distortionRadius) {
            const factor = (1 - dist / distortionRadius) * distortionStrength;
            px += Math.sin(timeRef.current * 2 + y * 0.01) * factor;
            py += Math.cos(timeRef.current * 2 + x * 0.01) * factor;
          }
        }
        px += Math.sin(timeRef.current + y * 0.02) * 2;
        py += Math.cos(timeRef.current + x * 0.02) * 2;

        ctx.beginPath();
        ctx.arc(px, py, 1.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    animationRef.current = requestAnimationFrame(animate);
  }, [primary, interactive]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    setInitialized(true);

    window.addEventListener("mousemove", handleMouseMove);

    const handleResize = () => {
      if (canvasRef.current) {
        canvasRef.current.width = window.innerWidth;
        canvasRef.current.height = window.innerHeight;
      }
    };
    window.addEventListener("resize", handleResize);

    animate();

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("resize", handleResize);
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [handleMouseMove, animate]);

  return (
    <canvas
      ref={canvasRef}
      className={`fixed inset-0 -z-10 pointer-events-none ${className}`}
      aria-hidden="true"
    />
  );
}

// Combined background with both particles and grid
export function InteractiveBackground({
  className = "",
  palette = "neon-cyber",
}: {
  className?: string;
  palette?: string;
}) {
  return (
    <div className={`fixed inset-0 -z-10 pointer-events-none ${className}`} aria-hidden="true">
      <GridBackground palette={palette} interactive={true} />
      <ParticleBackground palette={palette} interactive={true} />
    </div>
  );
}