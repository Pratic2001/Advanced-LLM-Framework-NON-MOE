"use client";

import { useParams } from "next/navigation";
import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Terminal,
  BarChart3,
  Activity,
  Cpu,
  Clock,
  Play,
  Square,
  Loader2,
  Download,
  AlertTriangle,
} from "lucide-react";

// Placeholder metric chart component
function SimpleChart({
  label,
  color,
  unit,
}: {
  label: string;
  color: string;
  unit: string;
}) {
  const [data, setData] = useState<{ x: number; y: number }[]>([]);

  useEffect(() => {
    const interval = setInterval(() => {
      setData((prev) => {
        const next = [...prev, { x: prev.length, y: Math.random() * 3 + 1 }];
        return next.slice(-50);
      });
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const maxY = Math.max(...data.map((d) => d.y), 1);
  const width = 400;
  const height = 120;
  const points = data.map(
    (d, i) => `${(i / Math.max(data.length - 1, 1)) * width},${height - (d.y / maxY) * height}`
  );
  const polyline = points.join(" ");

  return (
    <div className="glass rounded-xl p-4 border border-border/50">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-medium text-muted-foreground">{label}</span>
        <span className="text-xs text-muted-foreground">{unit}</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-24">
        {data.length > 1 && (
          <>
            <defs>
              <linearGradient id={`grad-${label}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity="0.2" />
                <stop offset="100%" stopColor={color} stopOpacity="0" />
              </linearGradient>
            </defs>
            <polyline
              fill={`url(#grad-${label})`}
              stroke={color}
              strokeWidth="2"
              fillOpacity="0.1"
              points={`0,${height} ${polyline} ${width},${height}`}
            />
          </>
        )}
      </svg>
    </div>
  );
}

// Placeholder log viewer
const sampleLogs = [
  "[2024-01-01 12:00:00] Initializing distributed training...",
  "[2024-01-01 12:00:01] Rank 0: Starting training loop",
  "[2024-01-01 12:00:02] Rank 0: Loaded 10000 training samples",
  "[2024-01-01 12:00:03] [step 1/1000] loss=3.452 lr=3e-04 vram=12.4GB tokens/s=1423",
  "[2024-01-01 12:00:04] [step 2/1000] loss=3.312 lr=3e-04 vram=12.4GB tokens/s=1532",
  "[2024-01-01 12:00:05] [step 3/1000] loss=3.178 lr=3e-04 vram=12.4GB tokens/s=1489",
  "[2024-01-01 12:00:06] [step 4/1000] loss=3.089 lr=3e-04 vram=12.4GB tokens/s=1501",
  "[2024-01-01 12:00:07] [step 5/1000] loss=3.001 lr=2.99e-04 vram=12.4GB tokens/s=1512",
  "[2024-01-01 12:00:08] [step 6/1000] loss=2.945 lr=2.99e-04 vram=12.4GB tokens/s=1498",
  "[2024-01-01 12:00:09] [step 7/1000] loss=2.892 lr=2.99e-04 vram=12.4GB tokens/s=1505",
  "[2024-01-01 12:00:10] [step 8/1000] loss=2.834 lr=2.99e-04 vram=12.4GB tokens/s=1518",
];

export default function JobDetailPage() {
  const params = useParams();
  const jobId = params.jobId as string;

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold">Job Detail</h1>
            <span className="text-xs px-2 py-0.5 rounded-full bg-neon-cyan/10 text-neon-cyan font-mono">
              {jobId?.slice(0, 8)}...
            </span>
          </div>
          <p className="text-muted-foreground text-sm mt-1">Torch DDP • Pretrain</p>
        </div>
        <div className="flex items-center gap-2">
          <button className="inline-flex items-center gap-2 px-3 py-2 rounded-lg glass text-sm hover:bg-accent/50 transition-all">
            <Square className="w-4 h-4" />
            Stop
          </button>
          <button className="inline-flex items-center gap-2 px-3 py-2 rounded-lg glass text-sm hover:bg-accent/50 transition-all">
            <Download className="w-4 h-4" />
            Export
          </button>
        </div>
      </div>

      {/* Status bar */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {[
          { label: "Status", value: "Running", color: "text-neon-cyan" },
          { label: "Epoch", value: "1/3", color: "text-foreground" },
          { label: "Step", value: "8/1000", color: "text-foreground" },
          { label: "Loss", value: "2.834", color: "text-blue-400" },
          { label: "Tokens/s", value: "1,518", color: "text-green-400" },
        ].map((stat) => (
          <div
            key={stat.label}
            className="glass rounded-lg p-3 border border-border/50"
          >
            <p className="text-xs text-muted-foreground">{stat.label}</p>
            <p className={`text-lg font-semibold mt-1 ${stat.color}`}>{stat.value}</p>
          </div>
        ))}
      </div>

      {/* Metrics grid */}
      <div>
        <h3 className="font-semibold mb-3 flex items-center gap-2">
          <Activity className="w-4 h-4 text-neon-cyan" />
          Live Metrics
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          <SimpleChart label="Training Loss" color="#00f0ff" unit="lower is better" />
          <SimpleChart label="VRAM Usage" color="#0088ff" unit="GB" />
          <SimpleChart label="Tokens/sec" color="#7c3aed" unit="higher is better" />
          <SimpleChart label="Learning Rate" color="#ec4899" unit="cosine schedule" />
        </div>
      </div>

      {/* Log viewer */}
      <div>
        <h3 className="font-semibold mb-3 flex items-center gap-2">
          <Terminal className="w-4 h-4 text-neon-cyan" />
          Live Logs
        </h3>
        <div className="rounded-xl border border-border/50 overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2 bg-accent/30 border-b border-border/50">
            <div className="flex items-center gap-2">
              <Terminal className="w-3.5 h-3.5 text-muted-foreground" />
              <span className="text-xs text-muted-foreground font-mono">output.log</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              <span className="text-xs text-muted-foreground">Streaming</span>
            </div>
          </div>
          <pre className="p-4 text-xs font-mono leading-relaxed max-h-96 overflow-y-auto bg-black/50">
            {sampleLogs.map((line, i) => (
              <div
                key={i}
                className={
                  line.includes("loss=")
                    ? "text-foreground"
                    : line.startsWith("[")
                    ? "text-foreground/80"
                    : "text-muted-foreground"
                }
              >
                {line}
              </div>
            ))}
          </pre>
        </div>
      </div>
    </div>
  );
}
