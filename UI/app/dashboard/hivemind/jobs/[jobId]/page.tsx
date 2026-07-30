"use client";

import { useParams } from "next/navigation";
import { useState, useEffect } from "react";
import {
  Terminal,
  Activity,
  Globe,
  Loader2,
  Square,
  Download,
} from "lucide-react";

// Peer metric mini-chart
function PeerMetric({ peerName, color }: { peerName: string; color: string }) {
  const [data, setData] = useState<number[]>([]);

  useEffect(() => {
    const interval = setInterval(() => {
      setData((prev) => [...prev.slice(-20), Math.random() * 2 + 1]);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const maxY = Math.max(...data, 1);
  const width = 200;
  const height = 60;
  const points = data
    .map((d, i) => `${(i / Math.max(data.length - 1, 1)) * width},${height - (d / maxY) * height}`)
    .join(" ");

  return (
    <div className="glass rounded-lg p-3 border border-border/50">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium">{peerName}</span>
        <div className="w-2 h-2 rounded-full bg-green-500" />
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-10">
        {data.length > 1 && (
          <>
            <defs>
              <linearGradient id={`peer-grad-${peerName}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity="0.2" />
                <stop offset="100%" stopColor={color} stopOpacity="0" />
              </linearGradient>
            </defs>
            <polyline
              fill={`url(#peer-grad-${peerName})`}
              stroke={color}
              strokeWidth="2"
              fillOpacity="0.1"
              points={`0,${height} ${points} ${width},${height}`}
            />
          </>
        )}
      </svg>
    </div>
  );
}

export default function HivemindJobDetailPage() {
  const params = useParams();
  const jobId = params.jobId as string;

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold">Hivemind Job</h1>
            <span className="text-xs px-2 py-0.5 rounded-full bg-neon-purple/10 text-neon-purple font-mono">
              {jobId?.slice(0, 8)}...
            </span>
          </div>
          <p className="text-muted-foreground text-sm mt-1">Hivemind • Pretrain</p>
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

      {/* Global stats */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {[
          { label: "Status", value: "Running", color: "text-neon-purple" },
          { label: "Connected Peers", value: "3", color: "text-green-400" },
          { label: "Global Step", value: "42", color: "text-foreground" },
          { label: "Avg Loss", value: "2.834", color: "text-blue-400" },
          { label: "Tokens/s (total)", value: "4,554", color: "text-green-400" },
        ].map((stat) => (
          <div key={stat.label} className="glass rounded-lg p-3 border border-border/50">
            <p className="text-xs text-muted-foreground">{stat.label}</p>
            <p className={`text-lg font-semibold mt-1 ${stat.color}`}>{stat.value}</p>
          </div>
        ))}
      </div>

      {/* Peer metrics */}
      <div>
        <h3 className="font-semibold mb-3 flex items-center gap-2">
          <Globe className="w-4 h-4 text-neon-purple" />
          Peer Metrics
        </h3>
        <div className="grid md:grid-cols-3 gap-3">
          <PeerMetric peerName="bootstrap-node" color="#7c3aed" />
          <PeerMetric peerName="worker-1" color="#00f0ff" />
          <PeerMetric peerName="worker-2" color="#0088ff" />
        </div>
      </div>

      {/* Log viewer */}
      <div>
        <h3 className="font-semibold mb-3 flex items-center gap-2">
          <Terminal className="w-4 h-4 text-neon-purple" />
          Live Logs
        </h3>
        <div className="rounded-xl border border-border/50 overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2 bg-accent/30 border-b border-border/50">
            <div className="flex items-center gap-2">
              <Terminal className="w-3.5 h-3.5 text-muted-foreground" />
              <span className="text-xs text-muted-foreground font-mono">hivemind.log</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              <span className="text-xs text-muted-foreground">Streaming</span>
            </div>
          </div>
          <pre className="p-4 text-xs font-mono leading-relaxed max-h-96 overflow-y-auto bg-black/50">
            <div className="text-muted-foreground">[2024-01-01 12:00:00] Initializing Hivemind peer...</div>
            <div className="text-muted-foreground">[2024-01-01 12:00:01] Listening on /ip4/0.0.0.0/tcp/31337</div>
            <div className="text-muted-foreground">[2024-01-01 12:00:02] Connected to bootstrap peer</div>
            <div className="text-muted-foreground">[2024-01-01 12:00:03] DHT peers: 3</div>
            <div className="text-foreground">[2024-01-01 12:00:04] [step 1] loss=3.452 lr=1e-04 peer_loss=3.452 avg_loss=3.452</div>
            <div className="text-foreground">[2024-01-01 12:00:05] [step 2] loss=3.312 lr=1e-04 peer_loss=3.301 avg_loss=3.307</div>
            <div className="text-foreground">[2024-01-01 12:00:06] [step 3] loss=3.178 lr=1e-04 peer_loss=3.165 avg_loss=3.171</div>
            <div className="text-foreground">[2024-01-01 12:00:07] [step 4] loss=3.089 lr=1e-04 peer_loss=3.092 avg_loss=3.086</div>
            <div className="text-muted-foreground">[2024-01-01 12:00:08] AllReduce round completed (3 peers, 1.2s)</div>
            <div className="text-foreground">[2024-01-01 12:00:09] [step 5] loss=3.001 lr=1e-04 peer_loss=3.010 avg_loss=3.005</div>
          </pre>
        </div>
      </div>
    </div>
  );
}
