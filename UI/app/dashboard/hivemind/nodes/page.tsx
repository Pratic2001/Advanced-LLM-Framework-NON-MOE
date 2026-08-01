"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  Plus,
  Server,
  Monitor,
  Cpu,
  HardDrive,
  Activity,
  Globe,
  Loader2,
  Trash2,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Radio,
} from "lucide-react";

interface Peer {
  id: string;
  name: string;
  host: string;
  role: "BOOTSTRAP" | "PEER";
  status: "ONLINE" | "OFFLINE" | "AUDITING" | "ERROR";
  gpuName?: string;
  gpuCount?: number;
  vramGb?: number;
  peerId?: string;
  nfsMounted: boolean;
}

const samplePeers: Peer[] = [];

export default function HivemindNodesPage() {
  const [peers, setPeers] = useState<Peer[]>(samplePeers);
  const [showAdd, setShowAdd] = useState(false);
  const [newPeer, setNewPeer] = useState({
    name: "",
    host: "",
    port: "22",
    username: "",
    role: "PEER" as "BOOTSTRAP" | "PEER",
  });
  const [adding, setAdding] = useState(false);

  const addPeer = async () => {
    setAdding(true);
    setTimeout(() => {
      const peer: Peer = {
        id: Date.now().toString(),
        name: newPeer.name,
        host: newPeer.host,
        role: newPeer.role,
        status: "AUDITING",
        nfsMounted: false,
      };
      setPeers((prev) => [...prev, peer]);
      setShowAdd(false);
      setNewPeer({ name: "", host: "", port: "22", username: "", role: "PEER" });
      setAdding(false);

      setTimeout(() => {
        setPeers((prev) =>
          prev.map((p) =>
            p.id === peer.id
              ? {
                  ...p,
                  status: "ONLINE",
                  gpuName: "NVIDIA RTX 4090",
                  gpuCount: 2,
                  vramGb: 48,
                  peerId: `12D3KooW${Math.random().toString(36).slice(2, 14)}`,
                  nfsMounted: true,
                }
              : p
          )
        );
      }, 3000);
    }, 1500);
  };

  const removePeer = (id: string) => {
    setPeers((prev) => prev.filter((p) => p.id !== id));
  };

  const statusIcon = (status: Peer["status"]) => {
    switch (status) {
      case "ONLINE":
        return <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />;
      case "OFFLINE":
        return <XCircle className="w-3.5 h-3.5 text-red-400" />;
      case "AUDITING":
        return <Loader2 className="w-3.5 h-3.5 text-yellow-400 animate-spin" />;
      case "ERROR":
        return <AlertTriangle className="w-3.5 h-3.5 text-red-400" />;
    }
  };

  const statusLabel = (status: Peer["status"]) => {
    switch (status) {
      case "ONLINE": return "Online";
      case "OFFLINE": return "Offline";
      case "AUDITING": return "Auditing...";
      case "ERROR": return "Error";
    }
  };

  const bootstrapPeer = peers.find((p) => p.role === "BOOTSTRAP");

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Peers</h1>
          <p className="text-muted-foreground text-sm mt-1">
            Manage Hivemind bootstrap peer and worker peers
          </p>
        </div>
        <button
          onClick={() => setShowAdd(true)}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-[hsl(var(--palette-primary))] to-[hsl(var(--palette-secondary))] text-white font-semibold text-sm hover:opacity-90 transition-all glow-primary"
        >
          <Plus className="w-4 h-4" />
          Add Peer
        </button>
      </div>

      {showAdd && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass rounded-xl p-5 border border-border/50"
        >
          <h3 className="font-semibold mb-4">Add Hivemind Peer</h3>
          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-xs font-medium mb-1.5">Name</label>
              <input
                type="text"
                value={newPeer.name}
                onChange={(e) => setNewPeer({ ...newPeer, name: e.target.value })}
                placeholder="e.g. peer-1"
                className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-[hsl(var(--palette-primary))] focus:ring-1 focus:ring-[hsl(var(--palette-primary))] outline-none text-sm"
              />
            </div>
            <div>
              <label className="block text-xs font-medium mb-1.5">Host / IP</label>
              <input
                type="text"
                value={newPeer.host}
                onChange={(e) => setNewPeer({ ...newPeer, host: e.target.value })}
                placeholder="e.g. 192.168.1.100"
                className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-[hsl(var(--palette-primary))] focus:ring-1 focus:ring-[hsl(var(--palette-primary))] outline-none text-sm"
              />
            </div>
            <div>
              <label className="block text-xs font-medium mb-1.5">SSH Port</label>
              <input
                type="text"
                value={newPeer.port}
                onChange={(e) => setNewPeer({ ...newPeer, port: e.target.value })}
                className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-[hsl(var(--palette-primary))] focus:ring-1 focus:ring-[hsl(var(--palette-primary))] outline-none text-sm"
              />
            </div>
            <div>
              <label className="block text-xs font-medium mb-1.5">Username</label>
              <input
                type="text"
                value={newPeer.username}
                onChange={(e) => setNewPeer({ ...newPeer, username: e.target.value })}
                placeholder="e.g. ubuntu"
                className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-[hsl(var(--palette-primary))] focus:ring-1 focus:ring-[hsl(var(--palette-primary))] outline-none text-sm"
              />
            </div>
            <div>
              <label className="block text-xs font-medium mb-1.5">Role</label>
              <select
                value={newPeer.role}
                onChange={(e) => setNewPeer({ ...newPeer, role: e.target.value as "BOOTSTRAP" | "PEER" })}
                className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-[hsl(var(--palette-primary))] focus:ring-1 focus:ring-[hsl(var(--palette-primary))] outline-none text-sm"
              >
                <option value="PEER">Worker Peer</option>
                <option value="BOOTSTRAP">Bootstrap Peer</option>
              </select>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={addPeer}
              disabled={adding || !newPeer.name || !newPeer.host}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-[hsl(var(--palette-primary))] to-[hsl(var(--palette-secondary))] text-white font-semibold text-sm hover:opacity-90 transition-all disabled:opacity-50 glow-primary"
            >
              {adding ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Adding...
                </>
              ) : (
                <>
                  <Plus className="w-4 h-4" />
                  Add & Audit
                </>
              )}
            </button>
            <button
              onClick={() => setShowAdd(false)}
              className="px-4 py-2 rounded-lg glass text-sm font-medium hover:bg-accent/50 transition-all"
            >
              Cancel
            </button>
          </div>
        </motion.div>
      )}

      {/* Bootstrap peer highlight */}
      {bootstrapPeer && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass rounded-xl p-5 border border-neon-purple/30"
          style={{ boxShadow: "0 0 15px rgba(124,58,237,0.1)" }}
        >
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-[hsl(var(--palette-primary))] to-[hsl(var(--palette-secondary))] flex items-center justify-center shrink-0">
                <Radio className="w-5 h-5 text-white" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="font-semibold">{bootstrapPeer.name}</h3>
                  <span className="text-xs px-2 py-0.5 rounded-full bg-neon-purple/10 text-neon-purple font-mono">
                    BOOTSTRAP
                  </span>
                </div>
                <p className="text-sm text-muted-foreground font-mono mt-0.5">
                  {bootstrapPeer.host}
                </p>
                {bootstrapPeer.peerId && (
                  <p className="text-xs text-muted-foreground font-mono mt-1">
                    Peer ID: {bootstrapPeer.peerId}
                  </p>
                )}
              </div>
            </div>
            <div className="flex items-center gap-1.5 text-xs">
              {statusIcon(bootstrapPeer.status)}
              <span className="text-muted-foreground">{statusLabel(bootstrapPeer.status)}</span>
            </div>
          </div>
          {bootstrapPeer.gpuName && (
            <div className="mt-4 grid grid-cols-2 md:grid-cols-3 gap-3">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Cpu className="w-3.5 h-3.5" />
                {bootstrapPeer.gpuCount}x {bootstrapPeer.gpuName}
              </div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Activity className="w-3.5 h-3.5" />
                {bootstrapPeer.vramGb}GB VRAM
              </div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Globe className="w-3.5 h-3.5 text-green-400" />
                NFS {bootstrapPeer.nfsMounted ? "Mounted" : "Unmounted"}
              </div>
            </div>
          )}
        </motion.div>
      )}

      {/* Worker peers */}
      {peers.filter((p) => p.role === "PEER").length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-muted-foreground mb-3">Worker Peers</h3>
          <div className="space-y-3">
            {peers
              .filter((p) => p.role === "PEER")
              .map((peer, i) => (
                <motion.div
                  key={peer.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="glass rounded-xl p-5 border border-border/50"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3">
                      <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-[hsl(var(--palette-primary))]/60 to-[hsl(var(--palette-secondary))]/60 flex items-center justify-center shrink-0">
                        <Monitor className="w-5 h-5 text-white" />
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="font-semibold">{peer.name}</h3>
                          <span className="text-xs px-2 py-0.5 rounded-full bg-accent/30 text-muted-foreground font-mono">
                            PEER
                          </span>
                        </div>
                        <p className="text-sm text-muted-foreground font-mono mt-0.5">
                          {peer.host}
                        </p>
                        {peer.peerId && (
                          <p className="text-xs text-muted-foreground font-mono mt-1 truncate max-w-[300px]">
                            Peer ID: {peer.peerId}
                          </p>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-1.5 text-xs">
                        {statusIcon(peer.status)}
                        <span className="text-muted-foreground">{statusLabel(peer.status)}</span>
                      </div>
                      <button
                        onClick={() => removePeer(peer.id)}
                        className="text-muted-foreground hover:text-red-400 transition-all"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>

                  {peer.gpuName && (
                    <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Cpu className="w-3.5 h-3.5" />
                        {peer.gpuCount}x {peer.gpuName}
                      </div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Activity className="w-3.5 h-3.5" />
                        {peer.vramGb}GB VRAM
                      </div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <HardDrive className="w-3.5 h-3.5" />
                        Connected
                      </div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Globe className={`w-3.5 h-3.5 ${peer.nfsMounted ? "text-green-400" : "text-red-400"}`} />
                        NFS {peer.nfsMounted ? "Mounted" : "Unmounted"}
                      </div>
                    </div>
                  )}
                </motion.div>
              ))}
          </div>
        </div>
      )}

      {/* Empty state */}
      {peers.length === 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass rounded-xl p-10 border border-border/50 text-center"
        >
          <div className="w-16 h-16 rounded-full bg-accent/30 flex items-center justify-center mx-auto mb-4">
            <Globe className="w-6 h-6 text-muted-foreground" />
          </div>
          <h3 className="font-semibold mb-2">No Peers Added</h3>
          <p className="text-sm text-muted-foreground max-w-md mx-auto">
            Add at least one bootstrap peer and one or more worker peers to start decentralized
            training. Hivemind peers connect via DHT.
          </p>
          <button
            onClick={() => setShowAdd(true)}
            className="mt-4 inline-flex items-center gap-2 px-4 py-2 rounded-lg glass text-sm font-medium hover:bg-accent/50 transition-all"
          >
            <Plus className="w-4 h-4" />
            Add Bootstrap Peer
          </button>
        </motion.div>
      )}
    </div>
  );
}
