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
} from "lucide-react";

interface Node {
  id: string;
  name: string;
  host: string;
  role: "HEAD" | "WORKER";
  status: "ONLINE" | "OFFLINE" | "AUDITING" | "ERROR";
  gpuName?: string;
  gpuCount?: number;
  vramGb?: number;
  cpuCores?: number;
  cpuRamGb?: number;
  nfsMounted: boolean;
}

const sampleNodes: Node[] = [];

export default function DeepSpeedNodesPage() {
  const [nodes, setNodes] = useState<Node[]>(sampleNodes);
  const [showAdd, setShowAdd] = useState(false);
  const [newNode, setNewNode] = useState({
    name: "",
    host: "",
    port: "22",
    username: "",
    role: "WORKER" as "HEAD" | "WORKER",
  });
  const [adding, setAdding] = useState(false);

  const addNode = async () => {
    setAdding(true);
    setTimeout(() => {
      const node: Node = {
        id: Date.now().toString(),
        name: newNode.name,
        host: newNode.host,
        role: newNode.role,
        status: "AUDITING",
        nfsMounted: false,
      };
      setNodes((prev) => [...prev, node]);
      setShowAdd(false);
      setNewNode({ name: "", host: "", port: "22", username: "", role: "WORKER" });
      setAdding(false);

      setTimeout(() => {
        setNodes((prev) =>
          prev.map((n) =>
            n.id === node.id
              ? {
                  ...n,
                  status: "ONLINE",
                  gpuName: "NVIDIA A100 80GB",
                  gpuCount: 4,
                  vramGb: 320,
                  cpuCores: 64,
                  cpuRamGb: 512,
                  nfsMounted: true,
                }
              : n
          )
        );
      }, 3000);
    }, 1500);
  };

  const removeNode = (id: string) => {
    setNodes((prev) => prev.filter((n) => n.id !== id));
  };

  const statusIcon = (status: Node["status"]) => {
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

  const statusLabel = (status: Node["status"]) => {
    switch (status) {
      case "ONLINE": return "Online";
      case "OFFLINE": return "Offline";
      case "AUDITING": return "Auditing...";
      case "ERROR": return "Error";
    }
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Nodes</h1>
          <p className="text-muted-foreground text-sm mt-1">
            Manage training nodes for DeepSpeed
          </p>
        </div>
        <button
          onClick={() => setShowAdd(true)}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-neon-blue to-blue-500 text-white font-semibold text-sm hover:opacity-90 transition-all"
        >
          <Plus className="w-4 h-4" />
          Add Node
        </button>
      </div>

      {showAdd && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass rounded-xl p-5 border border-border/50"
        >
          <h3 className="font-semibold mb-4">Add Remote Node</h3>
          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-xs font-medium mb-1.5">Name</label>
              <input
                type="text"
                value={newNode.name}
                onChange={(e) => setNewNode({ ...newNode, name: e.target.value })}
                placeholder="e.g. ds-worker-1"
                className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-neon-blue focus:ring-1 focus:ring-neon-blue outline-none text-sm"
              />
            </div>
            <div>
              <label className="block text-xs font-medium mb-1.5">Host / IP</label>
              <input
                type="text"
                value={newNode.host}
                onChange={(e) => setNewNode({ ...newNode, host: e.target.value })}
                placeholder="e.g. 192.168.1.100"
                className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-neon-blue focus:ring-1 focus:ring-neon-blue outline-none text-sm"
              />
            </div>
            <div>
              <label className="block text-xs font-medium mb-1.5">SSH Port</label>
              <input
                type="text"
                value={newNode.port}
                onChange={(e) => setNewNode({ ...newNode, port: e.target.value })}
                className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-neon-blue focus:ring-1 focus:ring-neon-blue outline-none text-sm"
              />
            </div>
            <div>
              <label className="block text-xs font-medium mb-1.5">Username</label>
              <input
                type="text"
                value={newNode.username}
                onChange={(e) => setNewNode({ ...newNode, username: e.target.value })}
                placeholder="e.g. ubuntu"
                className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-neon-blue focus:ring-1 focus:ring-neon-blue outline-none text-sm"
              />
            </div>
            <div>
              <label className="block text-xs font-medium mb-1.5">Role</label>
              <select
                value={newNode.role}
                onChange={(e) => setNewNode({ ...newNode, role: e.target.value as "HEAD" | "WORKER" })}
                className="w-full px-3 py-2 rounded-lg bg-background border border-border focus:border-neon-blue focus:ring-1 focus:ring-neon-blue outline-none text-sm"
              >
                <option value="HEAD">Head Node</option>
                <option value="WORKER">Worker Node</option>
              </select>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={addNode}
              disabled={adding || !newNode.name || !newNode.host}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-neon-blue to-blue-500 text-white font-semibold text-sm hover:opacity-90 transition-all disabled:opacity-50"
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

      {nodes.length === 0 ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass rounded-xl p-10 border border-border/50 text-center"
        >
          <div className="w-16 h-16 rounded-full bg-accent/30 flex items-center justify-center mx-auto mb-4">
            <Server className="w-6 h-6 text-muted-foreground" />
          </div>
          <h3 className="font-semibold mb-2">No Nodes Added</h3>
          <p className="text-sm text-muted-foreground max-w-md mx-auto">
            Add worker nodes to enable distributed DeepSpeed training with ZeRO optimization.
          </p>
          <button
            onClick={() => setShowAdd(true)}
            className="mt-4 inline-flex items-center gap-2 px-4 py-2 rounded-lg glass text-sm font-medium hover:bg-accent/50 transition-all"
          >
            <Plus className="w-4 h-4" />
            Add Your First Node
          </button>
        </motion.div>
      ) : (
        <div className="space-y-3">
          {nodes.map((node, i) => (
            <motion.div
              key={node.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              className="glass rounded-xl p-5 border border-border/50"
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3">
                  <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-neon-blue to-blue-500 flex items-center justify-center shrink-0">
                    <Monitor className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="font-semibold">{node.name}</h3>
                      <span className="text-xs px-2 py-0.5 rounded-full bg-accent/30 text-muted-foreground font-mono">
                        {node.role}
                      </span>
                    </div>
                    <p className="text-sm text-muted-foreground font-mono mt-0.5">
                      {node.host}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-1.5 text-xs">
                    {statusIcon(node.status)}
                    <span className="text-muted-foreground">{statusLabel(node.status)}</span>
                  </div>
                  <button
                    onClick={() => removeNode(node.id)}
                    className="text-muted-foreground hover:text-red-400 transition-all"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {node.gpuName && (
                <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Cpu className="w-3.5 h-3.5" />
                    {node.gpuCount}x {node.gpuName}
                  </div>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Activity className="w-3.5 h-3.5" />
                    {node.vramGb}GB VRAM
                  </div>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <HardDrive className="w-3.5 h-3.5" />
                    {node.cpuRamGb}GB RAM
                  </div>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Globe className={`w-3.5 h-3.5 ${node.nfsMounted ? "text-green-400" : "text-red-400"}`} />
                    NFS {node.nfsMounted ? "Mounted" : "Unmounted"}
                  </div>
                </div>
              )}
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
