"use client";

import { useState, useEffect, useCallback } from "react";
import { api } from "@/lib/base-path";

/**
 * Hook for managing nodes
 */
export function useNodes() {
  const [nodes, setNodes] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchNodes = useCallback(async () => {
    try {
      setLoading(true);
      const res = await fetch(api("/api/nodes"));
      if (!res.ok) throw new Error("Failed to fetch nodes");
      const data = await res.json();
      setNodes(data);
      setError(null);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchNodes();
  }, [fetchNodes]);

  return { nodes, loading, error, refetch: fetchNodes };
}

/**
 * Hook for adding a node
 */
export function useAddNode() {
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const addNode = async (params: {
    name: string;
    host: string;
    port?: number;
    username: string;
    role?: string;
  }) => {
    setAdding(true);
    setError(null);
    try {
      const res = await fetch(api("/api/nodes"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to add node");
      return data;
    } catch (err: any) {
      setError(err.message);
      return null;
    } finally {
      setAdding(false);
    }
  };

  return { addNode, adding, error };
}

/**
 * Hook for auditing a node
 */
export function useAuditNode() {
  const [auditing, setAuditing] = useState(false);

  const audit = async (nodeId: string) => {
    setAuditing(true);
    try {
      const res = await fetch(api(`/api/nodes/${nodeId}/audit`), { method: "POST" });
      return res.ok;
    } catch {
      return false;
    } finally {
      setAuditing(false);
    }
  };

  return { audit, auditing };
}

/**
 * Hook for mounting NFS on a node
 */
export function useMountNFS() {
  const [mounting, setMounting] = useState(false);

  const mount = async (nodeId: string, nfsServer: string, exportPath: string) => {
    setMounting(true);
    try {
      const res = await fetch(api(`/api/nodes/${nodeId}/mount`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nfsServer, exportPath, mountPoint: "/mnt/training" }),
      });
      return res.ok;
    } catch {
      return false;
    } finally {
      setMounting(false);
    }
  };

  return { mount, mounting };
}
