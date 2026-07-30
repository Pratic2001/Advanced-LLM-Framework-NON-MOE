"use client";

import { useState, useEffect, useCallback } from "react";
import type { BackendType } from "@/lib/schema";

/**
 * Hook for config presets
 */
export function useConfigPresets() {
  const [presets, setPresets] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchPresets = useCallback(async () => {
    try {
      setLoading(true);
      const res = await fetch("/api/configs");
      if (!res.ok) throw new Error("Failed to fetch presets");
      const data = await res.json();
      setPresets(data);
    } catch {
      // Silently fail
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPresets();
  }, [fetchPresets]);

  return { presets, loading, refetch: fetchPresets };
}

/**
 * Hook for saving a config preset
 */
export function useSaveConfig() {
  const [saving, setSaving] = useState(false);

  const save = async (params: {
    name: string;
    description?: string;
    backend: BackendType;
    config: Record<string, any>;
  }) => {
    setSaving(true);
    try {
      const res = await fetch("/api/configs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });
      return res.ok;
    } catch {
      return false;
    } finally {
      setSaving(false);
    }
  };

  return { save, saving };
}

/**
 * Hook for loading a specific config preset
 */
export function useConfigPreset(configId: string) {
  const [preset, setPreset] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!configId) {
      setLoading(false);
      return;
    }
    fetch(`/api/configs/${configId}`)
      .then((r) => r.json())
      .then(setPreset)
      .catch(() => setPreset(null))
      .finally(() => setLoading(false));
  }, [configId]);

  return { preset, loading };
}
