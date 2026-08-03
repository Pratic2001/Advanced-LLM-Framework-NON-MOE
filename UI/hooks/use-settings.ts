"use client";

import { useState, useEffect, useCallback } from "react";

/**
 * Hook for reading the current user settings (python interpreter path).
 */
export function useSettings() {
  const [pythonBin, setPythonBin] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/settings")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch settings");
        return res.json();
      })
      .then((data) => {
        if (!cancelled) setPythonBin(data.pythonBin ?? "");
      })
      .catch(() => {
        // Silently fail — the form falls back to the system default.
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const refetch = useCallback(async () => {
    try {
      const res = await fetch("/api/settings");
      if (!res.ok) throw new Error("Failed to fetch settings");
      const data = await res.json();
      setPythonBin(data.pythonBin ?? "");
    } catch {
      // Silently fail — the form falls back to the system default.
    } finally {
      setLoading(false);
    }
  }, []);

  return { pythonBin, loading, refetch };
}

/**
 * Hook for saving the python interpreter path.
 */
export function useSaveSettings() {
  const [saving, setSaving] = useState(false);

  const save = async (pythonBin: string) => {
    setSaving(true);
    try {
      const res = await fetch("/api/settings", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pythonBin }),
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
