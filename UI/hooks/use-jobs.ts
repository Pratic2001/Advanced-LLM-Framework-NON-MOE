"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import type { BackendType, ScriptName } from "@/lib/schema";

/**
 * Hook for managing jobs list
 */
export function useJobs() {
  const [jobs, setJobs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchJobs = useCallback(async () => {
    try {
      setLoading(true);
      const res = await fetch("/api/jobs");
      if (!res.ok) throw new Error("Failed to fetch jobs");
      const data = await res.json();
      setJobs(data);
      setError(null);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  return { jobs, loading, error, refetch: fetchJobs };
}

/**
 * Hook for managing a single job
 */
export function useJob(jobId: string) {
  const [job, setJob] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchJob = useCallback(async () => {
    if (!jobId) return;
    try {
      setLoading(true);
      const res = await fetch(`/api/jobs/${jobId}`);
      if (!res.ok) throw new Error("Job not found");
      const data = await res.json();
      setJob(data);
      setError(null);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  useEffect(() => {
    fetchJob();
  }, [fetchJob]);

  return { job, loading, error, refetch: fetchJob };
}

/**
 * Hook for job metrics
 */
export function useJobMetrics(jobId: string, limit = 500) {
  const [metrics, setMetrics] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchMetrics = useCallback(async () => {
    if (!jobId) return;
    try {
      const res = await fetch(`/api/jobs/${jobId}/metrics?limit=${limit}`);
      if (!res.ok) throw new Error("Failed to fetch metrics");
      const data = await res.json();
      setMetrics(data);
    } catch {
      // Silently fail
    } finally {
      setLoading(false);
    }
  }, [jobId, limit]);

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, [fetchMetrics]);

  return { metrics, loading };
}

/**
 * Hook for launching a job
 */
export function useLaunchJob() {
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const launch = async (params: {
    type: string;
    backend: BackendType;
    config: Record<string, any>;
    nodeIds?: string[];
  }) => {
    setLaunching(true);
    setError(null);
    try {
      const res = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to launch job");
      return data;
    } catch (err: any) {
      setError(err.message);
      return null;
    } finally {
      setLaunching(false);
    }
  };

  return { launch, launching, error };
}

/**
 * Hook for stopping a job
 */
export function useStopJob() {
  const [stopping, setStopping] = useState(false);

  const stop = async (jobId: string) => {
    setStopping(true);
    try {
      const res = await fetch(`/api/jobs/${jobId}/stop`, { method: "POST" });
      return res.ok;
    } catch {
      return false;
    } finally {
      setStopping(false);
    }
  };

  return { stop, stopping };
}
