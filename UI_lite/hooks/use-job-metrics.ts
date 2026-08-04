"use client";

import { useState, useEffect, useRef } from "react";
import { api } from "@/lib/base-path";

/**
 * Hook for job chart data - accumulates metrics over time
 */
export function useMetricsStream(jobId: string) {
  const [lossData, setLossData] = useState<{ step: number; loss: number }[]>([]);
  const [vramData, setVramData] = useState<{ step: number; vram: number }[]>([]);
  const [throughputData, setThroughputData] = useState<
    { step: number; tokens: number }[]
  >([]);
  const [lrData, setLrData] = useState<{ step: number; lr: number }[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  // Use refs to avoid stale closures in the polling interval
  const lossDataRef = useRef(lossData);
  lossDataRef.current = lossData;

  useEffect(() => {
    if (!jobId) return;

    // Initial fetch of metrics
    fetch(api(`/api/jobs/${jobId}/metrics`))
      .then((r) => r.json())
      .then((metrics) => {
        const loss: { step: number; loss: number }[] = [];
        const vram: { step: number; vram: number }[] = [];
        const tokens: { step: number; tokens: number }[] = [];
        const lr: { step: number; lr: number }[] = [];

        for (const m of metrics) {
          if (m.loss != null) loss.push({ step: m.step, loss: m.loss });
          if (m.vramUsed != null) vram.push({ step: m.step, vram: m.vramUsed });
          if (m.tokensPerSec != null)
            tokens.push({ step: m.step, tokens: m.tokensPerSec });
          if (m.lr != null) lr.push({ step: m.step, lr: m.lr });
        }

        setLossData(loss);
        setVramData(vram);
        setThroughputData(tokens);
        setLrData(lr);
      })
      .catch(() => {});

    // Poll for new metrics
    const interval = setInterval(async () => {
      try {
        const currentData = lossDataRef.current;
        const lastStep = currentData[currentData.length - 1]?.step ?? 0;
        const res = await fetch(
          api(`/api/jobs/${jobId}/metrics?after=${lastStep}&limit=500`)
        );
        const newMetrics = await res.json();

        for (const m of newMetrics) {
          if (m.loss != null)
            setLossData((prev) => [...prev, { step: m.step, loss: m.loss }]);
          if (m.vramUsed != null)
            setVramData((prev) => [...prev, { step: m.step, vram: m.vramUsed }]);
          if (m.tokensPerSec != null)
            setThroughputData((prev) => [
              ...prev,
              { step: m.step, tokens: m.tokensPerSec },
            ]);
          if (m.lr != null)
            setLrData((prev) => [...prev, { step: m.step, lr: m.lr }]);
        }
      } catch {}
    }, 3000);

    return () => clearInterval(interval);
  }, [jobId]);

  return { lossData, vramData, throughputData, lrData };
}
