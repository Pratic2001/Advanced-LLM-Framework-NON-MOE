/**
 * Metrics Collector
 * Parses training stdout for structured metrics and stores them in the database
 */

import prisma from "./db";
import { extractStep, parseMetricLine } from "./utils";

export interface ParsedMetrics {
  step: number;
  loss?: number;
  vramUsed?: number;
  tokensPerSec?: number;
  lr?: number;
  gradNorm?: number;
  throughput?: number;
  nodeName?: string;
}

export class MetricsCollector {
  private buffer: Map<string, ParsedMetrics[]> = new Map();
  private flushInterval: NodeJS.Timeout;
  private readonly MAX_BUFFER_SIZE = 100;

  constructor() {
    // Flush buffer every 5 seconds
    this.flushInterval = setInterval(() => this.flush(), 5000);
  }

  /**
   * Process a line of stdout from a training job
   */
  processLine(jobId: string, line: string): ParsedMetrics | null {
    const step = extractStep(line);
    const parsed = parseMetricLine(line);

    if (step === undefined || step === null || !parsed) return null;

    const metrics: ParsedMetrics = {
      step,
      loss: parsed.loss,
      vramUsed: parsed.vram,
      tokensPerSec: parsed["tokens/s"] || parsed.tokens_per_sec,
      lr: parsed.lr,
      gradNorm: parsed.grad_norm,
      throughput: parsed.throughput,
    };

    // Add to buffer
    if (!this.buffer.has(jobId)) {
      this.buffer.set(jobId, []);
    }

    const buf = this.buffer.get(jobId)!;
    buf.push(metrics);

    // Flush if buffer is large
    if (buf.length >= this.MAX_BUFFER_SIZE) {
      this.flushJob(jobId);
    }

    return metrics;
  }

  /**
   * Flush buffered metrics to database
   */
  private async flush(): Promise<void> {
    const jobIds = Array.from(this.buffer.keys());
    await Promise.all(jobIds.map((id) => this.flushJob(id)));
  }

  private async flushJob(jobId: string): Promise<void> {
    const buf = this.buffer.get(jobId);
    if (!buf || buf.length === 0) return;

    const batch = buf.splice(0, buf.length);

    try {
      await prisma.jobMetric.createMany({
        data: batch.map((m) => ({
          jobId,
          step: m.step,
          loss: m.loss,
          vramUsed: m.vramUsed,
          tokensPerSec: m.tokensPerSec,
          lr: m.lr,
          gradNorm: m.gradNorm,
          throughput: m.throughput,
          nodeName: m.nodeName,
        })),
      });
    } catch (error) {
      console.error(`[MetricsCollector] Failed to flush metrics for ${jobId}:`, error);
    }
  }

  /**
   * Get recent metrics for a job
   */
  async getMetrics(
    jobId: string,
    limit = 500,
    afterStep?: number
  ): Promise<ParsedMetrics[]> {
    const where: any = { jobId };
    if (afterStep !== undefined) {
      where.step = { gt: afterStep };
    }

    const records = await prisma.jobMetric.findMany({
      where,
      orderBy: { step: "asc" },
      take: limit,
    });

    return records.map((r) => ({
      step: r.step,
      loss: r.loss ?? undefined,
      vramUsed: r.vramUsed ?? undefined,
      tokensPerSec: r.tokensPerSec ?? undefined,
      lr: r.lr ?? undefined,
      gradNorm: r.gradNorm ?? undefined,
      throughput: r.throughput ?? undefined,
      nodeName: r.nodeName ?? undefined,
    }));
  }

  /**
   * Cleanup
   */
  destroy(): void {
    clearInterval(this.flushInterval);
    this.flush();
  }
}

// Singleton
let collectorInstance: MetricsCollector | null = null;

export function getMetricsCollector(): MetricsCollector {
  if (!collectorInstance) {
    collectorInstance = new MetricsCollector();
  }
  return collectorInstance;
}
