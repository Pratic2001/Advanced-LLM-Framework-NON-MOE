/**
 * GPU Monitor
 * Periodically polls local nvidia-smi for GPU metrics
 */

import { execSync } from "child_process";

export interface GPUStatus {
  index: number;
  name: string;
  temperature: number;
  utilization: number;
  memoryUsed: number;
  memoryTotal: number;
  powerDraw: number;
  powerLimit: number;
}

export class GPUMonitor {
  private interval: NodeJS.Timeout | null = null;
  private pollingIntervalMs: number;
  private listeners: Array<(status: GPUStatus[]) => void> = [];

  constructor(pollingIntervalMs = 5000) {
    this.pollingIntervalMs = pollingIntervalMs;
  }

  /**
   * Start monitoring
   */
  start(): void {
    if (this.interval) return;

    this.interval = setInterval(() => {
      this.poll();
    }, this.pollingIntervalMs);

    // Immediate first poll
    this.poll();
  }

  /**
   * Stop monitoring
   */
  stop(): void {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
  }

  /**
   * Add a listener for GPU status updates
   */
  onStatus(callback: (status: GPUStatus[]) => void): () => void {
    this.listeners.push(callback);
    return () => {
      this.listeners = this.listeners.filter((l) => l !== callback);
    };
  }

  /**
   * Poll GPU status
   */
  private poll(): void {
    try {
      const status = this.getGPUStatus();
      for (const listener of this.listeners) {
        listener(status);
      }
    } catch (error) {
      // nvidia-smi not available or error
    }
  }

  /**
   * Get current GPU status from nvidia-smi
   */
  getGPUStatus(): GPUStatus[] {
    try {
      const output = execSync(
        `nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null`,
        { timeout: 5000 }
      )
        .toString()
        .trim();

      if (!output) return [];

      const lines = output.split("\n");
      return lines
        .filter((l) => l.trim())
        .map((line) => {
          const parts = line.split(",").map((p) => p.trim());
          return {
            index: parseInt(parts[0]) || 0,
            name: parts[1] || "Unknown",
            temperature: parseInt(parts[2]) || 0,
            utilization: parseInt(parts[3]) || 0,
            memoryUsed: parseFloat(parts[4]) || 0,
            memoryTotal: parseFloat(parts[5]) || 0,
            powerDraw: parseFloat(parts[6]) || 0,
            powerLimit: parseFloat(parts[7]) || 0,
          };
        });
    } catch {
      return [];
    }
  }
}

// Singleton
let monitorInstance: GPUMonitor | null = null;

export function getGPUMonitor(): GPUMonitor {
  if (!monitorInstance) {
    monitorInstance = new GPUMonitor();
  }
  return monitorInstance;
}
