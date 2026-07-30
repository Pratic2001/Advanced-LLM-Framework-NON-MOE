/**
 * Job Manager
 * Singleton that manages the lifecycle of training jobs
 *
 * Lifecycle: PENDING → QUEUED → SETUP → RUNNING → COMPLETED/FAILED/CANCELLED
 */

import EventEmitter from "events";
import path from "path";
import { getProcessPool } from "./process-pool";
import prisma from "./db";
import { sendJobFailureAlert } from "./mail";
import type { ProcessPool } from "./process-pool";

export interface JobEvent {
  jobId: string;
  type: "log" | "metric" | "status" | "error";
  data: any;
  timestamp: Date;
}

export class JobManager extends EventEmitter {
  private static instance: JobManager;
  private pool: ProcessPool;
  private repoRoot: string;

  private constructor() {
    super();
    this.pool = getProcessPool();
    this.repoRoot = process.env.REPO_ROOT || path.resolve(process.cwd(), "..");
    if (!process.env.REPO_ROOT) {
      console.warn(
        "[JobManager] REPO_ROOT not set; falling back to cwd parent " +
          `(${this.repoRoot}). The subprocess may not find training scripts.`
      );
    }
    this.setMaxListeners(100);
  }

  static getInstance(): JobManager {
    if (!JobManager.instance) {
      JobManager.instance = new JobManager();
    }
    return JobManager.instance;
  }

  /**
   * Launch a job by spawning the CLI command as a subprocess
   */
  async launch(
    jobId: string,
    command: string,
    args: string[]
  ): Promise<boolean> {
    if (!this.pool.canSpawn()) {
      this.emit("job:error", {
        jobId,
        type: "error",
        data: { message: "Process pool is full" },
        timestamp: new Date(),
      });
      return false;
    }

    try {
      // Update job status to RUNNING
      await prisma.job.update({
        where: { id: jobId },
        data: { status: "RUNNING" },
      });

      this.emit("job:status", {
        jobId,
        type: "status",
        data: { status: "RUNNING" },
        timestamp: new Date(),
      });

      // Parse the full command string into executable and args
      const parts = command.split(/\s+/);
      const executable = parts[0];
      const commandArgs = [...args, ...parts.slice(1)];

      const proc = this.pool.spawn(jobId, executable, commandArgs);

      // Collect logs
      let logBuffer: string[] = [];
      const MAX_LOG_LINES = 10000;

      const appendLog = (text: string) => {
        logBuffer.push(text);
        if (logBuffer.length > MAX_LOG_LINES) {
          logBuffer = logBuffer.slice(-MAX_LOG_LINES);
        }
        this.emit("job:log", {
          jobId,
          type: "log",
          data: { text },
          timestamp: new Date(),
        });
      };

      if (proc.process.stdout) {
        proc.process.stdout.on("data", (chunk: Buffer) => {
          const text = chunk.toString();
          appendLog(text);

          // Parse metrics from stdout
          const metrics = this.parseMetrics(text);
          if (metrics) {
            this.emit("job:metric", {
              jobId,
              type: "metric",
              data: metrics,
              timestamp: new Date(),
            });
          }
        });
      }

      if (proc.process.stderr) {
        proc.process.stderr.on("data", (chunk: Buffer) => {
          appendLog(`[stderr] ${chunk.toString()}`);
        });
      }

      // Handle process exit
      proc.process.on("exit", async (code, signal) => {
        const logTail = logBuffer.slice(-100).join("");

        const status = code === 0 ? "COMPLETED" : "FAILED";
        await prisma.job.update({
          where: { id: jobId },
          data: {
            status,
            exitCode: code ?? undefined,
            logTail,
          },
        });

        // Save final metrics if any
        if (proc.process.pid) {
          try {
            // Store PID reference
            await prisma.job.update({
              where: { id: jobId },
              data: { pid: proc.process.pid },
            });
          } catch {}
        }

        // Send email alert on failure
        if (status === "FAILED") {
          try {
            const job = await prisma.job.findUnique({
              where: { id: jobId },
              include: { user: { select: { email: true } } },
            });
            if (job) {
              // Don't await — fire and forget to avoid blocking
              sendJobFailureAlert({
                jobId: job.id,
                jobType: job.type,
                backend: job.backend,
                status: job.status,
                exitCode: job.exitCode,
                errorMessage: job.errorMessage,
                logTail: job.logTail || "",
                createdAt: job.createdAt,
                userEmail: job.user.email,
              });
            }
          } catch (err) {
            console.error("[JobManager] Failed to send failure alert:", err);
          }
        }

        this.emit("job:status", {
          jobId,
          type: "status",
          data: { status, exitCode: code, signal },
          timestamp: new Date(),
        });
      });

      return true;
    } catch (error: any) {
      await prisma.job.update({
        where: { id: jobId },
        data: {
          status: "FAILED",
          logTail: `Error launching job: ${error.message}`,
          errorMessage: error.message,
        },
      });

      this.emit("job:error", {
        jobId,
        type: "error",
        data: { message: error.message },
        timestamp: new Date(),
      });

      // Send alert for launch failure
      try {
        const job = await prisma.job.findUnique({
          where: { id: jobId },
          include: { user: { select: { email: true } } },
        });
        if (job) {
          sendJobFailureAlert({
            jobId: job.id,
            jobType: job.type,
            backend: job.backend,
            status: "FAILED",
            exitCode: null,
            errorMessage: error.message,
            logTail: "",
            createdAt: job.createdAt,
            userEmail: job.user.email,
          });
        }
      } catch {}

      return false;
    }
  }

  /**
   * Stop a running job
   */
  async stop(jobId: string): Promise<boolean> {
    const result = this.pool.stop(jobId);
    if (result) {
      await prisma.job.update({
        where: { id: jobId },
        data: { status: "CANCELLED" },
      });
    }
    return result;
  }

  /**
   * Get the status of a job
   */
  getStatus(jobId: string): string {
    const proc = this.pool.get(jobId);
    if (!proc) return "unknown";
    return proc.status;
  }

  /**
   * Parse training metrics from stdout lines
   */
  private parseMetrics(text: string): Record<string, number> | null {
    const metrics: Record<string, number> = {};

    // Match patterns like "loss=2.345", "lr=1e-4", "tokens/s=1423"
    const pattern = /(\w[\w/]+)=([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)/g;
    let match;

    while ((match = pattern.exec(text)) !== null) {
      metrics[match[1]] = parseFloat(match[2]);
    }

    return Object.keys(metrics).length > 0 ? metrics : null;
  }
}
