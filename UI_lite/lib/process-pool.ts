/**
 * Process Pool
 * Manages a pool of child processes with configurable concurrency limits
 */

import { ChildProcess, spawn, SpawnOptions } from "child_process";
import path from "path";

export interface PoolProcess {
  id: string;
  process: ChildProcess;
  command: string;
  startTime: Date;
  status: "running" | "stopping";
}

export class ProcessPool {
  private processes: Map<string, PoolProcess> = new Map();
  private maxConcurrent: number;
  private repoRoot: string;

  constructor(maxConcurrent = 4) {
    this.maxConcurrent = maxConcurrent;
    // Resolve REPO_ROOT from env or default to two levels up from UI/
    this.repoRoot = process.env.REPO_ROOT || path.resolve(process.cwd(), "..");
  }

  get runningCount(): number {
    return this.processes.size;
  }

  get availableSlots(): number {
    return this.maxConcurrent - this.processes.size;
  }

  canSpawn(): boolean {
    return this.processes.size < this.maxConcurrent;
  }

  spawn(
    id: string,
    command: string,
    args: string[],
    options?: SpawnOptions
  ): PoolProcess {
    if (!this.canSpawn()) {
      throw new Error(
        `Process pool full (${this.maxConcurrent} max). Wait for a slot.`
      );
    }

    if (this.processes.has(id)) {
      throw new Error(`Process with id ${id} already exists`);
    }

    const proc = spawn(command, args, {
      cwd: this.repoRoot,
      stdio: ["ignore", "pipe", "pipe"],
      detached: true, // Create process group for clean cleanup
      ...options,
    });

    // Log errors
    proc.on("error", (err) => {
      console.error(`[ProcessPool] Process ${id} error:`, err.message);
    });

    // Clean up on exit
    proc.on("exit", (code, signal) => {
      console.log(
        `[ProcessPool] Process ${id} exited with code=${code} signal=${signal}`
      );
      this.processes.delete(id);
    });

    const poolProc: PoolProcess = {
      id,
      process: proc,
      command: `${command} ${args.join(" ")}`,
      startTime: new Date(),
      status: "running",
    };

    this.processes.set(id, poolProc);
    return poolProc;
  }

  get(id: string): PoolProcess | undefined {
    return this.processes.get(id);
  }

  stop(id: string, signal: NodeJS.Signals = "SIGTERM"): boolean {
    const proc = this.processes.get(id);
    if (!proc) return false;

    try {
      proc.status = "stopping";
      // Kill the process group (negative pid = process group)
      if (proc.process.pid) {
        process.kill(-proc.process.pid, signal);
      }
      return true;
    } catch (err) {
      console.error(`[ProcessPool] Failed to stop ${id}:`, err);
      return false;
    }
  }

  stopAll(signal: NodeJS.Signals = "SIGTERM"): void {
    this.processes.forEach((_proc, id) => {
      this.stop(id, signal);
    });
  }

  list(): PoolProcess[] {
    return Array.from(this.processes.values());
  }

  getLogStream(id: string): { stdout: NodeJS.ReadableStream | null; stderr: NodeJS.ReadableStream | null } {
    const proc = this.processes.get(id);
    if (!proc) {
      return { stdout: null, stderr: null };
    }
    return {
      stdout: proc.process.stdout,
      stderr: proc.process.stderr,
    };
  }
}

// Singleton
let poolInstance: ProcessPool | null = null;

export function getProcessPool(): ProcessPool {
  if (!poolInstance) {
    poolInstance = new ProcessPool();
  }
  return poolInstance;
}
