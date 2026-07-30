/**
 * SSH Manager
 * Manages SSH connections to remote nodes for command execution and file transfer
 */

import { NodeSSH } from "node-ssh";
import prisma from "./db";

interface SSHConfig {
  host: string;
  port: number;
  username: string;
  privateKey: string;
}

interface AuditResult {
  gpuName: string;
  gpuCount: number;
  vramGb: number;
  cpuCores: number;
  cpuRamGb: number;
  pythonVersion: string;
  cudaVersion: string;
  diskFreeGb: number;
}

export class SSHManager {
  private connections: Map<string, NodeSSH> = new Map();
  private maxConnections: number;

  constructor(maxConnections = 10) {
    this.maxConnections = maxConnections;
  }

  /**
   * Get or create an SSH connection to a node
   */
  async connect(nodeId: string, config: SSHConfig): Promise<NodeSSH> {
    const existing = this.connections.get(nodeId);
    if (existing && existing.isConnected()) {
      return existing;
    }

    // Clean up stale connection
    if (existing) {
      existing.dispose();
      this.connections.delete(nodeId);
    }

    if (this.connections.size >= this.maxConnections) {
      throw new Error(`Max SSH connections (${this.maxConnections}) reached`);
    }

    const ssh = new NodeSSH();
    await ssh.connect({
      host: config.host,
      port: config.port,
      username: config.username,
      privateKey: config.privateKey,
      readyTimeout: 10000,
      keepaliveInterval: 10000,
    });

    this.connections.set(nodeId, ssh);
    return ssh;
  }

  /**
   * Disconnect a node
   */
  disconnect(nodeId: string): void {
    const conn = this.connections.get(nodeId);
    if (conn) {
      conn.dispose();
      this.connections.delete(nodeId);
    }
  }

  /**
   * Execute a command on a remote node
   */
  async exec(
    nodeId: string,
    command: string,
    options?: { timeout?: number }
  ): Promise<{ stdout: string; stderr: string; code: number | null }> {
    const ssh = this.connections.get(nodeId);
    if (!ssh || !ssh.isConnected()) {
      throw new Error(`No SSH connection to node ${nodeId}`);
    }

    const result = await ssh.execCommand(command, {
      timeout: options?.timeout || 30000,
    });

    return {
      stdout: result.stdout,
      stderr: result.stderr,
      code: result.code,
    };
  }

  /**
   * Run hardware audit on a remote node
   */
  async audit(nodeId: string, config: SSHConfig): Promise<AuditResult> {
    const ssh = await this.connect(nodeId, config);

    // Run various commands to gather hardware info
    const [gpuInfo, cpuInfo, memInfo, pythonInfo, diskInfo] = await Promise.all([
      ssh.execCommand(
        `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no-gpu"`
      ),
      ssh.execCommand(`nproc`),
      ssh.execCommand(`free -g | awk '/Mem:/ {print $2}'`),
      ssh.execCommand(`python3 --version 2>/dev/null || echo "no-python"`),
      ssh.execCommand(`df -BG /mnt/training 2>/dev/null | tail -1 | awk '{print $4}' || echo "0G"`),
    ]);

    const gpuLines = gpuInfo.stdout.trim().split("\n");
    const firstGPU = gpuLines[0];

    let gpuName = "Unknown";
    let gpuCount = 0;
    let vramGb = 0;

    if (firstGPU && firstGPU !== "no-gpu") {
      const parts = firstGPU.split(",");
      gpuName = parts[0]?.trim() || "Unknown";
      vramGb = parseInt(parts[1]?.trim().replace(" MiB", "")) || 0;
      vramGb = Math.round(vramGb / 1024); // Convert MiB to GB
      gpuCount = gpuLines.length;
      // Filter out duplicate lines
      const uniqueGpus = new Set(gpuLines.map((l) => l.trim()));
      gpuCount = uniqueGpus.size;
    }

    const cpuCores = parseInt(cpuInfo.stdout.trim()) || 0;
    const cpuRamGb = parseInt(memInfo.stdout.trim()) || 0;

    return {
      gpuName,
      gpuCount,
      vramGb,
      cpuCores,
      cpuRamGb,
      pythonVersion: pythonInfo.stdout.trim(),
      cudaVersion: "12.4", // Would parse from nvcc --version
      diskFreeGb: parseInt(diskInfo.stdout.trim().replace("G", "")) || 0,
    };
  }

  /**
   * Mount NFS on a remote node
   */
  async mountNFS(
    nodeId: string,
    config: SSHConfig,
    nfsServer: string,
    exportPath: string,
    mountPoint: string
  ): Promise<void> {
    const ssh = await this.connect(nodeId, config);

    // Create mount point
    await ssh.execCommand(`sudo mkdir -p ${mountPoint}`);

    // Mount NFS
    const result = await ssh.execCommand(
      `sudo mount -t nfs ${nfsServer}:${exportPath} ${mountPoint}`
    );

    if (result.code !== 0) {
      throw new Error(`NFS mount failed: ${result.stderr}`);
    }

    // Update DB
    await prisma.node.update({
      where: { id: nodeId },
      data: {
        nfsMounted: true,
        nfsMountPath: mountPoint,
      },
    });
  }

  /**
   * Unmount NFS on a remote node
   */
  async unmountNFS(nodeId: string, config: SSHConfig): Promise<void> {
    const ssh = await this.connect(nodeId, config);

    const node = await prisma.node.findUnique({ where: { id: nodeId } });
    if (!node?.nfsMountPath) return;

    await ssh.execCommand(`sudo umount ${node.nfsMountPath}`);

    await prisma.node.update({
      where: { id: nodeId },
      data: {
        nfsMounted: false,
        nfsMountPath: null,
      },
    });
  }

  /**
   * Get number of active connections
   */
  get connectionCount(): number {
    return this.connections.size;
  }

  /**
   * Disconnect all
   */
  disconnectAll(): void {
    for (const [id] of this.connections) {
      this.disconnect(id);
    }
  }
}

// Singleton
let sshManagerInstance: SSHManager | null = null;

export function getSSHManager(): SSHManager {
  if (!sshManagerInstance) {
    sshManagerInstance = new SSHManager();
  }
  return sshManagerInstance;
}
