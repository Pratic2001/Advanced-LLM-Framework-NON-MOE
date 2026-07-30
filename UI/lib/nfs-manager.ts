/**
 * NFS Manager
 * Manages NFS server setup and client mounts
 */

import { exec } from "child_process";
import { promisify } from "util";

const asyncExec = promisify(exec);

export interface NFSConfig {
  exportPath: string;
  clients: string[];
  options?: string;
}

export class NFSManager {
  /**
   * Check if NFS server packages are installed
   */
  static async isServerInstalled(): Promise<boolean> {
    try {
      await asyncExec("dpkg -l nfs-kernel-server 2>/dev/null | grep -q ^ii", {
        timeout: 5000,
      });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Check if NFS common packages are installed (client)
   */
  static async isClientInstalled(): Promise<boolean> {
    try {
      await asyncExec("dpkg -l nfs-common 2>/dev/null | grep -q ^ii", {
        timeout: 5000,
      });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Create NFS export directory and set permissions
   */
  static async setupExportDir(exportPath: string): Promise<void> {
    await asyncExec(`sudo mkdir -p ${exportPath}`, { timeout: 10000 });
    await asyncExec(`sudo chown -R $(whoami):$(whoami) ${exportPath}`, {
      timeout: 10000,
    });
    await asyncExec(`sudo chmod 755 ${exportPath}`, { timeout: 5000 });
  }

  /**
   * Add an export to /etc/exports
   */
  static async addExport(config: NFSConfig): Promise<void> {
    const options = config.options || "rw,sync,no_subtree_check,no_root_squash";
    const line = `${config.exportPath} ${config.clients.join(" ")}(${options})`;

    // Check if export already exists
    const { stdout: exports } = await asyncExec(
      "sudo cat /etc/exports 2>/dev/null || echo ''",
      { timeout: 5000 }
    );

    if (!exports.includes(config.exportPath)) {
      await asyncExec(`echo "${line}" | sudo tee -a /etc/exports`, {
        timeout: 5000,
      });
    }

    // Reload exports
    await asyncExec("sudo exportfs -ra", { timeout: 10000 });
  }

  /**
   * Mount NFS on a client
   */
  static async mountClient(
    serverIp: string,
    exportPath: string,
    mountPoint: string
  ): Promise<void> {
    await asyncExec(`sudo mkdir -p ${mountPoint}`, { timeout: 5000 });
    await asyncExec(`sudo mount -t nfs ${serverIp}:${exportPath} ${mountPoint}`, {
      timeout: 15000,
    });
  }

  /**
   * Unmount NFS
   */
  static async unmount(mountPoint: string): Promise<void> {
    try {
      await asyncExec(`sudo umount ${mountPoint}`, { timeout: 10000 });
    } catch {
      // Not mounted or already unmounted
    }
  }

  /**
   * Add NFS mount to /etc/fstab for persistence across reboots
   */
  static async addToFstab(
    serverIp: string,
    exportPath: string,
    mountPoint: string
  ): Promise<void> {
    const fstabLine = `${serverIp}:${exportPath} ${mountPoint} nfs rw,defaults 0 0`;

    const { stdout: fstab } = await asyncExec("sudo cat /etc/fstab", {
      timeout: 5000,
    });
    if (!fstab.includes(mountPoint)) {
      await asyncExec(`echo "${fstabLine}" | sudo tee -a /etc/fstab`, {
        timeout: 5000,
      });
    }
  }

  /**
   * Check if NFS is mounted at a path
   */
  static async isMounted(mountPoint: string): Promise<boolean> {
    try {
      const { stdout: output } = await asyncExec("mount | grep nfs", {
        timeout: 5000,
      });
      return output.includes(mountPoint);
    } catch {
      return false;
    }
  }
}
