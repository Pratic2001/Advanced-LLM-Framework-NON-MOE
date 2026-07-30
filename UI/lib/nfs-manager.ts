/**
 * NFS Manager
 * Manages NFS server setup and client mounts
 */

import { execSync } from "child_process";

export interface NFSConfig {
  exportPath: string;
  clients: string[];
  options?: string;
}

export class NFSManager {
  /**
   * Check if NFS server packages are installed
   */
  static isServerInstalled(): boolean {
    try {
      execSync("dpkg -l nfs-kernel-server 2>/dev/null | grep -q ^ii", {
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
  static isClientInstalled(): boolean {
    try {
      execSync("dpkg -l nfs-common 2>/dev/null | grep -q ^ii", { timeout: 5000 });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Create NFS export directory and set permissions
   */
  static setupExportDir(exportPath: string): void {
    execSync(`sudo mkdir -p ${exportPath}`, { timeout: 10000 });
    execSync(`sudo chown -R $(whoami):$(whoami) ${exportPath}`, { timeout: 10000 });
    execSync(`sudo chmod 755 ${exportPath}`, { timeout: 5000 });
  }

  /**
   * Add an export to /etc/exports
   */
  static addExport(config: NFSConfig): void {
    const options = config.options || "rw,sync,no_subtree_check,no_root_squash";
    const line = `${config.exportPath} ${config.clients.join(" ")}(${options})`;

    // Check if export already exists
    const exports = execSync("sudo cat /etc/exports 2>/dev/null || echo ''")
      .toString()
      .trim();

    if (!exports.includes(config.exportPath)) {
      execSync(`echo "${line}" | sudo tee -a /etc/exports`, { timeout: 5000 });
    }

    // Reload exports
    execSync("sudo exportfs -ra", { timeout: 10000 });
  }

  /**
   * Mount NFS on a client
   */
  static mountClient(
    serverIp: string,
    exportPath: string,
    mountPoint: string
  ): void {
    execSync(`sudo mkdir -p ${mountPoint}`, { timeout: 5000 });
    execSync(`sudo mount -t nfs ${serverIp}:${exportPath} ${mountPoint}`, {
      timeout: 15000,
    });
  }

  /**
   * Unmount NFS
   */
  static unmount(mountPoint: string): void {
    try {
      execSync(`sudo umount ${mountPoint}`, { timeout: 10000 });
    } catch {
      // Not mounted or already unmounted
    }
  }

  /**
   * Add NFS mount to /etc/fstab for persistence across reboots
   */
  static addToFstab(
    serverIp: string,
    exportPath: string,
    mountPoint: string
  ): void {
    const fstabLine = `${serverIp}:${exportPath} ${mountPoint} nfs rw,defaults 0 0`;

    const fstab = execSync("sudo cat /etc/fstab").toString();
    if (!fstab.includes(mountPoint)) {
      execSync(`echo "${fstabLine}" | sudo tee -a /etc/fstab`, { timeout: 5000 });
    }
  }

  /**
   * Check if NFS is mounted at a path
   */
  static isMounted(mountPoint: string): boolean {
    try {
      const output = execSync("mount | grep nfs", { timeout: 5000 }).toString();
      return output.includes(mountPoint);
    } catch {
      return false;
    }
  }
}
