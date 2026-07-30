/**
 * Utility functions for the LLM Training UI
 */

/**
 * Format bytes to human-readable size
 */
export function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB", "PB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  const val = bytes / Math.pow(1024, i);
  return `${val.toFixed(1)} ${units[i]}`;
}

/**
 * Format a number with commas
 */
export function formatNumber(n: number): string {
  return n.toLocaleString();
}

/**
 * Format seconds to human-readable duration
 */
export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  if (seconds < 86400)
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  return `${Math.floor(seconds / 86400)}d ${Math.floor((seconds % 86400) / 3600)}h`;
}

/**
 * Format date to relative time string
 */
export function relativeTime(date: Date): string {
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const seconds = Math.floor(diff / 1000);

  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;
  return date.toLocaleDateString();
}

/**
 * Generate a unique color based on string hash (for charts, nodes, etc.)
 */
export function stringToColor(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  const colors = [
    "#00f0ff",
    "#0088ff",
    "#7c3aed",
    "#ec4899",
    "#22d3ee",
    "#34d399",
    "#f472b6",
    "#a78bfa",
    "#2dd4bf",
    "#fbbf24",
  ];
  return colors[Math.abs(hash) % colors.length];
}

/**
 * Truncate string with ellipsis
 */
export function truncate(str: string, maxLen: number): string {
  if (str.length <= maxLen) return str;
  return str.slice(0, maxLen) + "...";
}

/**
 * Deep clone a JSON-serializable object
 */
export function deepClone<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj));
}

/**
 * Parse metric line from training stdout (e.g., [step 100] loss=2.345 lr=1e-4)
 */
export function parseMetricLine(line: string): Record<string, number> | null {
  const metrics: Record<string, number> = {};

  // Match patterns like "loss=2.345", "lr=1e-4", "tokens/s=1423"
  const pattern = /(\w[\w/]+)=([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)/g;
  let match;

  while ((match = pattern.exec(line)) !== null) {
    metrics[match[1]] = parseFloat(match[2]);
  }

  return Object.keys(metrics).length > 0 ? metrics : null;
}

/**
 * Extract step number from log line
 */
export function extractStep(line: string): number | null {
  const match = line.match(/\[step\s+(\d+)\]/i);
  return match ? parseInt(match[1], 10) : null;
}

/**
 * Sleep for ms
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Merge class names (simple version — use clsx/tailwind-merge in components)
 */
export function cn(...classes: (string | boolean | undefined | null)[]): string {
  return classes.filter(Boolean).join(" ");
}
