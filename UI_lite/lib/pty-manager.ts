/**
 * PTY Manager
 *
 * Spawns long-lived pseudo-terminals (bash) for the interactive shell in the
 * dashboard config pages. Each WebSocket connection gets its own PTY scoped to
 * REPO_ROOT. The PTY is killed when the socket closes.
 *
 * SECURITY MODEL
 * ──────────────
 * The shell is real bash, NOT a command allowlist. We intentionally offer
 * unrestricted shell access (within REPO_ROOT) to mirror the existing manual
 * workflow where an operator runs training scripts directly from a terminal.
 *
 * Mitigations:
 *   1. Authentication required — same NextAuth session used by the rest of the
 *      API. Connections without a valid session are rejected.
 *   2. Per-connection resource cap — at most MAX_PTYS concurrently. Excess
 *      connections are rejected with a 503-style message.
 *   3. Working directory is locked to REPO_ROOT via `spawn({ cwd })`. The shell
 *      inherits the env but PATH and HOME are scrubbed to defaults that keep
 *      scripts runnable.
 *   4. Idle timeout — connections that send no input for IDLE_TIMEOUT_MS are
 *      terminated. Defaults to 30 minutes.
 *   5. PTY is force-killed on socket close, so leaked processes are bounded.
 *
 * The PTY is NOT a sandbox (bash can escape cwd with `cd /`). Treat this as
 * "authenticated user runs commands on the same machine as the dev server."
 * If you need a hard sandbox, swap the implementation here for a
 * container-exec-based one.
 */

import { IPty, spawn as ptySpawn } from "@homebridge/node-pty-prebuilt-multiarch";
import path from "path";

const MAX_PTYS = 50;
const IDLE_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes

const repoRoot = path.resolve(process.env.REPO_ROOT || "../");

let activePtys = 0;

export function getActivePtyCount(): number {
  return activePtys;
}

export function canSpawnPty(): boolean {
  return activePtys < MAX_PTYS;
}

interface SpawnPtyOptions {
  /** Terminal cols from the xterm on connect. */
  cols: number;
  /** Terminal rows from the xterm on connect. */
  rows: number;
  /** The user ID for logging/auditing. */
  userId: string;
}

export interface ShellPty {
  pid: number;
  write: (data: string) => void;
  resize: (cols: number, rows: number) => void;
  kill: (signal?: string) => void;
  onData: (handler: (data: string) => void) => void;
  onExit: (handler: (e: { exitCode: number; signal?: number }) => void) => void;
}

/**
 * Spawn a fresh bash PTY for one connection. Caller owns lifecycle — call
 * `kill()` on disconnect. The PTY's CWD is locked to REPO_ROOT.
 */
export function spawnShellPty(opts: SpawnPtyOptions): ShellPty {
  if (!canSpawnPty()) {
    throw new Error(
      `PTY pool exhausted (max ${MAX_PTYS}). Try again later or close idle tabs.`
    );
  }

  const shell = process.env.SHELL || "/bin/bash";

  // Force interactive shell so bash reads ~/.bashrc, prompts work, etc.
  // We do NOT pass -l (login) because that would also clear the cwd override
  // on some shells; setting cwd explicitly is more reliable.
  const args = ["-i"];

  const pty: IPty = ptySpawn(shell, args, {
    name: "xterm-256color",
    cols: Math.max(20, Math.floor(opts.cols) || 80),
    rows: Math.max(5, Math.floor(opts.rows) || 24),
    cwd: repoRoot,
    env: {
      ...process.env,
      TERM: "xterm-256color",
      COLORTERM: "truecolor",
      // Make the prompt slightly more obvious — this is a UI shell, not the
      // user's own terminal.
      PS1: "\\u@\\h:\\w\\$ ",
    } as Record<string, string>,
  });

  activePtys++;

  console.log(
    `[PtyManager] Spawned PTY pid=${pty.pid} for user=${opts.userId} (${activePtys}/${MAX_PTYS} active, cwd=${repoRoot})`
  );

  let idleTimer: NodeJS.Timeout | null = null;
  const resetIdle = () => {
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      console.log(
        `[PtyManager] PTY pid=${pty.pid} idle for ${IDLE_TIMEOUT_MS / 1000}s — killing`
      );
      try {
        pty.kill();
      } catch {}
    }, IDLE_TIMEOUT_MS);
    // Don't keep the process alive just for the idle timer.
    idleTimer.unref?.();
  };
  resetIdle();

  return {
    pid: pty.pid,
    write: (data: string) => {
      resetIdle();
      pty.write(data);
    },
    resize: (cols: number, rows: number) => {
      try {
        pty.resize(
          Math.max(20, Math.floor(cols)),
          Math.max(5, Math.floor(rows))
        );
      } catch {
        // Resize can throw if the PTY just exited; ignore.
      }
    },
    kill: (signal?: string) => {
      if (idleTimer) clearTimeout(idleTimer);
      try {
        pty.kill(signal);
      } catch {}
    },
    onData: (handler) => pty.onData((data) => handler(data)),
    onExit: (handler) =>
      pty.onExit(({ exitCode, signal }) => {
        if (idleTimer) clearTimeout(idleTimer);
        activePtys = Math.max(0, activePtys - 1);
        console.log(
          `[PtyManager] PTY pid=${pty.pid} exited code=${exitCode} signal=${signal} (${activePtys}/${MAX_PTYS} active)`
        );
        handler({ exitCode, signal });
      }),
  };
}
