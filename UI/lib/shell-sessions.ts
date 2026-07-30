/**
 * Shell Session Registry
 *
 * Maps a client-side `sessionId` (a random opaque token) to a live `ShellPty`,
 * scoped per-user. Lets us drive a long-lived PTY through short-lived HTTP
 * requests (SSE + POST) instead of a single WebSocket — required because
 * Tailscale Funnel aggressively kills WebSocket upgrades within milliseconds
 * of opening, while normal HTTP requests pass through fine.
 *
 * Lifecycle:
 *   1. Client POSTs `/api/shell/open` → server spawns PTY, returns sessionId.
 *   2. Client opens `/api/shell/stream?sessionId=...` → SSE stream of PTY
 *      stdout/stderr framed as `data:` lines.
 *   3. Client POSTs `/api/shell/input?sessionId=...` to feed stdin, and
 *      `/api/shell/resize?sessionId=...` to resize.
 *   4. On unmount / disconnect, client POSTs `/api/shell/close` (or the SSE
 *      stream ends and the server's idle timer eventually kills the PTY).
 *
 * Security:
 *   - sessionIds are 128-bit random tokens. Unguessable.
 *   - We refuse to return a session that doesn't belong to the requesting
 *     userId, so a leaked sessionId from another user doesn't grant access.
 *   - Per-user PTY count cap (separate from the global MAX_PTYS) prevents
 *     one user from monopolising the pool.
 */

import { spawnShellPty, getActivePtyCount, type ShellPty } from "./pty-manager";

const IDLE_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes
const MAX_PTYS_PER_USER = 5;

interface SessionRecord {
  pty: ShellPty;
  userId: string;
  /** Subscribers that get every PTY frame; flushed FIFO. */
  subscribers: Set<(chunk: string) => void>;
  /** Subscribers that get a single 'exit' notification then auto-unsubscribe. */
  exitSubscribers: Set<(info: { exitCode: number; signal?: number }) => void>;
  /** Idle timer — reset on every write/stream-attach. */
  idleTimer: NodeJS.Timeout | null;
  createdAt: number;
}

const sessions = new Map<string, SessionRecord>();
let userPtyCount = new Map<string, number>();

function resetIdle(rec: SessionRecord, sessionId: string) {
  if (rec.idleTimer) clearTimeout(rec.idleTimer);
  rec.idleTimer = setTimeout(() => {
    console.log(
      `[ShellSessions] session=${sessionId} idle for ${IDLE_TIMEOUT_MS / 1000}s — killing PTY`
    );
    killSession(sessionId);
  }, IDLE_TIMEOUT_MS);
  rec.idleTimer.unref?.();
}

function generateSessionId(): string {
  // 32 hex chars = 128 bits — collision-resistant enough for PTY session keys.
  return (
    "sh_" +
    Date.now().toString(36) +
    "_" +
    Math.random().toString(36).slice(2, 10) +
    Math.random().toString(36).slice(2, 10)
  );
}

export interface OpenShellResult {
  sessionId: string;
  pid: number;
  activePtys: number;
}

/**
 * Spawn a fresh bash PTY for this user. Throws if the global pool is full
 * or this user has hit their per-user cap.
 */
export function openShell(opts: {
  userId: string;
  cols: number;
  rows: number;
}): OpenShellResult {
  if (!sessions.size && !userPtyCount.size) {
    // First call — nothing to do, just initializing.
  }

  const userCount = userPtyCount.get(opts.userId) ?? 0;
  if (userCount >= MAX_PTYS_PER_USER) {
    throw new Error(
      `Too many open shells for this user (max ${MAX_PTYS_PER_USER}). Close an existing shell first.`
    );
  }

  let pty: ShellPty;
  try {
    pty = spawnShellPty({
      cols: opts.cols,
      rows: opts.rows,
      userId: opts.userId,
    });
  } catch (err) {
    throw err;
  }

  const sessionId = generateSessionId();
  const rec: SessionRecord = {
    pty,
    userId: opts.userId,
    subscribers: new Set(),
    exitSubscribers: new Set(),
    idleTimer: null,
    createdAt: Date.now(),
  };

  // Fan out PTY stdout to every subscriber.
  pty.onData((chunk) => {
    for (const sub of Array.from(rec.subscribers)) {
      try {
        sub(chunk);
      } catch (err) {
        console.error(`[ShellSessions] subscriber threw:`, err);
      }
    }
  });

  pty.onExit(({ exitCode, signal }) => {
    for (const exitSub of Array.from(rec.exitSubscribers)) {
      try {
        exitSub({ exitCode, signal });
      } catch (err) {
        console.error(`[ShellSessions] exit subscriber threw:`, err);
      }
    }
    // Tear down the session — PTY is gone.
    const r = sessions.get(sessionId);
    if (r) {
      if (r.idleTimer) clearTimeout(r.idleTimer);
      sessions.delete(sessionId);
      const c = userPtyCount.get(opts.userId) ?? 1;
      userPtyCount.set(opts.userId, Math.max(0, c - 1));
    }
  });

  sessions.set(sessionId, rec);
  userPtyCount.set(opts.userId, userCount + 1);
  resetIdle(rec, sessionId);

  return { sessionId, pid: pty.pid, activePtys: getActivePtyCount() };
}

/**
 * Look up a session. Returns null if the sessionId is unknown OR if the
 * caller doesn't own it. The userId check is what stops a leaked sessionId
 * from another user being usable.
 */
export function getSession(
  sessionId: string,
  userId: string
): SessionRecord | null {
  const rec = sessions.get(sessionId);
  if (!rec) return null;
  if (rec.userId !== userId) return null;
  resetIdle(rec, sessionId);
  return rec;
}

/**
 * Subscribe to PTY output. Returns an unsubscribe function. The handler
 * receives each chunk of stdout verbatim (it's the caller's job to encode
 * it onto whatever transport the consumer is using).
 */
export function subscribeOutput(
  rec: SessionRecord,
  handler: (chunk: string) => void
): () => void {
  rec.subscribers.add(handler);
  return () => {
    rec.subscribers.delete(handler);
  };
}

/**
 * Subscribe to PTY exit. Handler is invoked exactly once then auto-removed.
 * Useful for the SSE stream to emit a final `event: exit` frame and close.
 */
export function subscribeExit(
  rec: SessionRecord,
  handler: (info: { exitCode: number; signal?: number }) => void
): () => void {
  rec.exitSubscribers.add(handler);
  return () => {
    rec.exitSubscribers.delete(handler);
  };
}

/**
 * Write to PTY stdin. Resets the idle timer.
 */
export function writeInput(rec: SessionRecord, data: string): void {
  rec.pty.write(data);
}

/**
 * Resize PTY. No-op if PTY already exited.
 */
export function resizeShell(rec: SessionRecord, cols: number, rows: number): void {
  rec.pty.resize(cols, rows);
}

/**
 * Kill the PTY and remove the session.
 */
export function killSession(sessionId: string): boolean {
  const rec = sessions.get(sessionId);
  if (!rec) return false;
  try {
    rec.pty.kill();
  } catch {}
  if (rec.idleTimer) clearTimeout(rec.idleTimer);
  const c = userPtyCount.get(rec.userId) ?? 1;
  userPtyCount.set(rec.userId, Math.max(0, c - 1));
  sessions.delete(sessionId);
  return true;
}

/**
 * Diagnostics — for health/admin endpoints.
 */
export function getShellStats() {
  return {
    activeSessions: sessions.size,
    activePtys: getActivePtyCount(),
    byUser: Object.fromEntries(userPtyCount.entries()),
  };
}