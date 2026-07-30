/**
 * Simple in-memory rate limiter.
 * Stores request counts per key within a sliding window.
 * Note: In multi-process deployments, use an external store (Redis).
 */

interface RateLimitEntry {
  count: number;
  resetAt: number;
}

const store = new Map<string, RateLimitEntry>();

// Clean up expired entries every 60 seconds
setInterval(() => {
  const now = Date.now();
  store.forEach((entry, key) => {
    if (entry.resetAt <= now) {
      store.delete(key);
    }
  });
}, 60000).unref();

export interface RateLimitResult {
  success: boolean;
  remaining: number;
  resetAt: number;
}

/**
 * Check if a request should be rate-limited.
 *
 * @param key - Unique identifier (e.g. IP address or user ID)
 * @param maxRequests - Maximum number of requests allowed in the window
 * @param windowMs - Time window in milliseconds
 */
export function rateLimit(
  key: string,
  maxRequests: number,
  windowMs: number
): RateLimitResult {
  const now = Date.now();
  const entry = store.get(key);

  if (!entry || entry.resetAt <= now) {
    store.set(key, { count: 1, resetAt: now + windowMs });
    return { success: true, remaining: maxRequests - 1, resetAt: now + windowMs };
  }

  entry.count += 1;

  if (entry.count > maxRequests) {
    return { success: false, remaining: 0, resetAt: entry.resetAt };
  }

  return { success: true, remaining: maxRequests - entry.count, resetAt: entry.resetAt };
}

/**
 * Create a rate-limited handler wrapper for API routes.
 * Returns a 429 response if rate-limited.
 */
import { NextResponse } from "next/server";

export function withRateLimit(
  key: string,
  maxRequests: number,
  windowMs: number
): { success: boolean; headers: Record<string, string> } | NextResponse {
  const result = rateLimit(key, maxRequests, windowMs);

  const headers: Record<string, string> = {
    "X-RateLimit-Limit": String(maxRequests),
    "X-RateLimit-Remaining": String(result.remaining),
    "X-RateLimit-Reset": String(Math.ceil(result.resetAt / 1000)),
  };

  if (!result.success) {
    return NextResponse.json(
      { error: "Too many requests. Please try again later." },
      {
        status: 429,
        headers,
      }
    );
  }

  return { success: true, headers };
}
