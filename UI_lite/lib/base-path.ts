// Client- and server-safe base path for the funnel subpath. See
// UI/lib/base-path.ts (this app is served at /lite).
export const BASE_PATH = (process.env.NEXT_PUBLIC_BASE_PATH ?? "").replace(/\/$/, "");

export const api = (p: string) => `${BASE_PATH}${p}`;