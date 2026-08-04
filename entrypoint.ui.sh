#!/usr/bin/env bash
# =============================================================================
# UI container entrypoint — runs both UI and UI_lite server.ts processes.
#
# Compose mounts ./secrets/ui.env and ./secrets/ui-lite.env at /secrets/. We
# pass each through tsx's --env-file flag because:
#
#   • Bare `node`/`tsx` does NOT auto-load .env files (Next dev/start does).
#   • Both apps run as separate processes so each gets its own env file,
#     because they differ in BASE_PATH, NEXT_PUBLIC_BASE_PATH,
#     NEXT_PUBLIC_NEXTAUTH_URL, and PORT.
#
# If either process dies, kill the other and exit with the same code so
# docker-compose's restart policy can take over.
# =============================================================================

set -eu

UI_PORT="${UI_PORT:-3000}"
LITE_PORT="${LITE_PORT:-3001}"

# Allow overrides for the env-file paths. Useful for CI smoke tests where
# the bind mount under /secrets may not exist; tests can pass UI_ENV_FILE /
# LITE_ENV_FILE as plain env vars (no file dependency).
UI_ENV_FILE="${UI_ENV_FILE:-/secrets/ui.env}"
LITE_ENV_FILE="${LITE_ENV_FILE:-/secrets/ui-lite.env}"

if [[ -f "${UI_ENV_FILE}" ]]; then
  echo "[entrypoint] starting UI on :${UI_PORT} (BASE_PATH=/heavy, env=${UI_ENV_FILE})"
  TSX_FLAGS="--env-file=${UI_ENV_FILE}"
else
  echo "[entrypoint] starting UI on :${UI_PORT} (BASE_PATH=/heavy, env=process)"
  TSX_FLAGS=""
fi
(
  cd /app/ui
  PORT="${UI_PORT}" exec tsx ${TSX_FLAGS} server.ts
) &
UI_PID=$!

if [[ -f "${LITE_ENV_FILE}" ]]; then
  echo "[entrypoint] starting UI_lite on :${LITE_PORT} (BASE_PATH=/lite, env=${LITE_ENV_FILE})"
  TSX_FLAGS="--env-file=${LITE_ENV_FILE}"
else
  echo "[entrypoint] starting UI_lite on :${LITE_PORT} (BASE_PATH=/lite, env=process)"
  TSX_FLAGS=""
fi
(
  cd /app/ui_lite
  PORT="${LITE_PORT}" exec tsx ${TSX_FLAGS} server.ts
) &
LITE_PID=$!

# Wait for either process to exit, then tear down the other.
wait "${UI_PID}"
UI_EXIT=$?
echo "[entrypoint] UI exited with code ${UI_EXIT}, stopping UI_lite"
kill "${LITE_PID}" 2>/dev/null || true
wait "${LITE_PID}" 2>/dev/null || true
exit "${UI_EXIT}"