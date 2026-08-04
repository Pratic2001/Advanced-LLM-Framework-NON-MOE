#!/bin/sh
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
#
# NOTE: This script is POSIX sh, not bash. node:22-alpine ships BusyBox ash
# with no bash installed, so we use POSIX `[ -f ... ]` and the `&&` short-
# circuit instead of `[[ -f ... ]]`. `set -u` keeps unbound variables from
# silently expanding to "".
# =============================================================================

set -eu

UI_PORT="${UI_PORT:-3000}"
LITE_PORT="${LITE_PORT:-3001}"

# Allow overrides for the env-file paths. Useful for CI smoke tests where
# the bind mount under /secrets may not exist; tests can pass UI_ENV_FILE /
# LITE_ENV_FILE as plain env vars (no file dependency).
UI_ENV_FILE="${UI_ENV_FILE:-/secrets/ui.env}"
LITE_ENV_FILE="${LITE_ENV_FILE:-/secrets/ui-lite.env}"

# Decide whether to feed tsx an --env-file=... flag. If the file is missing
# the process still runs, just without the .env overlay (env vars from the
# container process environment will still be picked up by Next).
if [ -f "${UI_ENV_FILE}" ]; then
  echo "[entrypoint] starting UI on :${UI_PORT} (BASE_PATH=/heavy, env=${UI_ENV_FILE})"
  UI_FLAGS="--env-file=${UI_ENV_FILE}"
else
  echo "[entrypoint] starting UI on :${UI_PORT} (BASE_PATH=/heavy, env=process)"
  UI_FLAGS=""
fi

if [ -f "${LITE_ENV_FILE}" ]; then
  echo "[entrypoint] starting UI_lite on :${LITE_PORT} (BASE_PATH=/lite, env=${LITE_ENV_FILE})"
  LITE_FLAGS="--env-file=${LITE_ENV_FILE}"
else
  echo "[entrypoint] starting UI_lite on :${LITE_PORT} (BASE_PATH=/lite, env=process)"
  LITE_FLAGS=""
fi

# Start UI in a subshell so backgrounding works even if the parent later
# changes directory. `exec` lets tsx become the child so wait() picks up
# its exit code.
(
  cd /app/ui
  PORT="${UI_PORT}" exec tsx ${UI_FLAGS} server.ts
) &
UI_PID=$!

(
  cd /app/ui_lite
  PORT="${LITE_PORT}" exec tsx ${LITE_FLAGS} server.ts
) &
LITE_PID=$!

# Wait for either process to exit, then tear down the other.
wait "${UI_PID}"
UI_EXIT=$?
echo "[entrypoint] UI exited with code ${UI_EXIT}, stopping UI_lite"
kill "${LITE_PID}" 2>/dev/null || true
wait "${LITE_PID}" 2>/dev/null || true
exit "${UI_EXIT}"
