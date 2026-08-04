#!/bin/bash
# =============================================================================
# install.sh — fresh-PC setup helper for the LLMForge container stack.
#
# What it does:
#   1. Sanity-checks that docker + docker compose are installed.
#   2. Generates the four ./secrets/*.env files the stack needs.
#      (Only prompts for values you don't already have — re-runs are safe.)
#   3. Generates an ed25519 SSH keypair so the UI container can SSH into
#      the trainer container. Existing ./secrets/ui-sshkey is reused if
#      found.
#   4. `docker compose pull` and `docker compose up -d`.
#   5. Prints next-step diagnostics (URL, container status, nvidia-smi).
#
# What it does NOT do:
#   • Install docker itself — see the README's "Fresh-PC prerequisites".
#   • Install the Tailscale auth key — you must paste `tskey-...` when
#     prompted (it's tied to your account, never stored in version control).
#   • Push new images. Pull only.
#
# Usage:
#   ./install.sh                 # interactive (recommended for first run)
#   ./install.sh --non-interactive \
#     --public-hostname=h.ts.net \
#     --ts-authkey=tskey-xxx \
#     --dockerhub-username=pratic2001 \
#     --tag=v0.2.0
#
# Environment overrides (any of these can be set instead of being prompted):
#   PUBLIC_HOSTNAME, TS_AUTHKEY, DOCKERHUB_USERNAME, IMAGE_TAG
# =============================================================================

set -euo pipefail

# ── CLI flag parsing ────────────────────────────────────────────────────────
NON_INTERACTIVE=0
PUBLIC_HOSTNAME="${PUBLIC_HOSTNAME:-}"
TS_AUTHKEY="${TS_AUTHKEY:-}"
DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-pratic2001}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --non-interactive) NON_INTERACTIVE=1; shift ;;
    --public-hostname=*) PUBLIC_HOSTNAME="${1#*=}"; shift ;;
    --ts-authkey=*) TS_AUTHKEY="${1#*=}"; shift ;;
    --dockerhub-username=*) DOCKERHUB_USERNAME="${1#*=}"; shift ;;
    --tag=*) IMAGE_TAG="${1#*=}"; shift ;;
    -h|--help)
      sed -n '2,40p' "$0"
      exit 0
      ;;
    *) echo "unknown flag: $1"; exit 2 ;;
  esac
done

# ── Helpers ────────────────────────────────────────────────────────────────
BOLD=$'\e[1m'; DIM=$'\e[2m'; RESET=$'\e[0m'; OK=$'\e[32m'; WARN=$'\e[33m'; ERR=$'\e[31m'
say()  { printf "%s\n" "$*"; }
hdr()  { printf "\n%s%s%s\n" "$BOLD" "$*" "$RESET"; }
ok()   { printf "%s✓%s %s\n" "$OK" "$RESET" "$*"; }
warn() { printf "%s!%s %s\n" "$WARN" "$RESET" "$*" >&2; }
err()  { printf "%s✗%s %s\n" "$ERR" "$RESET" "$*" >&2; }

# Prompt for a value if it's empty. In non-interactive mode, error out.
prompt_or_die() {
  local var_name="$1"
  local prompt_text="$2"
  local current="${!var_name:-}"
  if [[ -n "$current" ]]; then
    return 0
  fi
  if [[ "$NON_INTERACTIVE" == "1" ]]; then
    err "--${var_name,,} is required in --non-interactive mode"
    exit 1
  fi
  local value
  read -r -p "$(printf '%s' "$prompt_text: '")" value </dev/tty
  value="${value## }"; value="${value%% }"
  if [[ -z "$value" ]]; then
    err "no value entered for $var_name"
    exit 1
  fi
  printf -v "$var_name" '%s' "$value"
  export "$var_name"
}

# Generate a 32-char base64 secret if $1 isn't already set.
ensure_secret() {
  local var_name="$1"
  local current="${!var_name:-}"
  if [[ -n "$current" ]]; then
    return 0
  fi
  local generated
  generated="$(openssl rand -base64 32 | tr -d '\n')"
  printf -v "$var_name" '%s' "$generated"
  export "$var_name"
}

# ── Step 1: prerequisites ───────────────────────────────────────────────────
hdr "1/6 · Checking prerequisites"

command -v docker >/dev/null 2>&1 || { err "docker not installed — see README 'Fresh-PC prerequisites'"; exit 1; }
command -v docker >/dev/null && docker compose version >/dev/null 2>&1 \
  || { err "'docker compose' plugin not installed (Docker 20.10+ required)"; exit 1; }
command -v openssl >/dev/null 2>&1 || { err "openssl not installed"; exit 1; }
ok "docker + docker compose + openssl present"

# Optional: tailscale. Only required if the user is going to run the
# `ts-*` sidecars (they are on by default — see Step 5).
if command -v tailscale >/dev/null 2>&1; then
  ok "tailscale installed"
else
  warn "tailscale CLI not found — install it from https://tailscale.com/download if you want to run the bundled sidecars"
fi

# ── Step 2: gather config ───────────────────────────────────────────────────
hdr "2/6 · Gathering config"

prompt_or_die PUBLIC_HOSTNAME "Your Tailscale MagicDNS hostname (e.g. pratic-battleaxb450mkm2.tail5e5151.ts.net)"
prompt_or_die TS_AUTHKEY "Tailscale auth key (tskey-...; https://login.tailscale.com/admin/settings/keys)"

# Generate runtime secrets if missing. The same secret value is used for
# NEXTAUTH_SECRET (decoded by ws-manager for the interactive shell cookie)
# and AUTH_SECRET (used by Auth.js to sign the session cookie). They MUST
# match — see UI/.env.example for the failure mode.
ensure_secret NEXTAUTH_SECRET
AUTH_SECRET="$NEXTAUTH_SECRET"
export AUTH_SECRET
ensure_secret SSH_KEY_ENCRYPTION_KEY
# SSH_KEY_ENCRYPTION_KEY must be exactly 32 hex chars (AES-256-GCM).
if [[ "${#SSH_KEY_ENCRYPTION_KEY}" -ne 32 ]]; then
  SSH_KEY_ENCRYPTION_KEY="$(openssl rand -hex 16 | tr -d '\n')"
  export SSH_KEY_ENCRYPTION_KEY
  warn "SSH_KEY_ENCRYPTION_KEY regenerated to 32 hex chars"
fi

# DATABASE_URL defaults to a postgres-on-host that the UI reaches via
# host.docker.internal. Override if you're putting postgres somewhere else.
DATABASE_URL="${DATABASE_URL:-postgresql://postgres:postgres@host.docker.internal:5432/llm_training_ui}"
export DATABASE_URL

ok "public hostname  : $PUBLIC_HOSTNAME"
ok "docker hub user  : $DOCKERHUB_USERNAME"
ok "image tag        : $IMAGE_TAG"
ok "postgres URL     : $DATABASE_URL"

# ── Step 3: SSH keypair ────────────────────────────────────────────────────
hdr "3/6 · SSH keypair (UI → trainer)"

mkdir -p secrets
if [[ ! -f secrets/ui-sshkey ]]; then
  if [[ "$NON_INTERACTIVE" == "1" ]]; then
    ssh-keygen -t ed25519 -N '' -f secrets/ui-sshkey -C "llmforge-ui-$(date -u +%FT%TZ)" >/dev/null
    ok "generated secrets/ui-sshkey (ed25519)"
  else
    read -r -p "Generate new SSH keypair for UI→trainer? [Y/n] " ans </dev/tty || true
    ans="${ans:-Y}"
    if [[ "$ans" =~ ^[Yy]$ ]]; then
      ssh-keygen -t ed25519 -N '' -f secrets/ui-sshkey -C "llmforge-ui-$(date -u +%FT%TZ)" >/dev/null
      ok "generated secrets/ui-sshkey (ed25519)"
    else
      err "without an SSH keypair the UI cannot reach the trainer container"
      err "place secrets/ui-sshkey (private) and secrets/ui-sshkey.pub (public) here and re-run"
      exit 1
    fi
  fi
else
  ok "reusing existing secrets/ui-sshkey"
fi

# ── Step 4: secrets/*.env ──────────────────────────────────────────────────
hdr "4/6 · Writing secrets/*.env"

# NEXTAUTH_URL points at the bare origin so Auth.js can derive its
# auth base path; the UI passes NEXT_PUBLIC_NEXTAUTH_URL with /api/auth
# appended so the client knows where to POST credentials.
NEXTAUTH_URL="https://${PUBLIC_HOSTNAME}"

cat > secrets/router.env <<EOF
# Generated by install.sh — do not edit by hand.
NEXT_PUBLIC_HEAVY_UI_URL=https://${PUBLIC_HOSTNAME}/heavy
NEXT_PUBLIC_LITE_UI_URL=https://${PUBLIC_HOSTNAME}/lite
EOF

cat > secrets/ui.env <<EOF
# Generated by install.sh — do not edit by hand.
NODE_ENV=production
PORT=3000
HOSTNAME=0.0.0.0
DATABASE_URL=${DATABASE_URL}
NEXTAUTH_URL=${NEXTAUTH_URL}
NEXTAUTH_SECRET=${NEXTAUTH_SECRET}
AUTH_SECRET=${AUTH_SECRET}
AUTH_TRUST_HOST=true
NEXT_PUBLIC_BASE_PATH=/heavy
NEXT_PUBLIC_NEXTAUTH_URL=https://${PUBLIC_HOSTNAME}/heavy/api/auth
SSH_KEY_ENCRYPTION_KEY=${SSH_KEY_ENCRYPTION_KEY}
REPO_ROOT=../Advanced-LLM-Framework-NON-MOE/
EOF

cat > secrets/ui-lite.env <<EOF
# Generated by install.sh — do not edit by hand.
NODE_ENV=production
PORT=3001
HOSTNAME=0.0.0.0
DATABASE_URL=${DATABASE_URL}
NEXTAUTH_URL=${NEXTAUTH_URL}
NEXTAUTH_SECRET=${NEXTAUTH_SECRET}
AUTH_SECRET=${AUTH_SECRET}
AUTH_TRUST_HOST=true
NEXT_PUBLIC_BASE_PATH=/lite
NEXT_PUBLIC_NEXTAUTH_URL=https://${PUBLIC_HOSTNAME}/lite/api/auth
SSH_KEY_ENCRYPTION_KEY=${SSH_KEY_ENCRYPTION_KEY}
REPO_ROOT=../Advanced-LLM-Framework-NON-MOE/
EOF

cat > secrets/trainer.env <<EOF
# Generated by install.sh — do not edit by hand.
# Trainer container has no runtime env requirements; this file exists so
# docker-compose's env_file directive does not error on first run.
EOF

ok "secrets/{router,ui,ui-lite,trainer}.env"

# Export for docker compose.
export TAG="$IMAGE_TAG"

# ── Step 5: pull + boot ────────────────────────────────────────────────────
hdr "5/6 · Pulling & starting containers"

# If the trainer is on a different host in a multi-PC setup, the user
# would have started this script on the UI-host with --no-trainer or
# with a custom compose file. We always start by checking that all three
# images are reachable.
docker compose pull ui-router ui trainer 2>&1 | tail -5 || {
  err "docker compose pull failed — are you logged in to Docker Hub as $DOCKERHUB_USERNAME?"
  err "  docker login -u $DOCKERHUB_USERNAME"
  exit 1
}
ok "images pulled"

# Make sure Tailscale state dirs are clean on first run.
docker volume ls --format '{{.Name}}' | grep -q '^llmforge_ts-state-' \
  || ok "(first run) tailscale state will be created on first boot"

docker compose up -d
ok "containers started"

# ── Step 6: diagnostics ────────────────────────────────────────────────────
hdr "6/6 · Verifying"

# Wait up to 60s for the UI containers to report healthy.
for i in $(seq 1 30); do
  if docker compose ps --format json 2>/dev/null \
     | grep -q '"Health":"healthy"' 2>/dev/null; then
    break
  fi
  sleep 2
done

docker compose ps

if docker compose ps --services --filter "status=running" 2>/dev/null | grep -q '^trainer$'; then
  echo
  if docker exec trainer nvidia-smi 2>/dev/null | head -20; then
    ok "GPU is visible inside the trainer container"
  else
    warn "trainer is up but nvidia-smi failed — install the NVIDIA Container Toolkit"
    warn "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
  fi
fi

cat <<EOF

${BOLD}Done.${RESET}  Open https://${PUBLIC_HOSTNAME}/ in a browser on your tailnet.

Next steps:
  1. Log in (default credentials: see the project's README or your operator).
  2. Settings → Nodes → Register Node:
       hostname = trainer
       port     = 22
       username = trainer
       key      = $(cat secrets/ui-sshkey)
  3. Test Connection — should report your GPU model.
  4. Launch a smoke pretrain job.

To update later:
  TAG=v0.3.0 ./install.sh        # pulls new tag, restarts in place

To tear down:
  docker compose down            # keep secrets/ and Docker volumes
  docker compose down -v         # also wipe tailscale state + workspace
EOF