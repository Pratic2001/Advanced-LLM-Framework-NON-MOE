# LLMForge — Web UI

> **Browser-based control center for the [Advanced LLM Framework](../).**
> Orchestrate the full LLM training pipeline — tokenizer → data packing → pretrain →
> SFT → GRPO/DPO — across single-node, multi-node, and Hivemind (decentralized)
> backends, with live monitoring and WebSocket-driven charts.

**Stack:** Next.js 16 (App Router, Turbopack) · React 19 · TypeScript 5.9 ·
PostgreSQL 14+ via Prisma 6 · NextAuth v5 (Auth.js) · Tailwind v4 · Framer Motion 12

---

## Table of contents

- [Prerequisites](#prerequisites)
- [Quickstart — development (lazy compilation)](#quickstart--development-lazy-compilation)
- [Quickstart — production (fully precompiled, with WebSockets)](#quickstart--production-fully-precompiled-with-websockets)
- [Docker — production](#docker--production)
- [Bare-metal / VM — production](#bare-metal--vm--production)
- [Reverse proxy + TLS (production)](#reverse-proxy--tls-production)
- [Environment variables reference](#environment-variables-reference)
- [Database commands](#database-commands)
- [Upgrades and rollback](#upgrades-and-rollback)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Tool | Version | Check |
|---|---|---|
| Node.js | 20 LTS or 22 LTS | `node --version` |
| npm | 10+ | `npm --version` |
| PostgreSQL | 14+ (16 recommended) | `psql --version` |
| Python | 3.10+ | `python3 --version` |
| OpenSSL | any | `openssl version` |
| Git | any | `git --version` |

---

## Quickstart — development (lazy compilation)

> **What this gives you:** fastest edit-restart loop, on-demand route
> compilation. Pages and API routes compile the first time you hit them.
> WebSockets work because `next dev` ships its own WS upgrade. **Not for
> production.**

```bash
# ── 1. Get the source ──────────────────────────────────────────────────────
git clone <repo-url> Advanced-LLM-Framework-NON-MOE
cd Advanced-LLM-Framework-NON-MOE/UI

# ── 2. Install dependencies (devDeps are REQUIRED here) ────────────────────
# Why --include=dev: a plain `npm install` may prune devDeps (Tailwind
# PostCSS plugin, Prisma CLI, etc.) on certain npm configs and break the
# build. --include=dev forces them to be installed.
npm install --include=dev

# ── 3. Start PostgreSQL (Ubuntu/Debian — adjust for your OS) ───────────────
sudo apt update && sudo apt install -y postgresql postgresql-contrib
sudo systemctl start postgresql && sudo systemctl enable postgresql
sudo -u postgres psql -c "CREATE USER postgres WITH PASSWORD 'postgres';"
sudo -u postgres psql -c "ALTER USER postgres WITH SUPERUSER;"
sudo -u postgres psql -c "CREATE DATABASE llm_training_ui OWNER postgres;"
# Switch local auth to md5 so the password above is honored:
sudo sed -i 's/peer/md5/g' $(sudo -u postgres psql -c "SHOW hba_file;" -t | tr -d ' ')
sudo systemctl restart postgresql

# ── 4. Configure secrets ───────────────────────────────────────────────────
cp .env.example .env.local
# Edit .env.local — at minimum, replace the placeholder AUTH_SECRET and
# NEXTAUTH_SECRET (use the same value for both). See the env reference below
# for the full list.
sed -i "s|your-secret-key-change-in-production|$(openssl rand -base64 48)|" .env/local \
  || sed -i "s|your-secret-key-change-in-production|$(openssl rand -base64 48)|" .env.local

# ── 5. Generate the Prisma client + push the schema to Postgres ────────────
npx prisma generate
npx prisma db push
# Verify the tables were created:
psql "postgresql://postgres:postgres@localhost:5432/llm_training_ui" -c "\dt"
# Expect: ConfigPreset, Job, JobMetric, Node, Pipeline, PipelineStage,
#         SshKey, SystemEvent, User

# ── 6. Start the dev server ────────────────────────────────────────────────
npm run dev
#   →  Local:    http://localhost:3000
#   →  Press Ctrl-C to stop
```

Open <http://localhost:3000>, click **Get Started**, register the first user at
`/signup` (sign-up is open by default — see how to lock it down in
[Troubleshooting](#troubleshooting)).

That's it for development. To re-enter later:

```bash
cd Advanced-LLM-Framework-NON-MOE/UI
npm run dev
```

If you change `prisma/schema.prisma`:

```bash
npx prisma generate
npx prisma db push   # or: npx prisma migrate dev --name <change>
```

---

## Quickstart — production (fully precompiled, with WebSockets)

> **What this gives you:** every page, layout, and API route compiled ahead of
> time into `.next/`. **No route compiles at request time.** WebSockets work
> because the bundled custom server (`server.ts`) keeps `WSManager` mounted on
> top of Next's HTTP handler. `next start` alone does **not** support WS — see
> the warning at the end of this section.

```bash
# ── 1. Get the source ──────────────────────────────────────────────────────
git clone <repo-url> Advanced-LLM-Framework-NON-MOE
cd Advanced-LLM-Framework-NON-MOE/UI

# ── 2. Install deps ────────────────────────────────────────────────────────
npm install --include=dev

# ── 3. Provision PostgreSQL (same as the dev section, or a remote DB) ──────
sudo apt update && sudo apt install -y postgresql postgresql-contrib
sudo systemctl start postgresql && sudo systemctl enable postgresql
sudo -u postgres psql -c "CREATE USER llmforge WITH PASSWORD 'STRONG_PASSWORD';"
sudo -u postgres psql -c "CREATE DATABASE llmforge OWNER llmforge;"
sudo sed -i 's/peer/md5/g' $(sudo -u postgres psql -c "SHOW hba_file;" -t | tr -d ' ')
sudo systemctl restart postgresql
PGPASSWORD='STRONG_PASSWORD' psql -h localhost -U llmforge -d llmforge -c "SELECT 1;"

# ── 4. Generate strong secrets ─────────────────────────────────────────────
echo "AUTH_SECRET=$(openssl rand -base64 48)"
echo "NEXTAUTH_SECRET=$(openssl rand -base64 48)"   # MUST equal AUTH_SECRET
echo "SSH_KEY_ENCRYPTION_KEY=$(openssl rand -hex 16)" # exactly 32 hex chars
#   ↑ losing SSH_KEY_ENCRYPTION_KEY destroys every stored SSH key.

# ── 5. Write .env.production (NOT .env.local — server.ts loads the former) ─
cat > .env.production <<'EOF'
NODE_ENV=production
PORT=3000
HOSTNAME=0.0.0.0

DATABASE_URL="postgresql://llmforge:STRONG_PASSWORD@localhost:5432/llmforge"

# Use the values you generated above — keep AUTH_SECRET and NEXTAUTH_SECRET equal.
AUTH_SECRET="paste-from-step-4"
NEXTAUTH_SECRET="paste-from-step-4"
AUTH_URL="http://localhost:3000"
NEXTAUTH_URL="http://localhost:3000"
AUTH_TRUST_HOST="true"

SSH_KEY_ENCRYPTION_KEY="paste-from-step-4"
REPO_ROOT="../"
EOF
chmod 600 .env.production

# ── 6. Generate Prisma client + push schema (or migrate deploy later) ──────
npx prisma generate
npx prisma db push

# ── 7. Precompile everything ───────────────────────────────────────────────
# This is the step that eliminates lazy/on-demand compilation — every route
# is emitted into .next/ now, not on first request.
npm run build

# ── 8. Start the WebSocket-capable server from the prebuilt output ────────
npm run start:ws
#   →  Loads .env.local (override path with --env-file if you used a different
#      name), passes dev:false into Next because NODE_ENV=production, so Next
#      serves only .next/ — nothing compiles at request time.
#   →  http://<HOSTNAME>:<PORT>  (defaults to 0.0.0.0:3000)
```

> ⚠️ **`npm start` (i.e. `next start`) does NOT support WebSockets** — it runs
> Next's standalone server, which never initializes `WSManager`, so `/api/ws`
> upgrades crash with `Cannot read properties of undefined (reading 'bind')`.
> Always use `npm run start:ws` for production if you need live job charts or
> the interactive terminal. **Sanity check:** the first request to *any* route
> should return immediately with no compile delay — everything was already
> emitted by `npm run build`.

---

## Docker — production

The included `Dockerfile` produces a Next.js standalone image (~150 MB) that
runs as the non-root `nextjs` user (uid 1001) and listens on `:3000`.

```bash
# ── 1. Prepare a deployment directory ──────────────────────────────────────
sudo mkdir -p /opt/llmforge-ui && cd /opt/llmforge-ui
# Copy the project (Dockerfile, package.json, package-lock.json, app/, lib/,
# hooks/, components/, public/, prisma/, next.config.mjs, tsconfig.json,
# server.ts, instrumentation.ts, proxy.ts).

# ── 2. Build the image ─────────────────────────────────────────────────────
sudo docker build -t llmforge-ui:1.0.0 .

# ── 3. Provide secrets on the host, NOT inside the image ───────────────────
sudo tee /opt/llmforge-ui/.env.production >/dev/null <<'EOF'
NODE_ENV=production
PORT=3000
HOSTNAME=0.0.0.0
AUTH_TRUST_HOST=true
AUTH_URL=https://forge.example.com
NEXTAUTH_URL=https://forge.example.com
AUTH_SECRET=paste-from-step-4-of-quickstart
NEXTAUTH_SECRET=paste-from-step-4-of-quickstart
SSH_KEY_ENCRYPTION_KEY=paste-from-step-4-of-quickstart
DATABASE_URL=postgresql://llmforge:STRONG_PASSWORD@db.internal:5432/llmforge?sslmode=require
REPO_ROOT=/framework
EOF
sudo chmod 600 /opt/llmforge-ui/.env.production

# ── 4. Apply Prisma migrations once (image does not auto-migrate) ──────────
sudo docker run --rm \
  --env-file /opt/llmforge-ui/.env.production \
  llmforge-ui:1.0.0 \
  npx prisma migrate deploy

# ── 5. Start the container ─────────────────────────────────────────────────
sudo docker run -d \
  --name llmforge-ui \
  --restart unless-stopped \
  --env-file /opt/llmforge-ui/.env.production \
  -p 127.0.0.1:3000:3000 \
  -v /mnt/framework:/framework:ro \
  -v /var/lib/llmforge/output:/output \
  llmforge-ui:1.0.0

# ── 6. Verify ──────────────────────────────────────────────────────────────
curl -s http://127.0.0.1:3000/api/system/health | jq
# Expect: { "status": "ok", "db": "ok", ... }
```

`docker-compose.yml` for the full stack (UI + Postgres on a private network):

```yaml
services:
  ui:
    image: llmforge-ui:1.0.0
    restart: unless-stopped
    env_file: .env.production
    ports:
      - "127.0.0.1:3000:3000"
    volumes:
      - /mnt/framework:/framework:ro
      - /var/lib/llmforge/output:/output
    depends_on: [db]

  db:
    image: postgres:16-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB: llmforge
      POSTGRES_USER: llmforge
      POSTGRES_PASSWORD: STRONG_PASSWORD
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

```bash
sudo docker compose run --rm ui npx prisma migrate deploy
sudo docker compose up -d
```

---

## Bare-metal / VM — production

Use this when you don't want Docker (e.g., a single GPU workstation that also
runs training jobs locally).

```bash
# ── 1. Provision a dedicated system user ───────────────────────────────────
sudo useradd -r -m -d /var/lib/llmforge -s /bin/bash llmforge
sudo mkdir -p /opt/llmforge-ui /var/lib/llmforge/output
sudo chown -R llmforge:llmforge /opt/llmforge-ui /var/lib/llmforge/output

# ── 2. Install ─────────────────────────────────────────────────────────────
sudo -u llmforge -H bash <<'EOF'
cd /opt/llmforge-ui
git clone <repo-url> .
cd UI
npm install --include=dev
npx prisma generate
npm run build
EOF

# ── 3. Write .env.production ───────────────────────────────────────────────
sudo -u llmforge tee /opt/llmforge-ui/UI/.env.production >/dev/null <<'EOF'
NODE_ENV=production
PORT=3000
HOSTNAME=0.0.0.0
DATABASE_URL="postgresql://llmforge:STRONG_PASSWORD@localhost:5432/llmforge"
AUTH_SECRET="paste-from-step-4"
NEXTAUTH_SECRET="paste-from-step-4"
AUTH_URL="http://localhost:3000"
NEXTAUTH_URL="http://localhost:3000"
AUTH_TRUST_HOST="true"
SSH_KEY_ENCRYPTION_KEY="paste-from-step-4"
REPO_ROOT="../"
EOF
sudo chmod 600 /opt/llmforge-ui/UI/.env.production
sudo chown llmforge:llmforge /opt/llmforge-ui/UI/.env.production

# ── 4. Apply migrations ────────────────────────────────────────────────────
# Use `migrate deploy` in production, NOT `db push`. `deploy` applies SQL
# files in order from prisma/migrations/ — safe to re-run on every deploy.
sudo -u llmforge -H bash -c 'cd /opt/llmforge-ui/UI && npx prisma migrate deploy'

# ── 5. Install the systemd unit ────────────────────────────────────────────
sudo tee /etc/systemd/system/llmforge-ui.service >/dev/null <<'EOF'
[Unit]
Description=LLMForge UI (Next.js + WebSocket)
After=network-online.target postgresql.service
Wants=network-online.target

[Service]
Type=simple
User=llmforge
Group=llmforge
WorkingDirectory=/opt/llmforge-ui/UI
EnvironmentFile=/opt/llmforge-ui/UI/.env.production
ExecStart=/usr/bin/npm run start:ws
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/llmforge/output
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now llmforge-ui
sudo systemctl status llmforge-ui
curl -s http://127.0.0.1:3000/api/system/health | jq
```

---

## Reverse proxy + TLS (production)

The UI is bound to `127.0.0.1:3000`. Add a reverse proxy for TLS termination,
`Host` / `X-Forwarded-*` headers (which Auth.js needs), and WebSocket
upgrades for `/api/ws`.

### Caddy (automatic HTTPS via Let's Encrypt)

```caddyfile
# /etc/caddy/Caddyfile
forge.example.com {
    encode zstd gzip
    reverse_proxy 127.0.0.1:3000 {
        header_up Host {host}
        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-For {remote_host}
        header_up X-Forwarded-Proto {scheme}
    }
}
```

```bash
sudo systemctl reload caddy
```

### nginx

```nginx
# /etc/nginx/sites-available/forge
upstream llmforge_app {
    server 127.0.0.1:3000;
    keepalive 32;
}

server {
    listen 80;
    server_name forge.example.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name forge.example.com;

    ssl_certificate     /etc/letsencrypt/live/forge.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/forge.example.com/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    add_header X-Content-Type-Options "nosniff" always;

    client_max_body_size 50m;   # training-config uploads

    location / {
        proxy_pass         http://llmforge_app;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_set_header   Connection        "";
        proxy_read_timeout 3600s;
    }

    # WebSocket — required for live charts + interactive shell
    location /api/ws {
        proxy_pass         http://llmforge_app;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade           $http_upgrade;
        proxy_set_header   Connection        "upgrade";
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 3600s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/forge /etc/nginx/sites-enabled/forge
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d forge.example.com    # one-shot if not already issued
```

After this, update `AUTH_URL` and `NEXTAUTH_URL` in `.env.production` to
`https://forge.example.com` and restart the UI.

---

## Environment variables reference

All variables live in `.env.local` (dev) or `.env.production` (prod). The
custom server (`server.ts`) uses `tsx --env-file=.env.local`, so `.env.local`
also covers dev runs. For systemd, point `EnvironmentFile=` at `.env.production`.

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | yes | `postgresql://USER:PASS@HOST:PORT/DB`. Append `?sslmode=require` in prod. |
| `AUTH_SECRET` | yes | Base64 string, ≥48 chars in prod. Used by Auth.js to sign JWTs / encrypt cookies. |
| `NEXTAUTH_SECRET` | yes | **Set to the SAME value as `AUTH_SECRET`.** Used by `lib/ws-manager.ts` to decode the cookie on the `/api/ws` upgrade. If they differ, the interactive shell silently disconnects with `Connection closed before shell opened`. |
| `AUTH_URL` | yes (prod) | Canonical app origin, e.g. `https://forge.example.com`. Used by Auth.js for redirects and cookie scoping. |
| `NEXTAUTH_URL` | yes | Same as `AUTH_URL`. Required by `lib/env.ts`. |
| `AUTH_TRUST_HOST` | recommended `"true"` | Allows Auth.js to accept the request host. Default is true in this UI. Set `"false"` only on a recognised platform (Vercel/Netlify/Cloudflare). |
| `SSH_KEY_ENCRYPTION_KEY` | yes | **Exactly 32 hex chars** (16 bytes) for AES-256-GCM. Generates SSH-key ciphertext at rest. Generate with `openssl rand -hex 16`. **Losing this destroys every stored SSH key.** |
| `REPO_ROOT` | yes | Path to the framework directory containing `train_pretrain.py`, `train_sft.py`, etc. Defaults to `../`. |
| `HOSTNAME` | optional | Bind address. Defaults to `localhost`. Use `0.0.0.0` in prod. |
| `PORT` | optional | Defaults to `3000`. |
| `NODE_ENV` | yes (prod) | `production` makes `server.ts` pass `dev: false` so the prebuilt `.next/` is used. |
| `SMTP_*` / `NOTIFICATION_EMAIL` | optional | Email alerts on job failures. |

---

## Database commands

```bash
# Generate Prisma client (re-run after schema.prisma changes)
npx prisma generate

# Dev: sync schema → DB without migration files
npx prisma db push

# Prod: apply SQL files in prisma/migrations/ in order (safe to re-run)
npx prisma migrate deploy

# GUI browser for your data
npx prisma studio

# Reset database (drops all data, re-syncs schema)
npx prisma migrate reset --force

# Drop and recreate the database
PGPASSWORD=postgres psql -U postgres -h localhost -c "DROP DATABASE llm_training_ui;"
PGPASSWORD=postgres psql -U postgres -h localhost -c "CREATE DATABASE llm_training_ui;"
npx prisma db push

# Verify connectivity
psql "$DATABASE_URL" -c "SELECT 1;"
```

---

## Upgrades and rollback

### Docker upgrade

```bash
cd /opt/llmforge-ui
sudo -u llmforge git pull --ff-only
cd UI
sudo -u llmforge npm install --include=dev
sudo -u llmorge npx prisma generate
sudo -u llmforge npx prisma migrate deploy           # BEFORE rolling the image
cd ..
sudo docker build -t llmforge-ui:1.1.0 .
sudo docker stop llmforge-ui && sudo docker rm llmforge-ui
sudo docker run -d --name llmforge-ui \
  --restart unless-stopped \
  --env-file .env.production \
  -p 127.0.0.1:3000:3000 \
  -v /mnt/framework:/framework:ro \
  -v /var/lib/llmforge/output:/output \
  llmforge-ui:1.1.0
```

### Bare-metal upgrade

```bash
cd /opt/llmforge-ui/UI
sudo -u llmforge git pull --ff-only
sudo -u llmforge npm install --include=dev
sudo -u llmforge npx prisma generate
sudo -u llmforge npx prisma migrate deploy
sudo -u llmforge npm run build
sudo systemctl restart llmforge-ui
```

### Rollback

Docker:

```bash
sudo docker stop llmforge-ui && sudo docker rm llmforge-ui
sudo docker run -d --name llmforge-ui \
  --restart unless-stopped \
  --env-file .env.production \
  -p 127.0.0.1:3000:3000 \
  -v /mnt/framework:/framework:ro \
  -v /var/lib/llmforge/output:/output \
  llmforge-ui:1.0.0     # previous tag
```

Bare-metal:

```bash
cd /opt/llmforge-ui
sudo -u llmforge git log --oneline -10              # find the last good commit
sudo -u llmforge git checkout <sha> -- UI
cd UI
sudo -u llmforge npm install --include=dev
sudo -u llmforge npm run build
sudo systemctl restart llmforge-ui
```

> **Database rollbacks are risky.** A forward migration that altered columns
> may not have a clean inverse. If you must roll back an app version that
> depended on a new column, restore the latest `pg_dump` into a fresh database
> and switch `DATABASE_URL` instead.

---

## Monitoring

Minimum viable monitoring in production:

```bash
# Health probe — alert on non-200 or db != "ok"
curl -fsS https://forge.example.com/api/system/health | jq -e '.status=="ok" and .db=="ok"'

# Process
systemctl status llmforge-ui       # or: docker ps --filter name=llmforge-ui
journalctl -u llmforge-ui -f       # or: docker logs -f llmforge-ui

# Disk — alert when /var/lib/llmforge/output is >85% full
df -h /var/lib/llmforge/output
```

Sample Prometheus blackbox probe:

```yaml
- name: llmforge_ui_health
  interval: 30s
  prober: http
  url: https://forge.example.com/api/system/health
  valid_status_codes: [200]
  fail_if_body_not_matches_regexp: ['"status":"ok"']
```

---

## Troubleshooting

### `UntrustedHost: Host must be trusted` from `/api/auth/*`

```bash
# .env.local — must be "true" (or unset; the UI defaults to true)
AUTH_TRUST_HOST="true"
```

If you're serving from a non-localhost host, also make sure your reverse proxy
forwards the original `Host` header and set `AUTH_TRUST_HOST="false"` only on
recognised platforms.

### PostgreSQL connection refused

```bash
sudo systemctl status postgresql
ss -tlnp | grep 5432
sudo grep -E "^(local|host)" $(sudo -u postgres psql -c "SHOW hba_file;" -t | tr -d ' ')
```

### `Cannot find module '@tailwindcss/postcss'` (or other missing deps after `npm install`)

```bash
rm -rf node_modules package-lock.json
npm install --include=dev    # --include=dev forces devDeps
npx prisma generate
```

### Prisma migration fails

```bash
# Dev shortcut — sync schema without writing migration files
npx prisma db push
# Nuclear reset
npx prisma migrate reset --force
```

### WebSocket not connecting

- **`next start` doesn't support WebSockets.** Use `npm run start:ws` (or the
  custom `server.ts` directly).
- If you're behind a reverse proxy, make sure `/api/ws` upgrades `Connection`
  to `upgrade` and forwards `Upgrade`. See the nginx config above.

### `Connection closed before shell opened` in the interactive terminal

`AUTH_SECRET` and `NEXTAUTH_SECRET` differ. Set them to the **exact same
value**, clear cookies for the host in your browser, and restart the UI.

### Lock down public sign-up

`/signup` is open by default (first user is created there). To make accounts
invite-only, edit `app/api/auth/register/route.ts` to require an
`INVITE_TOKEN` header and read it from `.env.production`. The proxy
(`proxy.ts`) already protects everything except `/signup` and `/login`.

### SSH connection fails

```bash
ssh -p 22 user@remote-host "echo OK"
ssh-keygen -R remote-host                  # clear cached host key
chmod 600 ~/.ssh/id_ed25519
```

### Job fails immediately

1. Check the **Jobs** page for the error message.
2. Verify `REPO_ROOT` points to the directory containing the Python scripts.
3. Install Python deps: `pip install -r ../requirements.txt`.
4. Try running the equivalent CLI command manually.

---

## Useful npm scripts

```bash
npm run dev          # Next.js dev server (lazy compilation)
npm run build        # Precompile every route into .next/
npm run start:ws     # Production server with WebSocket support
npm run start        # Plain next start — does NOT support WebSockets
npm run lint         # ESLint
npm run type-check   # tsc --noEmit
npm run db:generate  # prisma generate
npm run db:push      # prisma db push
npm run db:migrate   # prisma migrate dev
npm run db:studio    # prisma studio
npm test             # vitest run
```