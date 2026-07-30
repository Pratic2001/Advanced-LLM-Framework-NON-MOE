# LLMForge — Web UI

> **A browser-based control center for the Dense LLM Framework.** Launch, monitor, and orchestrate your entire LLM training pipeline — tokenizer training → data packing → pretrain → SFT → GRPO/DPO — across single-node, multi-node, and decentralized (Hivemind) backends, all from a modern web interface.

<p align="center">
  <img src="https://img.shields.io/badge/status-stable-brightgreen" alt="Status"/>
  <img src="https://img.shields.io/badge/next.js-14-black" alt="Next.js 14"/>
  <img src="https://img.shields.io/badge/postgresql-16%2B-blue" alt="PostgreSQL 16+"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"/>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [PostgreSQL Setup](#postgresql-setup)
  - [Ubuntu / Debian](#ubuntu--debian)
  - [Fedora / RHEL](#fedora--rhel)
  - [Arch Linux](#arch-linux)
- [Installation](#installation)
- [Configuration](#configuration)
- [Production Deployment](#production-deployment)
  - [Pre-Deployment Checklist](#pre-deployment-checklist)
  - [Option A: Docker (Recommended)](#option-a-docker-recommended)
  - [Option B: Bare-Metal / VM](#option-b-bare-metal--vm)
  - [Option C: Reverse Proxy + TLS](#option-c-reverse-proxy--tls)
  - [Post-Deployment](#post-deployment)
  - [Upgrades](#upgrades)
  - [Rollback](#rollback)
  - [Monitoring](#monitoring)
- [Running the UI](#running-the-ui)
- [Usage Scenarios](#usage-scenarios)
  - [Scenario A: Single Node (Local Machine)](#scenario-a-single-node-local-machine)
  - [Scenario B: Head Node + Worker Node (Separate Servers)](#scenario-b-head-node--worker-node-separate-servers)
  - [Scenario C: Multi-Node Cluster (3+ Machines)](#scenario-c-multi-node-cluster-3-machines)
  - [Scenario D: Decentralized Hivemind Training](#scenario-d-decentralized-hivemind-training)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)

---

## Overview

The LLMForge UI wraps the entire CLI-based [Dense LLM Framework](https://github.com/your-repo) into a graphical interface with:

- **3 Backend Tabs** — Torch DDP, DeepSpeed (ZeRO), Hivemind (decentralized)
- **Setup Wizards** — Passwordless SSH, NFS, Python environment setup with copyable commands
- **Node Management** — Add, audit, and manage remote training nodes
- **Flag Configuration** — Typed form fields for every CLI flag across all 6 training scripts
- **Pipeline Orchestration** — One-click launch of multi-stage pipelines
- **Live Monitoring** — Real-time logs, loss curves, VRAM usage, throughput charts via WebSocket
- **Config Presets** — Save and reuse training configurations

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Node.js | 18+ (20+ recommended) | Needed to run the Next.js dev server |
| npm | 9+ | Comes with Node.js |
| PostgreSQL | 14+ (16 recommended) | Database for users, jobs, nodes, metrics |
| Python | 3.10+ | The training framework itself (outside UI) |
| Git | any | For the repo |

Check your versions:

```bash
node --version   # should be ≥18
npm --version    # should be ≥9
psql --version   # should be ≥14
python3 --version
```

---

## PostgreSQL Setup

The UI uses PostgreSQL via Prisma ORM. Choose your OS below.

### Ubuntu / Debian

```bash
# 1. Install PostgreSQL
sudo apt update
sudo apt install -y postgresql postgresql-contrib

# 2. Start and enable the service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# 3. Create a database and user
sudo -u postgres psql -c "CREATE USER postgres WITH PASSWORD 'postgres';"
sudo -u postgres psql -c "ALTER USER postgres WITH SUPERUSER;"
sudo -u postgres psql -c "CREATE DATABASE llm_training_ui OWNER postgres;"

# 4. Verify
psql -h localhost -U postgres -d llm_training_ui -c "SELECT 1;"
# (enter password: postgres)

# 5. (Optional) Allow password auth — edit pg_hba.conf
# Find the file:
# sudo -u postgres psql -c "SHOW hba_file;"
# Then change "peer" to "md5" for local lines, then restart:
# sudo systemctl restart postgresql
```

> **Security note:** Change the password in production. The defaults above are for local development only.

### Fedora / RHEL

```bash
# 1. Install PostgreSQL
sudo dnf install -y postgresql-server postgresql-contrib

# 2. Initialize and start
sudo postgresql-setup --initdb
sudo systemctl start postgresql
sudo systemctl enable postgresql

# 3. Create database and user
sudo -u postgres psql -c "CREATE USER postgres WITH PASSWORD 'postgres';"
sudo -u postgres psql -c "ALTER USER postgres WITH SUPERUSER;"
sudo -u postgres psql -c "CREATE DATABASE llm_training_ui OWNER postgres;"

# 4. On Fedora, pg_hba.conf uses "peer" by default — switch to "md5"
# Edit /var/lib/pgsql/data/pg_hba.conf and change:
#   local   all   all   peer  →  local   all   all   md5
#   host    all   all   ident →  host    all   all   md5
sudo systemctl restart postgresql

# 5. Verify
psql -h localhost -U postgres -d llm_training_ui -c "SELECT 1;"
```

### Arch Linux

```bash
# 1. Install PostgreSQL
sudo pacman -S postgresql

# 2. Initialize the database cluster
sudo -u postgres initdb -D /var/lib/postgres/data

# 3. Start and enable
sudo systemctl start postgresql
sudo systemctl enable postgresql

# 4. Create database and user
sudo -u postgres psql -c "CREATE USER postgres WITH PASSWORD 'postgres';"
sudo -u postgres psql -c "ALTER USER postgres WITH SUPERUSER;"
sudo -u postgres psql -c "CREATE DATABASE llm_training_ui OWNER postgres;"

# 5. Edit pg_hba.conf to allow md5 auth
# Usually at /var/lib/postgres/data/pg_hba.conf
# Change "peer" to "md5" for local lines, then:
sudo systemctl restart postgresql

# 6. Verify
psql -h localhost -U postgres -d llm_training_ui -c "SELECT 1;"
```

---

## Installation

```bash
# 1. Navigate to the UI directory
cd UI

# 2. Install dependencies
npm install --legacy-peer-deps

# 3. Set up environment variables
cp .env.example .env.local
# Then edit .env.local to match your Postgres credentials (see below)

# 4. Generate Prisma client and push schema to database
npx prisma generate
npx prisma db push

# 5. Verify the database tables were created
psql -h localhost -U postgres -d llm_training_ui -c "\dt"
# You should see: ConfigPreset, Job, JobMetric, Node, Pipeline,
#                 PipelineStage, SshKey, SystemEvent, User
```

> **Note:** No seed script is configured. After `db push`, register your first user at `/signup` in the UI, which will create the account via the `/api/auth/register` endpoint.

---

## Database Quick Reference

All the exact commands for setting up and managing the database:

### PostgreSQL setup (Ubuntu)

```bash
# Install
sudo apt update && sudo apt install -y postgresql postgresql-contrib
sudo systemctl start postgresql && sudo systemctl enable postgresql

# Create user and database
sudo -u postgres psql -c "CREATE USER postgres WITH PASSWORD 'postgres';"
sudo -u postgres psql -c "ALTER USER postgres WITH SUPERUSER;"
sudo -u postgres psql -c "CREATE DATABASE llm_training_ui OWNER postgres;"

# Allow password auth (edit pg_hba.conf — change "peer" to "md5")
sudo sed -i 's/peer/md5/g' $(sudo -u postgres psql -c "SHOW hba_file;" -t | tr -d ' ')
sudo systemctl restart postgresql

# Verify
PGPASSWORD=postgres psql -U postgres -h localhost -d llm_training_ui -c "SELECT 1;"
```

### Change the Postgres password

```bash
# Inside psql
PGPASSWORD=postgres psql -U postgres -h localhost -c "ALTER USER postgres PASSWORD 'newpassword';"

# Then update DATABASE_URL in .env.local:
#   DATABASE_URL="postgresql://postgres:newpassword@localhost:5432/llm_training_ui"
```

### Prisma commands

```bash
# Generate Prisma client (re-run after schema changes)
npx prisma generate

# Push schema to database (creates/updates tables, no migration files)
npx prisma db push

# Use migrations instead (for tracking schema changes in version control)
npx prisma migrate dev --name init

# Open Prisma Studio (GUI browser for your data)
npx prisma studio

# Reset database (drops all data and re-syncs schema)
npx prisma migrate reset --force
```

### Drop and recreate the database

```bash
PGPASSWORD=postgres psql -U postgres -h localhost -c "DROP DATABASE llm_training_ui;"
PGPASSWORD=postgres psql -U postgres -h localhost -c "CREATE DATABASE llm_training_ui;"
npx prisma db push
```

### Verify the connection string

```bash
# The DATABASE_URL format is:
#   postgresql://USER:PASSWORD@HOST:PORT/DATABASE
# Default dev values:
#   postgresql://postgres:postgres@localhost:5432/llm_training_ui

# Test it directly:
PGPASSWORD=postgres psql -U postgres -h localhost -d llm_training_ui -c "\dt"
```

---

## Configuration

Edit `.env.local` in the `UI/` directory:

```env
# ─── Database ─────────────────────────────────────
# Use the connection string for your PostgreSQL setup
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/llm_training_ui"

# ─── Auth.js (NextAuth v5) ─────────────────────────
# AUTH_SECRET signs JWTs and encrypts cookies — required.
# Generate with: openssl rand -base64 32
AUTH_SECRET="change-this-to-a-random-secret-in-production"

# AUTH_URL sets the canonical app origin. Optional — leave unset and Auth.js
# will infer it from the request when trustHost is enabled.
# AUTH_URL="http://localhost:3000"

# AUTH_TRUST_HOST must be true when running on localhost or any host that
# isn't a known platform (Vercel/Netlify/etc.). Without it, /api/auth/* throws
# `UntrustedHost: Host must be trusted`. Defaults to true in this UI.
AUTH_TRUST_HOST="true"

# ─── SSH Key Encryption ────────────────────────────
# 32 hex chars for AES-256-GCM (used to encrypt SSH keys at rest)
SSH_KEY_ENCRYPTION_KEY="0123456789abcdef0123456789abcdef"

# ─── Repo Root ────────────────────────────────────
# Absolute or relative path to the training framework root
# (where train_pretrain.py, train_sft.py, etc. live)
REPO_ROOT="../"
```

---

## Production Deployment

A production deployment serves the UI over HTTPS on a public or LAN host, with PostgreSQL on a separate (or the same) machine, runs `npm run start` (or the Docker image) under a process manager, and is fronted by a reverse proxy for TLS termination and WebSocket pass-through. Pick one of the three options below based on your environment.

### Pre-Deployment Checklist

Tick every box before you ship.

- [ ] **Strong `AUTH_SECRET`.** Generate with `openssl rand -base64 48` (or `openssl rand -hex 32`). Never reuse the dev secret.
- [ ] **Strong `SSH_KEY_ENCRYPTION_KEY`.** Exactly 32 hex chars (16 bytes) for AES-256-GCM. Generate with `openssl rand -hex 16`. **Losing this key destroys all stored SSH keys.**
- [ ] **Strong Postgres password** and a dedicated user (not `postgres` if avoidable).
- [ ] **`AUTH_URL`** set to the canonical public origin, e.g. `https://forge.example.com`. Required when serving behind a reverse proxy; optional if `AUTH_TRUST_HOST` is true and Auth.js can infer the host.
- [ ] **`DATABASE_URL`** uses the production host and TLS mode (`?sslmode=require`).
- [ ] **`NODE_ENV=production`** (the Dockerfile and `next start` both set this).
- [ ] **HTTPS is terminated upstream** (nginx / Caddy / a managed LB). Cookies are `Secure` in production.
- [ ] **Reverse proxy forwards the original `Host` and `X-Forwarded-*` headers** so Auth.js trusts the request and middleware can build correct redirect URLs.
- [ ] **Backups** of the Postgres database — at minimum a daily `pg_dump` with offsite copy. Test restore quarterly.
- [ ] **Firewall** opens `3000/tcp` (UI) only to the proxy host; Postgres `5432/tcp` only allows the UI host (or is socket-only).
- [ ] **First user** will be created via `/signup`. To make sign-up private, see the *Disable public sign-up* note below.

**Generate the secrets once and store them somewhere safe before building:**

```bash
echo "AUTH_SECRET=$(openssl rand -base64 48)"
echo "SSH_KEY_ENCRYPTION_KEY=$(openssl rand -hex 16)"
```

### Option A: Docker (Recommended)

The included `Dockerfile` is a 3-stage Node 22-alpine build that produces a [Next.js standalone](https://nextjs.org/docs/app/api-reference/config/next-config-js/output#standalone) image (~150 MB). It runs as a non-root user (`nextjs`, uid 1001) and listens on port 3000.

#### 1. Prepare a deployment directory

```bash
mkdir -p /opt/llmforge-ui && cd /opt/llmforge-ui
# Copy in: the project source (or a built tarball) so the Dockerfile can find it.
# You only need the Dockerfile, package.json, package-lock.json, app/, lib/,
# hooks/, components/, public/, prisma/, next.config.mjs, tsconfig.json,
# server.ts, instrumentation.ts, middleware.ts.
```

#### 2. Build the image

```bash
docker build -t llmforge-ui:1.0.0 .
# Tagged :1.0.0 so you can pin to a known-good image during rollback.
```

#### 3. Provide secrets at runtime, not in the image

Create `.env.production` on the host (NOT inside the image):

```bash
cat > /opt/llmforge-ui/.env.production <<'EOF'
NODE_ENV=production
PORT=3000
HOSTNAME=0.0.0.0
AUTH_TRUST_HOST=true
AUTH_URL=https://forge.example.com

# 48+ chars — generated above
AUTH_SECRET=replace-with-the-secret-you-generated

# 32 hex chars — generated above
SSH_KEY_ENCRYPTION_KEY=replace-with-the-key-you-generated

DATABASE_URL=postgresql://llmforge:STRONG_PASSWORD@db.internal:5432/llmforge?sslmode=require

# Absolute path inside the container to where the framework is mounted.
# This is a path *inside* the container; mount the framework volume below.
REPO_ROOT=/framework
EOF
chmod 600 /opt/llmforge-ui/.env.production
```

#### 4. Start the container

```bash
docker run -d \
  --name llmforge-ui \
  --restart unless-stopped \
  --env-file /opt/llmforge-ui/.env.production \
  -p 127.0.0.1:3000:3000 \
  -v /mnt/framework:/framework:ro \   # training scripts + data (read-only)
  -v /var/lib/llmforge/output:/output \  # training checkpoints
  llmforge-ui:1.0.0
```

Notes:
- `--restart unless-stopped` brings the container back after a host reboot.
- Bind `127.0.0.1:3000` (loopback only) — the reverse proxy on the same host connects to `127.0.0.1:3000` and the UI is **not** directly reachable from the internet.
- Mount the training framework and the output directory so jobs survive container restarts.
- **Migrations:** the image does not run `prisma migrate deploy` automatically. Before the first start:

  ```bash
  docker run --rm \
    --env-file /opt/llmforge-ui/.env.production \
    llmforge-ui:1.0.0 \
    npx prisma migrate deploy
  ```

  Then start the main container as above.

#### 5. Docker Compose (alternative)

```yaml
# /opt/llmforge-ui/docker-compose.yml
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
    depends_on:
      - db
    # healthcheck is built into the image at /api/system/health

  db:
    image: postgres:16-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB: llmforge
      POSTGRES_USER: llmforge
      POSTGRES_PASSWORD: STRONG_PASSWORD
    volumes:
      - pgdata:/var/lib/postgresql/data
    # Expose ONLY to the UI service — see notes below.

volumes:
  pgdata:
```

Run migrations once after first deploy:

```bash
docker compose run --rm ui npx prisma migrate deploy
docker compose up -d
```

#### 6. Verify

```bash
curl -s http://127.0.0.1:3000/api/system/health | jq
# Should return { "status": "ok", "db": "ok", ... }
```

### Option B: Bare-Metal / VM

Use this when you don't want Docker (e.g., on a single GPU workstation where you also want to run training jobs directly, without bind-mounting).

#### 1. Provision

- Linux (Ubuntu 22.04+ / Debian 12+ / RHEL 9+) — Windows is not supported.
- Node.js 20 LTS via [`nvm`](https://github.com/nvm-sh/nvm), [fnm](https://github.com/Schniz/fnm), or NodeSource.
- PostgreSQL 16+ via your distro's package manager (see PostgreSQL Setup above).
- A dedicated `llmforge` system user:

  ```bash
  sudo useradd -r -m -d /var/lib/llmforge -s /bin/bash llmforge
  sudo mkdir -p /opt/llmforge-ui /var/lib/llmforge/output
  sudo chown -R llmforge:llmforge /opt/llmforge-ui /var/lib/llmforge/output
  ```

#### 2. Install

```bash
sudo -u llmforge -H bash <<'EOF'
cd /opt/llmforge-ui
git clone https://github.com/your-repo/Advanced-LLM-Framework-NON-MOE.git .
cd UI
npm ci --legacy-peer-deps
npx prisma generate
npm run build
EOF
```

#### 3. Configure

Write `/opt/llmforge-ui/UI/.env.production` (see the file above for contents). Make sure permissions are tight:

```bash
sudo chown llmforge:llmforge /opt/llmforge-ui/UI/.env.production
sudo chmod 600 /opt/llmforge-ui/UI/.env.production
```

Run migrations:

```bash
sudo -u llmforge -H bash -c 'cd /opt/llmforge-ui/UI && npx prisma migrate deploy'
```

> **Use `migrate deploy` in production, not `db push`.** `migrate deploy` applies SQL files from `prisma/migrations/` in order — it's safe to re-run on each deploy.

#### 4. Run under a process manager (systemd)

`/etc/systemd/system/llmforge-ui.service`:

```ini
[Unit]
Description=LLMForge UI (Next.js)
After=network-online.target postgresql.service
Wants=network-online.target

[Service]
Type=simple
User=llmforge
Group=llmforge
WorkingDirectory=/opt/llmforge-ui/UI
EnvironmentFile=/opt/llmforge-ui/UI/.env.production
Environment=NODE_ENV=production
Environment=PORT=3000
ExecStart=/usr/bin/node node_modules/next/dist/bin/next start
Restart=on-failure
RestartSec=5
# Hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/llmforge/output /var/lib/llmforge/cache
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

> If you need WebSocket support for the live charts page, swap `ExecStart` to run the custom server: `ExecStart=/usr/bin/node server.js` (after `next build`).

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llmforge-ui
sudo systemctl status llmforge-ui
curl -s http://127.0.0.1:3000/api/system/health | jq
```

### Option C: Reverse Proxy + TLS

A reverse proxy gives you TLS, rate-limiting, and the `Host` / `X-Forwarded-Proto` headers Auth.js needs. Bind the UI only to `127.0.0.1:3000` (above) so nothing in this section is optional.

The middleware already sets `Strict-Transport-Security` in production. Pick whichever proxy you prefer:

#### Caddy (automatic HTTPS via Let's Encrypt / ZeroSSL)

`/etc/caddy/Caddyfile`:

```caddyfile
forge.example.com {
    encode zstd gzip
    reverse_proxy 127.0.0.1:3000 {
        # WebSocket upgrade for /api/ws and HMR
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

#### nginx

`/etc/nginx/sites-available/forge`:

```nginx
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

    # Security headers (the app sets some of these; nginx adds the rest)
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

    # WebSocket — required for the Jobs live chart
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
```

Certbot one-shot for the cert above:

```bash
sudo certbot --nginx -d forge.example.com
```

### Post-Deployment

1. **Smoke test the auth flow.** Open the URL, sign up, sign out, sign back in. Confirm `/api/system/health` returns `db: ok` and that the **Jobs** page loads (charts need the WebSocket).
2. **Verify the cookie name.** In DevTools → Application → Cookies, look for `authjs.session-token` (HTTPS) or `__Secure-authjs.session-token`. If you only see a `next-auth.session-token`, the proxy is stripping cookies.
3. **Backups.** Schedule a daily backup of the Postgres database:
   ```bash
   # /etc/cron.d/llmforge-backup
   0 3 * * * pg_dump -h db.internal -U llmforge llmforge | \
     gzip > /var/backups/llmforge/$(date -u +\%Y\%m\%dT\%H\%M\%SZ).sql.gz
   ```
4. **Disable public sign-up** if you want accounts to be invite-only. Edit `app/api/auth/register/route.ts` to require an `INVITE_TOKEN` header and read it from `.env.production`. The middleware already protects everything except `/signup` itself.

### Upgrades

```bash
# 1. Pull the new source
cd /opt/llmforge-ui
sudo -u llmforge git pull --ff-only

# 2. Install + build
cd UI
sudo -u llmforge npm ci --legacy-peer-deps
sudo -u llmforge npx prisma generate

# 3. Apply migrations BEFORE rolling the new image
sudo -u llmforge npx prisma migrate deploy

# 4. Rebuild the image and roll
cd ..
docker build -t llmforge-ui:1.1.0 .
docker stop llmforge-ui && docker rm llmforge-ui
docker run -d --name llmforge-ui \
  --restart unless-stopped \
  --env-file .env.production \
  -p 127.0.0.1:3000:3000 \
  -v /mnt/framework:/framework:ro \
  -v /var/lib/llmforge/output:/output \
  llmforge-ui:1.1.0
```

For the bare-metal variant, swap `docker build` for `npm run build`, then:

```bash
sudo systemctl restart llmforge-ui
```

### Rollback

Container: redeploy the previous tag.

```bash
docker stop llmforge-ui && docker rm llmforge-ui
docker run -d --name llmforge-ui \
  --restart unless-stopped \
  --env-file .env.production \
  -p 127.0.0.1:3000:3000 \
  -v /mnt/framework:/framework:ro \
  -v /var/lib/llmforge/output:/output \
  llmforge-ui:1.0.0   # <-- previous image
```

Bare-metal: check out the previous commit and restart.

```bash
cd /opt/llmforge-ui
sudo -u llmforge git log --oneline -10   # find the last good commit
sudo -u llmforge git checkout <sha> -- UI
cd UI && sudo -u llmforge npm ci --legacy-peer-deps && sudo -u llmforge npm run build
sudo systemctl restart llmforge-ui
```

> **Database rollbacks** are risky — a forward migration that altered columns may not have a clean inverse. If you must roll back an app version that depended on a new column, restore the latest `pg_dump` into a fresh database and switch `DATABASE_URL` instead.

### Monitoring

Minimum viable monitoring in production:

- **Health endpoint.** Hit `https://forge.example.com/api/system/health` every 30 s with an external probe (curl + jq, [Blackbox exporter](https://github.com/prometheus/blackbox_exporter), [UptimeRobot](https://uptimerobot.com)). Alert on non-200 or `db != "ok"`.
- **Process check.** `systemctl status llmforge-ui` or `docker ps` + a restart-count alert.
- **Disk.** Alert when the disk hosting `/var/lib/llmforge/output` is >85% full — a runaway training job can fill it overnight.
- **Logs.** Capture both app and reverse-proxy logs to a centralised sink (Loki, CloudWatch, syslog-ng). `journalctl -u llmforge-ui -f` or `docker logs -f llmforge-ui` for ad-hoc.
- **Postgres metrics.** `pg_stat_activity`, replica lag if you set one up.

A Prometheus blackbox-probe snippet:

```yaml
- name: llmforge_ui_health
  interval: 30s
  prober: http
  url: https://forge.example.com/api/system/health
  valid_status_codes: [200]
  fail_if_body_not_matches_regexp: ['"status":"ok"']
```

---

## Running the UI

### Development mode

```bash
cd UI
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### Production build

```bash
npm run build
npm start
```

### With WebSocket support (for live charts)

```bash
npx tsx server.ts
# or if built:
node server.js
```

This starts a custom HTTP server that also handles WebSocket upgrades at `/api/ws`.

---

## Custom CLI Arguments

Every backend config page (Torch/DDP, DeepSpeed, Hivemind) now has an **Extra CLI Arguments** textarea that lets you pass arbitrary CLI flags verbatim. This is useful for flags that aren't covered by the structured form fields, such as `--gradient_checkpointing`, `--use_flash_attn_2`, or custom `--optimizer` settings.

### Torch / DDP and DeepSpeed

1. Go to the backend tab → **Configure**.
2. Fill in the structured form fields (Model, Training, etc.) as usual.
3. Scroll to the **Extra CLI Arguments** section.
4. Type any additional flags, one per line or space-separated:
   ```
   --gradient_checkpointing
   --use_flash_attn_2
   --optimizer adamw
   ```
5. Click **Run Training** — your custom args are appended verbatim after the generated flags.

### Hivemind (Per-Node CLI Arguments)

Hivemind training runs the same script on every peer, but each peer often needs different networking arguments (e.g. the bootstrap peer doesn't use `--bootstrap_peer`, while worker peers do). The Hivemind config page supports **per-peer CLI arguments** in addition to global extra args.

1. Go to **Hivemind** tab → **Configure**.
2. Fill in shared training flags as usual.
3. In the **Global Extra CLI Arguments** section, add flags that apply to **all** peers:
   ```
   --compression float16
   --max_peers 32
   ```
4. In the **Peer Configuration** section, add each peer with its **peer-specific arguments**:
   - **Bootstrap peer:** `--host_maddrs /ip4/0.0.0.0/tcp/31337`
   - **Worker peers:** `--bootstrap_peer /ip4/<bootstrap-ip>/tcp/31337/p2p/<peer-id> --announce_maddrs /ip4/<peer-ip>/tcp/31337`
5. When you click **Run Hivemind Training**, the UI creates **one job per peer** — each with the shared config, global extra args, and its own peer-specific args combined.

### Equivalent CLI commands

The UI generates these commands internally (torch example):

```bash
# Without extra args:
torchrun --nproc_per_node=1 --nnodes=1 train_pretrain.py \
  --model_type dense --batch_size 8 --learning_rate 3e-4 ...

# With extra args:
torchrun --nproc_per_node=1 --nnodes=1 train_pretrain.py \
  --model_type dense --batch_size 8 --learning_rate 3e-4 ... \
  --gradient_checkpointing --use_flash_attn_2
```

The extra args are appended after all generated flags, so they can override defaults if the training script reads them left-to-right.

---

## Usage Scenarios

### Scenario A: Single Node (Local Machine)

**One machine, one GPU (or multiple).** You run everything locally — the UI launches training as subprocesses on the same machine.

```
┌─────────────────────────────────────────┐
│         Your Machine                    │
│  ┌─────────┐  ┌──────────────────────┐ │
│  │ Browser │  │  UI (Next.js)        │ │
│  │ :3000   │──│  spawns subprocesses │ │
│  └─────────┘  │  train_pretrain.py   │ │
│               │  train_sft.py        │ │
│               └──────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │  PostgreSQL (localhost:5432)       │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

**Setup steps:**

1. Complete the PostgreSQL setup above.
2. Start the UI: `cd UI && npm run dev`
3. Register an account at `/signup`.
4. Go to the **Torch / DDP** tab → **Setup** tab and follow the environment setup commands.
5. Skip **Nodes** (no remote nodes needed).
6. Go to **Configure** tab, fill in flags, click **Run Pipeline**.

**What happens:** The UI spawns `torchrun --nproc_per_node=<gpu_count> --nnodes=1 train_pretrain.py ...` as a subprocess. Logs and metrics stream to the Jobs page in real time via the process pool in `lib/job-manager.ts`.

```bash
# Equivalent CLI command the UI generates internally:
torchrun --nproc_per_node=1 --nnodes=1 train_pretrain.py \
  --model_type dense \
  --vocab_size 100352 \
  --hidden_dim 768 \
  --num_layers 12 \
  --batch_size 8 \
  --learning_rate 3e-4 \
  --data_path /mnt/training/data \
  --output_dir /mnt/training/output
```

---

### Scenario B: Head Node + Worker Node (Separate Servers)

**Two machines:** the UI runs on the head node (which has PostgreSQL), and training is distributed across both the head and a worker via SSH.

```
┌──────────────────────────┐    SSH     ┌──────────────────┐
│   Head Node (192.168.1.10) │◄──────────┤  Worker Node     │
│  ┌────────┐  ┌─────────┐ │   port 22  │  (192.168.1.20)  │
│  │ Browser│  │ UI      │ │            │  ┌──────────────┐│
│  │ :3000  │──│ spawns  ├─┼────────────┼──┤ GPU 0..N     ││
│  └────────┘  │ via SSH │ │            │  └──────────────┘│
│              └─────────┘ │            │  nfs mounted:    │
│  ┌──────────────────────┐│            │  /mnt/training   │
│  │ PostgreSQL           ││            └──────────────────┘
│  └──────────────────────┘│
│  ┌──────────────────────┐│
│  │ NFS Server           ││
│  │ /srv/nfs/training    ││
│  └──────────────────────┘│
└──────────────────────────┘
```

**Setup steps:**

**On the head node (the machine running the UI):**

```bash
# 1. Set up NFS server
sudo apt install -y nfs-kernel-server
sudo mkdir -p /srv/nfs/training
sudo chown -R $(whoami):$(whoami) /srv/nfs/training
echo "/srv/nfs/training 192.168.1.0/24(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports
sudo exportfs -ra

# 2. Generate an SSH key (passwordless)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
ssh-copy-id user@192.168.1.20   # copy to worker

# 3. Verify
ssh user@192.168.1.20 "echo OK"
```

**On each worker node:**

```bash
# 1. Install NFS client
sudo apt install -y nfs-common
sudo mkdir -p /mnt/training
sudo mount -t nfs 192.168.1.10:/srv/nfs/training /mnt/training
echo "192.168.1.10:/srv/nfs/training /mnt/training nfs rw,defaults 0 0" | sudo tee -a /etc/fstab

# 2. Verify
ls /mnt/training   # should show head node's files

# 3. Ensure Python environment is the same as head node
# (clone the repo, install requirements)
```

**In the UI:**

1. Log in and go to any backend tab → **Nodes**.
2. Click **Add Node**, enter:
   - Name: `worker-1`
   - Host: `192.168.1.20`
   - Port: `22`
   - Username: `user`
   - Role: `Worker`
3. The UI automatically audits the node (GPU count, VRAM, CPU cores, RAM).
4. Click **Mount NFS** — the UI runs the mount command via SSH.
5. Go to **Configure**, fill in training flags, click **Run Pipeline**.

The UI builds the following commands internally:

**Torch DDP multi-node:**
```bash
torchrun --nproc_per_node=1 --nnodes=2 \
  --rdzv_endpoint=192.168.1.10:29500 --rdzv_backend=c10d \
  train_pretrain.py \
  --batch_size 8 --learning_rate 3e-4 ...
```

**DeepSpeed multi-node:**
```bash
deepspeed --num_gpus 1 --num_nodes 2 --hostfile /mnt/training/hostfile \
  train_pretrain.py \
  --batch_size 8 --learning_rate 3e-4 --zero_stage 2 ...
```

---

### Scenario C: Multi-Node Cluster (3+ Machines)

**3+ machines:** one head node (UI + NFS server) and N worker nodes. All GPUs contribute to training.

```
          ┌──────────────────┐
          │  Head Node       │
          │  UI + NFS + DB   │
          │  192.168.1.10    │
          └────────┬─────────┘
                   │ SSH
     ┌─────────────┼─────────────┐
     │             │             │
┌────▼────┐  ┌────▼────┐  ┌────▼────┐
│Worker 1 │  │Worker 2 │  │Worker 3 │
│1.20     │  │1.30     │  │1.40     │
│4x A100  │  │8x H100  │  │4x A100  │
└─────────┘  └─────────┘  └─────────┘
```

**Setup steps (same as Scenario B, repeated for each worker):**

1. Set up NFS server on the head node (export to the whole subnet).
2. Generate SSH key on the head node, copy to all workers.
3. Install NFS client and mount on every worker.
4. Clone the repo and install Python deps on every worker.
5. In the UI **Nodes** tab, add each worker.

**Torch DDP** automatically discovers all nodes via the `rdzv_endpoint` rendezvous. **DeepSpeed** uses a generated hostfile. **Hivemind** uses a bootstrap peer.

The `--hostfile` for DeepSpeed is auto-generated as:

```
192.168.1.20 slots=4
192.168.1.30 slots=8
192.168.1.40 slots=4
```

You can also use different GPU counts per node — the framework handles heterogeneous setups (automatically detected during the audit step).

---

### Scenario D: Decentralized Hivemind Training

Hivemind uses a **peer-to-peer** architecture instead of master-worker. There is no head node — every peer is autonomous and contributes gradients that are averaged via distributed all-reduce.

```
┌──────────┐     ┌──────────┐
│ Peer A   │◄────►│ Peer B   │
│ GPU 4090 │ DHT  │ GPU A100 │
└────┬─────┘     └────┬─────┘
     │                │
     │   ┌──────────┐ │
     └──►│ Peer C   │◄┘
         │ GPU 3090 │
         └──────────┘
```

**Setup (one bootstrap peer + any number of worker peers):**

1. On the bootstrap peer (designated first node):
   - Open ports 31337-31339 in the firewall.
   - Start the UI, add this node as **Bootstrap Peer**.
   - The UI shows the bootstrap multi-address, e.g.:
     `/ip4/192.168.1.10/tcp/31337/p2p/12D3KooW...`

2. On each worker peer:
   - Open the same ports.
   - Add the node as a **Worker Peer**, providing the bootstrap multi-address.

3. **Configure** tab includes Hivemind-specific flags:
   - `bootstrap_peer` — the bootstrap multi-address
   - `compression` — gradient compression method (FP16, 8-bit)
   - `target_batch_size` — total batch across all peers
   - `max_peers` — max nodes in the swarm

4. Launch training — each peer runs independently and syncs via the DHT.

```bash
# Equivalent CLI on each peer (the UI constructs this):
python3 train_pretrain.py \
  --bootstrap_peer /ip4/192.168.1.10/tcp/31337/p2p/12D3KooW... \
  --host_maddrs /ip4/0.0.0.0/tcp/31337 \
  --compression float16 \
  --target_batch_size 2048 \
  --batch_size_per_peer 4 \
  --data_path /mnt/training/data
```

---

## Pipeline Walkthrough

The UI can orchestrate a full **multi-stage pipeline** with a single click:

```
tokenizer_train.py  →  hf_to_packed.py  →  train_pretrain.py  →  train_sft.py  →  train_grpo.py or train_dpo.py
      │                      │                    │                   │                    │
      │  trains a BPE        │  tokenizes +       │  pre-trains a     │  fine-tunes        │  aligns with
      │  tokenizer from      │  packs raw text    │  foundation       │  on instructions   │  preferences
      │  scratch             │  into .pt files    │  LLM              │  (chat)            │  (RLHF)
```

1. Go to any backend tab → **Configure**.
2. Select the pipeline stages you want (checkboxes: Tokenizer, Packing, Pretrain, SFT, GRPO, DPO).
3. Fill in flags for each stage (grouped into collapsible sections: Model, Data, Training, Optimizer, etc.).
4. Click **Run Pipeline**.
5. The Jobs page shows each stage's status — when one completes, the next automatically starts.
6. Click a job to see **live logs** and **metric charts** (loss, VRAM, throughput, LR).

---

## API Reference

The UI exposes a REST API at `/api/*` for programmatic access:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/register` | POST | Register a new user |
| `/api/auth/[...nextauth]` | * | NextAuth built-in routes |
| `/api/nodes` | GET | List all nodes |
| `/api/nodes` | POST | Add a new node |
| `/api/nodes/:id` | GET | Get node details |
| `/api/nodes/:id` | DELETE | Remove a node |
| `/api/nodes/:id/audit` | POST | Run hardware audit |
| `/api/nodes/:id/mount` | POST | Mount NFS on node |
| `/api/nodes/:id/mount` | DELETE | Unmount NFS |
| `/api/jobs` | GET | List jobs (latest 50) |
| `/api/jobs` | POST | Launch a single job |
| `/api/jobs/pipeline` | POST | Launch a multi-stage pipeline |
| `/api/jobs/:id` | GET | Get job details + metrics |
| `/api/jobs/:id/stop` | POST | Stop a running job |
| `/api/jobs/:id/logs` | GET | Get job log tail |
| `/api/jobs/:id/metrics` | GET | Get job metrics (step, loss, etc.) |
| `/api/configs` | GET | List config presets |
| `/api/configs` | POST | Save a config preset |
| `/api/configs/:id` | PUT | Update a preset |
| `/api/configs/:id` | DELETE | Delete a preset |
| `/api/system/audit` | GET | Local GPU/CPU/RAM info |
| `/api/system/health` | GET | Health check + DB status |
| `/api/ws` | WebSocket | Real-time job updates |

---

## Troubleshooting

### "UntrustedHost: Host must be trusted" from /api/auth/*

Auth.js v5 requires the host of the incoming request to be explicitly trusted. The UI enables this by default, so this error only appears when you override `AUTH_TRUST_HOST`:

```bash
# In .env.local — make sure this is set to "true" (or unset entirely, since true is the default)
AUTH_TRUST_HOST="true"
```

If you intentionally serve the UI from a host that isn't `localhost` (e.g. a LAN IP or a reverse proxy), keep `AUTH_TRUST_HOST="true"` and ensure your reverse proxy forwards the original `Host` header. Set `AUTH_TRUST_HOST="false"` only behind a platform that Auth.js recognises automatically (Vercel, Netlify, Cloudflare, etc.).

### PostgreSQL connection refused

```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Check the port
ss -tlnp | grep 5432

# Verify pg_hba.conf allows md5 auth for local connections
sudo grep -E "local|host" $(sudo -u postgres psql -c "SHOW hba_file;" -t | tr -d ' ')
```

### "Cannot find module" errors

```bash
# Reinstall node_modules
rm -rf node_modules
npm install --legacy-peer-deps
npx prisma generate
```

### Prisma migration fails

```bash
# Push the schema directly (good for dev)
npx prisma db push

# Or reset and start fresh
npx prisma migrate reset --force
```

### WebSocket not connecting

The built-in Next.js dev server doesn't handle WebSocket upgrades. Use the custom server:

```bash
npx tsx UI/server.ts
```

Or run the dev server for HTTP and a separate WebSocket server behind a reverse proxy (nginx/caddy).

### Job fails immediately

1. Check the **Jobs** page for the error message.
2. Verify `REPO_ROOT` in `.env.local` points to the directory containing the Python scripts.
3. Ensure Python deps are installed: `pip install -r ../requirements.txt`.
4. Try running the equivalent CLI command manually to debug.

### SSH connection fails

```bash
# Test from the machine running the UI
ssh -p 22 user@remote-host "echo OK"

# Check for host key issues
ssh-keygen -R remote-host   # clear cached host key

# Verify private key permissions
chmod 600 ~/.ssh/id_ed25519
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Browser (Next.js App Router)                  │
│  ┌─────────┐  ┌──────────┐  ┌────────┐  ┌────────────────────┐ │
│  │ Auth    │  │ Dashboard│  │ Config │  │ Jobs / Charts      │ │
│  │ pages   │  │ + 3 Tabs │  │ Forms  │  │ Realtime Logs      │ │
│  └────┬────┘  └────┬─────┘  └───┬────┘  └─────────┬──────────┘ │
│       │            │            │                  │            │
│       └────────────┼────────────┼──────────────────┘ (WebSocket)│
└────────────────────┼────────────┼───────────────────────────────┘
                     │ HTTP       │ WS
              ┌──────┴────────────┴───────────┐
              │     Next.js API Routes          │
              │  /api/auth/* /api/nodes/*       │
              │  /api/jobs/* /api/configs/*     │
              └──────────────┬──────────────────┘
                             │
              ┌──────────────┴──────────────────┐
              │       lib/ (Business Logic)      │
              │  job-manager.ts  ssh-manager.ts │
              │  command-builder.ts  pipeline-   │
              │  metrics-collector.ts  orchestr- │
              │  gpu-monitor.ts  ws-manager.ts  │
              └──────────────┬──────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐        ┌────▼────┐        ┌────▼────┐
    │ Local   │        │ SSH to  │        │ WebSock │
    │ Subproc │        │ Remote  │        │ Clients │
    │ spawn() │        │ Nodes   │        │ (browsr)│
    └─────────┘        └─────────┘        └─────────┘
         │                   │
    ┌────▼────┐        ┌────▼────┐
    │Python   │        │ Remote  │
    │Training │        │ nvidia- │
    │Scripts  │        │ smi,etc │
    └─────────┘        └─────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    PostgreSQL (via Prisma)                         │
│  Users │ Nodes │ Jobs │ Metrics │ ConfigPresets │ Pipelines      │
└──────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
UI/
├── app/                     # Next.js App Router pages
│   ├── (auth)/              # Login / Signup
│   ├── (dashboard)/         # Protected dashboard
│   │   ├── torchtab/        # Torch DDP tab + sub-pages
│   │   ├── deepspeed/       # DeepSpeed tab + sub-pages
│   │   └── hivemind/        # Hivemind tab + sub-pages
│   └── api/                 # REST API route handlers
├── lib/
│   ├── schema/              # Typed flag definitions (12 files)
│   ├── job-manager.ts       # Job lifecycle singleton
│   ├── ssh-manager.ts       # SSH connection pool
│   ├── command-builder.ts   # Config → CLI command builder
│   ├── pipeline-orchestrator.ts  # Multi-stage runner
│   ├── metrics-collector.ts # Stdout → metric parser
│   ├── ws-manager.ts        # WebSocket broadcast
│   └── process-pool.ts      # Process concurrency limiter
├── hooks/                   # React hooks (6 files)
├── prisma/schema.prisma     # Database schema
└── server.ts                # Custom server (WebSocket support)
```
