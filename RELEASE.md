# Release Pipeline

This document describes the GitHub Actions release workflow at
`.github/workflows/release.yml`. It is the source of truth for **how a
release is cut, what artifacts it produces, and how to wire the optional
PDF email notification**.

> Looking for how to *use* a released stack on a fresh PC?
> See [README.md → Containerized Deployment](./README.md#-containerized-deployment).

---

## At a glance

```text
git tag v0.2.0 && git push --tags
              │
              ▼
┌────────────────────────────────────────────────────────────────────┐
│  .github/workflows/release.yml                                     │
│                                                                    │
│   build-and-push ──► generate-env-bundle ──► notify-release        │
│        │                    │                      │               │
│        │                    │                      └─► release.pdf  │
│        │                    │                          uploaded as   │
│        │                    │                          artifact     │
│        │                    │                          + emailed    │
│        │                    └─► env-bundle.tar.gz                   │
│        │                         uploaded as artifact              │
│        ▼                                                         │
│   pratic2001/llmforge-{ui-router,ui,trainer}:v0.2.0               │
│   (also tagged :latest)                                           │
└────────────────────────────────────────────────────────────────────┘
```

Two artifacts ship on every successful run:

| Artifact | Contents | Used by |
|---|---|---|
| `env-bundle` | `secrets/{router,ui,ui-lite,trainer}.env` (the runtime env files the compose stack expects) | Anyone deploying the stack — drop into `./secrets/` next to `docker-compose.yml` and run `docker compose up -d` |
| `release-summary-<tag>` | One-page PDF with tag, head SHA, Docker image digests (resolved live from Docker Hub), "What's New" commit log since the previous tag, and deployment instructions | Anyone who wants a shareable record of what shipped |

The PDF is also **emailed** to a recipient of your choice if the SMTP
secrets below are configured. The artifact is the canonical source of
truth — email is a convenience.

---

## Triggers

| Trigger | Use it for |
|---|---|
| `push` of a tag matching `v*` (e.g. `v0.2.0`, `v1.0.0-rc.3`) | Cutting a real release |
| `workflow_dispatch` (manual button in the Actions tab) | Re-running a failed release without re-tagging. Optional `tag` input lets you pin a specific version to publish as `:latest` without tagging `main` |

A single concurrency group (`release-${{ github.ref }}`) prevents two
releases of the same ref from racing.

---

## Jobs, in order

### 1. `build-and-push` (matrix over the 3 services)

For each of `ui-router`, `ui`, `trainer`:

1. **Compute image tag** — defaults to `latest` for `workflow_dispatch`,
   otherwise uses the pushed git tag.
2. **Log in to Docker Hub** using `DOCKERHUB_USERNAME` + `DOCKERHUB_TOKEN`.
3. **Set up Buildx** for layer caching.
4. **Compute build args** — `PUBLIC_HOSTNAME` and the two
   `NEXT_PUBLIC_*_UI_URL` build args get inlined at build time so the
   rendered UI knows its public URL.
5. **Build & push** the image with `docker/build-push-action@v6` using the
   GHA layer cache. Tags both `:vX.Y.Z` and `:latest`.

### 2. `generate-env-bundle`

Runs after `build-and-push` regardless of smoke-test result (the
env-bundle is what your downstream `install.sh` or `docker compose`
deploy needs).

Generates `secrets/{router,ui,ui-lite,trainer}.env` from the same GH
Secrets the runtime stack needs (`NEXTAUTH_SECRET`, `AUTH_SECRET`,
`DATABASE_URL`, `SSH_KEY_ENCRYPTION_KEY`, `PUBLIC_HOSTNAME`) and packs
them into `env-bundle.tar.gz`.

### 3. `smoke-test`

Spins up the `ui` and `ui-router` services with `docker compose up -d`,
waits for them to come up healthy, and probes a known route to confirm
Next is serving real HTML — then tears down.

**Why this step exists:** every release is the moment a downstream user
will `docker compose pull && docker compose up -d` for the first time.
A build that succeeds but a server that can't actually answer an HTTP
request is a silent production outage. This job catches it before any
artifact is published.

**How it probes:** the smoke-test hits `/dashboard` on each port. Next
strips the request's `basePath` before middleware sees it, so this probe
works regardless of `NEXT_PUBLIC_BASE_PATH` or `NEXTAUTH_URL`. (See
[Why not /api/auth/providers?](#why-not-apiauthproviders) below.)

**Cold start:** CI runners do a fresh `npm install` inside the entrypoint
on every boot, plus Next's on-disk compile plus Prisma's first-run
generate. The 90×2s = 3 minute wait budget covers this on the slowest
runner.

### 4. `notify-release` — Generate release PDF + email

Generates the `release-summary-<tag>.pdf` artifact and (optionally)
emails it.

Steps:

1. **Determine tag + previous tag.** For a tag-triggered run, that's
   `GITHUB_REF_NAME` and the previous `v[0-9]...` tag from
   `git tag --sort=-version:refname`. For a manual run with the `tag`
   input, same logic but using the input.
2. **Resolve image digests from Docker Hub** by querying the registry
   API for each tag — gives a verifiable record of what was actually
   pushed, not just "the build claimed success."
3. **Generate release PDF** using `pdfkit` (no browser dependency,
   no headless Chromium). The PDF contains:
   - Header: project name, tag, ISO timestamp, head SHA
   - Image list with digests (`sha256:...`) and sizes
   - "What's New" commit log since the previous tag (or the last 25
     commits if no previous tag exists)
   - Deployment instructions
4. **Upload release-summary-<tag>.pdf** as a workflow artifact (always).
5. **Email release PDF** via `dawidd6/action-send-mail@v3` (only if SMTP
   secrets are set, see below).

---

## GitHub Actions Secrets

Required for **any** release to work:

| Secret | Used by | Description |
|---|---|---|
| `DOCKERHUB_USERNAME` | build-and-push | Docker Hub login, e.g. `pratic2001` |
| `DOCKERHUB_TOKEN` | build-and-push | Docker Hub Personal Access Token (read+write) |
| `PUBLIC_HOSTNAME` | build-and-push, env-bundle | Your tailnet hostname, e.g. `pratic-battleaxb450mkm2.tail5e5151.ts.net` |
| `NEXTAUTH_SECRET` | env-bundle | 32+ char secret (`openssl rand -base64 32`) |
| `AUTH_SECRET` | env-bundle | Same value as `NEXTAUTH_SECRET` (NextAuth v5) |
| `SSH_KEY_ENCRYPTION_KEY` | env-bundle | 32 hex chars for AES-256-GCM |
| `DATABASE_URL` | env-bundle | `postgresql://...` (the UI's Postgres) |

Optional — only needed if you want the trainer service and the
trainer-as-Node flow:

| Secret | Used by | Description |
|---|---|---|
| `SSH_PUBLIC_KEY` | trainer image build | The public half of an ed25519 keypair — gets baked into the trainer image's `authorized_keys` |
| `SSH_PRIVATE_KEY` | env-bundle | The private half — mounted into the UI container, used by `node-ssh` |

Optional — only needed if you want the release PDF **emailed**:

| Secret | Used by | Description |
|---|---|---|
| `SMTP_HOST` | notify-release | e.g. `smtp.gmail.com` |
| `SMTP_PORT` | notify-release | `465` (default, TLS) |
| `SMTP_SECURE` | notify-release | `true` (default) |
| `SMTP_USERNAME` | notify-release | Sender email address, e.g. `cpratic8@gmail.com` |
| `SMTP_PASSWORD` | notify-release | **App password**, not the account password. Gmail: <https://myaccount.google.com/apppasswords> (requires 2FA) |
| `NOTIFY_EMAIL` | notify-release | Recipient address. Defaults to `SMTP_USERNAME` if unset. |

> **Without SMTP secrets, the workflow still works** — the PDF is always
> uploaded as an artifact. The "Email release PDF" step is skipped with a
> notice that points you at the Settings page.

---

## Appendix: Why not `/api/auth/providers`?

The smoke-test originally probed `/api/auth/providers` because that's
the canonical "is NextAuth alive?" endpoint. It turned out to be the
wrong choice for this stack. The root cause was a triple-stack problem
unique to the `ui` container:

1. **One container, two servers.** The `ui` container runs BOTH the
   Heavy UI on port 3000 AND UI_lite on port 3001 in the same process
   tree, supervised by an `entrypoint.sh`.
2. **`env_file` stacks.** `docker-compose.yml` lists both
   `secrets/ui.env` and `secrets/ui-lite.env` in the `ui` service's
   `env_file:`. Compose merges them all into the same shell environment.
3. **`tsx --env-file=` merges, doesn't unset.** When the entrypoint
   launches the second server, `tsx --env-file=/secrets/ui-lite.env`
   reads the file on top of the *inherited* env (which still has
   `NEXT_PUBLIC_BASE_PATH=/heavy` from the first launch), instead of
   starting from a clean slate. One of the two server processes ends
   up with the wrong `NEXT_PUBLIC_BASE_PATH` and `NEXTAUTH_URL`.
4. **NextAuth v5 derives `basePath` from `NEXTAUTH_URL`'s pathname.**
   Specifically, in `next-auth/lib/env.js` `setEnvDefaults()`:
   `config.basePath || (config.basePath = pathname)`. With
   `NEXTAUTH_URL=https://example.test` (no path), basePath becomes `/`,
   which breaks `parseActionAndProviderId('/api/auth/providers')` →
   `UnknownAction` 400.

`/dashboard` is basePath-stripped by Next *before* middleware sees it,
so it works regardless of `NEXT_PUBLIC_BASE_PATH` or `NEXTAUTH_URL`. We
collapsed the old two-step "wait, then probe /api/auth/providers" into
a single combined step.

There are other fixes you could pick instead — splitting the two UI
servers into separate containers, or having the entrypoint `unset`
`NEXT_PUBLIC_BASE_PATH` between launches. But `/dashboard` is the
smallest correct change that doesn't perturb the rest of the pipeline.

---

## Appendix: Smoke-test cold-start math

The smoke-test waits up to 90×2s = 3 minutes for the containers to come
healthy. Breakdown of where that time goes on a fresh CI runner:

| Stage | Approx time |
|---|---|
| `docker compose pull` (multi-GB images) | 30–60s |
| Tailscale sidecar boot + auth | 5–15s |
| UI entrypoint: `npm ci` (Heavy + Lite share deps, so ~once) | 30–60s |
| First-boot Next compile | 10–20s |
| Prisma `generate` on first launch | 5–15s |
| **Total** | **80–170s** |

The 3-minute budget is comfortable for the slowest observed runner. If
you bump to a bigger image set or add a new service, increase
`$(seq 1 90)` proportionally.
