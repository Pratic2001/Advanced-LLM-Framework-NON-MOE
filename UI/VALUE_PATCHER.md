# value_patcher.py — tuning the deep-space background

`value_patcher.py` is the single place you go to trade **beauty** against
**performance** in the deep-space background. It edits the numeric constants
inside `components/CosmosWebGL.tsx` — star density, accretion-disk thickness,
meteor/comet volume and timing, per-pixel geodesic ray-tracing steps, bloom
width, supernova particle counts, and more — then leaves the file ready for the
next dev hot-reload or production build.

It does **not** touch any shader logic or layout code. It only swaps numbers
behind strict, verified anchors, so the file always stays syntactically valid
TypeScript.

---

## Quick start

```bash
# 1. Open value_patcher.py and edit the CONFIG "tuning sheet" at the top.
#    Every knob has a comment saying what it does and which way to move it.

# 2. Preview what's currently in the file (no writes):
python3 value_patcher.py --show

# 3. See exactly what would change (no writes):
python3 value_patcher.py --diff

# 4. Patch the file in place:
python3 value_patcher.py

# 5. Undo to the last pre-patch original (restores from the auto-backup):
python3 value_patcher.py --restore
```

After patching:

- The **dev server** (`localhost:3210`) hot-reloads the change — hard-refresh
  the page to see it.
- For production: `npm run build`.
- Type-check any time: `npx tsc --noEmit`.

Running the script with an unedited `CONFIG` is a **no-op** — the tuning-sheet
values are exactly what's already in the file (the current tuned state), so
nothing changes until you edit a number.

---

## The tuning sheet (`CONFIG`)

Everything you can edit lives in one dict at the top of the script. **Lower
numbers = faster / cheaper rendering. Raise numbers = prettier / heavier.**

| Group | Key | Default | Range | What it does |
|------|-----|--------:|------:|--------------|
| Sky | `star_count` | 14000 | 200–30000 | Points in the starfield. Cheap to cut — 3000 looks fine on small screens. |
| Sky | `nebula_count` | 8 | 0–20 | Drifting nebula sprites behind the hole. |
| Black hole | `horizon_segments` | 64 | 12–256 | Tessellation of the black event-horizon sphere. |
| Black hole | `rim_segments` | 64 | 12–256 | Tessellation of the photon-ring rim sphere. |
| Disk | `disk_layer_count` | 8 | 1–80 | Volumetric sheets the accretion disk is stacked from. **The heavy one** — each layer is a full shaded mesh. |
| Disk | `disk_angular_segments` | 200 | 24–512 | Smoothness of each disk sheet around the ring. |
| Disk | `disk_radial_segments` | 28 | 1–96 | Smoothness of each disk sheet toward the hole. |
| Halos | `halo_radial_segments` | 20 | 8–64 | Cross-section smoothness of each polar halo torus. |
| Halos | `halo_tubular_segments` | 220 | 24–512 | Along-the-ring smoothness of each halo torus. |
| Supernova | `iris_width_segments` | 160 | 8–256 | Latitudinal segments of each blast's 3D iris shell. Keep high (≥160) so the displaced shell reads smooth, not faceted. |
| Supernova | `iris_height_segments` | 128 | 8–192 | Longitudinal segments of the iris shell. Keep high (≥128) for a smooth shell. |
| Supernova | `nova_particles_big` | 1150 | 100–8000 | Particles per big (ambient/clicked) supernova. Fewer cuts the spawn-time allocation burst + per-frame ejecta update (the frame hitch right as a nova ignites) while keeping the clumps sculpted. |
| Supernova | `nova_particles_small` | 750 | 50–4000 | Particles per small supernova. |
| Supernova | `ambient_nova_interval` | 15 | 2–600 | Base seconds between spontaneous background supernovae (plus a random spread). Raise to have fewer blast-trigger frame dips. |
| Post pass | `geodesic_steps` | 112 | 40–400 | RK4 photon-geodesic integration steps **per pixel**. The single biggest frame cost. 80–120 still looks great. |
| Post pass | `bloom_taps` | 128 | 32–1024 | Wide-radius bloom samples per pixel. |
| Post pass | `max_novae` | 4 | 1–10 | How many live explosions the ray tracer treats as lensing obstacles. Patches both the JS constant and the GLSL `#define`. |
| Render res | `supersample_factor` | 1.15 | 1.0–3.0 | Internal render-resolution multiplier. Scene + per-pixel ray march both render at (drawing-buffer size × this), then the final blit downsamples to the screen — the only AA that reaches per-pixel ray-traced edges. See "Supersampling" below. |
| Meteors | `meteor_first_delay` | 1.5 | 0–60 | Seconds before the first meteor shower arrives. |
| Meteors | `meteor_interval` | 3.5 | 0.2–100000 | Base seconds between showers (plus a random spread). |
| Meteors | `meteors_per_shower` | 8 | 1–200 | Base streak count per shower. |
| Meteors | `meteor_shower_variance` | 10 | 0–200 | Random extra streaks added per shower. |
| Comets | `comet_first_delay` | 14 | 0–600 | Seconds before the first comet arrives. |
| Comets | `comet_interval` | 30 | 1–100000 | Base seconds between comets (plus a random spread). |
| Comets | `comet_trail_points` | 26 | 4–400 | Particles in each comet's trailing wisp. |

The Default column is the **current tuned state** of the file (what `CONFIG`
holds), not a historical maximum. `--show` also prints each knob's
*reference* default in parentheses.

> **Turning effects off:** set `meteor_interval` / `comet_interval` to a large
> number (e.g. `100000`) and the shower/comet simply never fires again in any
> realistic session. Set `nebula_count` to `0` for no nebulae.

### Supersampling (`supersample_factor`)

Most of deep-space's look is not geometry — it's a **per-pixel ray march** in
the post pass (real photon-geodesic tracing through the black hole's
spacetime, then bloom + grade). Ordinary MSAA can never smooth that, because
there are no polygon edges to sample — each pixel is computed independently.

`supersample_factor` fixes this the only way that reaches per-pixel content:
the scene **and** the ray march both render at `(drawing-buffer size × factor)`,
and the final blit downsamples to the screen with linear filtering. Every
screen pixel is an average of `factor²` ray-traced samples, which visibly
smooths the black-hole shadow edge, lensing caustics, star points and the
accretion-disk rim.

- `1.0` — device resolution. No smoothing; the cheapest.
- `1.15` — **default.** ~32% more ray-march pixels per frame. A clear smoothing
  win at a modest cost — the sweet spot for mid-range GPUs.
- `1.25` — ~56% more pixels; a further quality jump for fast dedicated GPUs.
- `1.5` — ~2.25× the ray-march work (resolution scales by *area*). Very heavy;
  only worth it on top-end GPUs.

The bloom width is unchanged by this knob: bloom offsets stay measured in CSS
pixels, and the final downsample cancels the supersample scale.

---

## Presets

Quick-start profiles are included. They override `CONFIG` entirely, so to use
your own numbers just don't pass `--preset`.

| Preset | Idea |
|--------|------|
| `--preset performance` | Aimed at weak iGPUs / 60 fps on small screens. ~½ the geodesic steps, 128-tap bloom, 8 disk layers, 700/350 nova particles, supersampling **off** (`1.0`), fewer meteors. |
| `--preset balanced` | Middle ground — noticeably lighter than the reference, still rich. Geodesic steps up, supersampling `1.15`. |

```bash
python3 value_patcher.py --preset performance   # apply
python3 value_patcher.py --diff --preset balanced   # preview first
```

These are starting points, not gospel — tune from there. The "right" setting
depends on the target machine; the biggest levers are `geodesic_steps` (per-pixel
ray march) and `supersample_factor` (render resolution — it scales by *area*, so
`1.5` is ~2.25× the ray-march work), followed by `disk_layer_count` and
`star_count`.

---

## How it works

1. **Anchors, not line numbers.** Each knob owns a strict regex that locates
   its literal in `CosmosWebGL.tsx` (e.g. `const int STEPS = 128;`,
   `const SUPERSAMPLE = 1.25;` or the `createStarfield(19000)` call). The
   regex must match **exactly once**; the patcher only rewrites the captured
   *numbers*, preserving all surrounding code verbatim.

2. **Transactional (all-or-nothing).** Every knob is validated (type + range)
   and every anchor is checked *before* anything is written. If any anchor is
   missing, duplicated, or any value is out of range, the script aborts and
   leaves the file untouched — you can't end up half-patched.

3. **Backup + restore.** The first real change saves a backup next to the
   target as `CosmosWebGL.tsx.bak.patcher`. `--restore` puts it back and
   removes the backup. Re-running with the same `CONFIG` is a **no-op**
   (idempotent).

4. **Disk layer resampling.** The disk is defined by an array of 20 vertical
   offsets. `disk_layer_count = 20` reproduces the reference array
   byte-for-byte. Any other count is resampled from the *sorted* reference
   distribution, so a thinner or fatter disk keeps the same thickness profile
   (dense near the mid-plane, sparse toward the poles).

---

## Commands

| Command | Effect |
|---------|--------|
| *(no flag)* | Apply `CONFIG` (or `--preset`) to the file. |
| `--show` | Print every value currently in the file, plus a copy-paste `CONFIG` block. |
| `--diff` | Dry run: print the unified diff, write nothing. |
| `--apply` | Explicitly apply (same as the default action). |
| `--restore` | Restore the pre-patch original from the backup. |
| `--preset <name>` | Start from a preset instead of `CONFIG`. |
| `--file <path>` | Target a different `CosmosWebGL.tsx` (useful for CI/testing). |

---

## Troubleshooting

**"expected exactly 1 anchor, found 0 / found N"**
The literal the knob targets has moved or been edited in `CosmosWebGL.tsx`
(e.g. after a manual edit or a future port). Nothing was written. Find the key
in the error message, check the corresponding code in the `.tsx`, and either
fix the file or adjust the pattern in the script's `TUNING` table.

**"out of range"**
The value you entered is outside the knob's sanity bounds. Nothing was
written. Pick a number inside `[min, max]` (see the table above).

**I want my exact disk layers back.**
`--show` prints the current `disk_layer_count`; setting it back to `20`
recreates the reference array exactly. If you had a *custom* array you want to
restore, `--restore` brings back the pre-patch file.

**The patched file looks wrong / doesn't compile.**
Run `npx tsc --noEmit`. If it's clean, the numbers are fine and the look is
just a tuning choice — move the knob back or `--restore`. If it reports an
error, the anchor patched the wrong thing; report the key and the offending
line, and use `--restore` to get back to a known-good file.
