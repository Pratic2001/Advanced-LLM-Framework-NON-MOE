#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
value_patcher.py — tune the deep-space background's beauty/performance knobs.

This script edits the numeric constants inside
    components/CosmosWebGL.tsx
(the deep-space background of the LLM training UI).  It is the one place you
go to trade *beauty* against *performance*: star density, how thick the
accretion disk is, how many meteors per shower, how many geodesic ray-tracing
steps each pixel pays for, bloom width, supernova particle counts, and so on.

HOW TO USE
----------
1. Open this file and edit the CONFIG dictionary below (the "tuning sheet").
   Every key is a knob; the comment above each group says what it controls and
   what lowering/raising it does.  Values are copied EXACTLY into the .tsx.
2. Run:
       python value_patcher.py            # patch the file (writes in place)
       python value_patcher.py --show     # preview what's currently in the file
       python value_patcher.py --diff     # dry-run: print the diff, write nothing
       python value_patcher.py --restore  # restore the pre-patch original
3. The dev server (localhost:3210) hot-reloads the change.  For production:
       npm run build
   Type-check any time with:
       npx tsc --noEmit

SAFETY
------
* The patch is transactional: every anchor must be found and every value must
  pass its range check, or NOTHING is written.
* Before the first real change a backup is saved next to the target as
  `CosmosWebGL.tsx.bak.patcher`; `--restore` brings it back.
* Anchors are strict regexes — the script only ever swaps the *numbers*, never
  the surrounding code, so the .tsx always stays syntactically valid.

DEFAULTS
--------
The CONFIG below starts at the exact reference-port values (maximum beauty).
It is a no-op to run the script without editing anything.
See VALUE_PATCHER.md for a full tunable reference and preset guidance.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

# ===========================================================================
#  ██████  ██████  ███    ██ ███████ ██  ██████  ██
#  ██      ██    ██ ████   ██ ██      ██ ██      ██
#  ██      ██    ██ ██ ██  ██ █████   ██ ██████  ██
#  ██      ██    ██ ██  ██ ██ ██      ██      ██
#   ██████  ██████  ██   ████ ██      ██ ██████  ██████
#
#  ==================================================================
#   TUNING SHEET — edit the numbers below, then run this script.
#   Defaults = the reference-port values (maximum beauty).
#   LOWER numbers = faster / cheaper rendering.
#   RAISE numbers = prettier / heavier rendering.
#  ==================================================================

CONFIG = {
    # ---------------- Sky (cheap, huge visual payoff) ----------------
    # Number of stars (THREE.Points). 3000 is already plenty on small screens.
    "star_count": 19000,
    # Number of additive nebula sprites drifting in the background.
    "nebula_count": 10,

    # ---------------- Black hole & ring (sphere tessellation) ----------------
    # Tessellation of the black event-horizon sphere (both segments equal).
    "horizon_segments": 64,
    # Tessellation of the photon-ring rim sphere that hugs the horizon.
    "rim_segments": 64,

    # ---------------- Supernova remnant iris (3D shell) ----------------
    # Width/height segments of the iris sphere that grows around each blast.
    # Kept high (160x128) so the vertex-displaced shell reads smooth/high-poly
    # rather than faceted — this is what removed the "low poly hypernova" look.
    "iris_width_segments": 160,
    "iris_height_segments": 128,

    # ---------------- Accretion disk (the heavy one) ----------------
    # How many volumetric sheets the disk is stacked from. 20 = thickest.
    # Fewer layers = much cheaper. Non-20 counts are auto-resampled from the
    # reference distribution (see VALUE_PATCHER.md).
    "disk_layer_count": 10,
    # Angular tessellation of each disk sheet (smoothness around the ring).
    "disk_angular_segments": 200,
    # Radial tessellation of each disk sheet (smoothness toward the hole).
    "disk_radial_segments": 28,

    # ---------------- Polar halo bands (lensed arcs) ----------------
    # Cross-section smoothness of each halo torus (radialSegments).
    "halo_radial_segments": 20,
    # Along-the-ring smoothness of each halo torus (tubularSegments).
    "halo_tubular_segments": 220,

    # ---------------- Supernova particle ejecta ----------------
    # Particle count per big (ambient/click) supernova. 1400 keeps the
    # asymmetric clumps and jet-axis streaks dense enough to read sculpted.
    "nova_particles_big": 1400,
    # Particle count per small supernova.
    "nova_particles_small": 900,

    # ---------------- Post pass (per-pixel, the true cost driver) ----------------
    # RK4 photon-geodesic integration steps per pixel. 200 = crisp lensing.
    # 80-120 still looks great and is dramatically faster.
    "geodesic_steps": 128,
    # Bloom taps per pixel (512-tap wide-radius bloom).
    "bloom_taps": 128,
    # Max simultaneously-lensable supernovae (must be >= 1).
    "max_novae": 6,

    # ---------------- Meteor showers ----------------
    # Seconds before the first shower (reference: 1.5 + rand*1.5).
    "meteor_first_delay": 1.5,
    # Base seconds between showers (reference: 3.5 + rand*4). Raise to 100000
    # to effectively turn meteor showers off.
    "meteor_interval": 3.5,
    # Base number of streaks in one shower (reference: 8 + rand*10).
    "meteors_per_shower": 8,
    # Random extra streaks added per shower (the "+rand*10" part).
    "meteor_shower_variance": 10,

    # ---------------- Comets ----------------
    # Seconds before the first comet (reference: 14 + rand*8).
    "comet_first_delay": 14,
    # Base seconds between comets (reference: 30 + rand*24). Raise to 100000
    # to effectively turn comets off.
    "comet_interval": 30,
    # Points in each comet's particle tail (wisp length/softness).
    "comet_trail_points": 26,

    # ---------------- Ambient supernovae ----------------
    # Base seconds between spontaneous background supernovae
    # (reference: 9 + rand*6).
    "ambient_nova_interval": 9,
}

# Quick-start profiles. `--preset performance` overrides CONFIG entirely, so
# to use your own numbers just don't pass --preset. Numbers are starting
# points — tune from there.
PRESETS = {
    "performance": {
        "star_count": 3000,
        "nebula_count": 4,
        "horizon_segments": 32,
        "rim_segments": 32,
        "iris_width_segments": 24,
        "iris_height_segments": 16,
        "disk_layer_count": 8,
        "disk_angular_segments": 96,
        "disk_radial_segments": 16,
        "halo_radial_segments": 12,
        "halo_tubular_segments": 120,
        "nova_particles_big": 700,
        "nova_particles_small": 350,
        "geodesic_steps": 90,
        "bloom_taps": 128,
        "max_novae": 4,
        "meteor_first_delay": 2.0,
        "meteor_interval": 6.0,
        "meteors_per_shower": 6,
        "meteor_shower_variance": 6,
        "comet_first_delay": 20,
        "comet_interval": 45,
        "comet_trail_points": 16,
        "ambient_nova_interval": 12,
    },
    "balanced": {
        "star_count": 6000,
        "nebula_count": 6,
        "horizon_segments": 48,
        "rim_segments": 48,
        "iris_width_segments": 160,
        "iris_height_segments": 128,
        "disk_layer_count": 14,
        "disk_angular_segments": 150,
        "disk_radial_segments": 22,
        "halo_radial_segments": 16,
        "halo_tubular_segments": 180,
        "nova_particles_big": 1400,
        "nova_particles_small": 900,
        "geodesic_steps": 140,
        "bloom_taps": 256,
        "max_novae": 5,
        "meteor_first_delay": 1.8,
        "meteor_interval": 4.5,
        "meteors_per_shower": 8,
        "meteor_shower_variance": 8,
        "comet_first_delay": 16,
        "comet_interval": 38,
        "comet_trail_points": 22,
        "ambient_nova_interval": 10,
    },
}


# ===========================================================================
#  Everything below is the patcher engine — you normally do NOT edit it.
# ===========================================================================

DEFAULT_TARGET = Path(__file__).resolve().parent / "components" / "CosmosWebGL.tsx"
BACKUP_SUFFIX = ".bak.patcher"


class PatchError(Exception):
    """Raised when an anchor can't be matched — the write is aborted."""


class ConfigError(Exception):
    """Raised when a CONFIG value fails type/range validation."""


def _pat(pattern: str) -> re.Pattern:
    """Compile a pattern verbosely? No — raw, so anchors stay grep-able."""
    return re.compile(pattern)


# Reference disk-layer offsets (the port's exact 20 values). Kept verbatim so
# disk_layer_count=20 reproduces the reference byte-for-byte; other counts are
# resampled from the *sorted* version of this distribution.
REF_DISK_LAYER_OFFSETS = [
    -0.967165, -0.640324, -0.596942, -0.544649, -0.533872,
    -0.398153, -0.278787, -0.275193, -0.274899, -0.193609,
    -0.027831, 0.110460, 0.127309, 0.208466, 0.231603,
    0.319092, 0.494153, 0.807735, 0.906307, 0.999499,
]

# One entry per knob.  `patterns` is a list of (regex, [group-indexes]): the
# regex locates the literal in the .tsx, the group-indexes name the numbers to
# replace.  Multi-pattern entries (max_novae) patch the same value in several
# places; multi-group entries (segments) patch several numbers at once.
TUNING = [
    # ---- Sky ----
    {
        "key": "star_count", "group": "Sky",
        "label": "Star count",
        "desc": "Number of points in the starfield.",
        "type": "int", "min": 200, "max": 30000, "default": 9000,
        "patterns": [(_pat(r"(createStarfield\()(\d+)(\))"), [2])],
    },
    {
        "key": "nebula_count", "group": "Sky",
        "label": "Nebula count",
        "desc": "Number of drifting nebula sprites.",
        "type": "int", "min": 0, "max": 20, "default": 7,
        "patterns": [(_pat(
            r"(for \(let i = 0; i < )(\d+)(; i\+\+\) \{\n"
            r"\s+const c = nebulaColors\[i % nebulaColors\.length\]\[1\];)"),
            [2])],
    },

    # ---- Black hole & ring ----
    {
        "key": "horizon_segments", "group": "Black hole",
        "label": "Horizon sphere segments",
        "desc": "Tessellation (width & height) of the event-horizon sphere.",
        "type": "int", "min": 12, "max": 256, "default": 64,
        "patterns": [(_pat(
            r"(new THREE\.SphereGeometry\(R_EH, )(\d+)(, )(\d+)(\))"), [2, 4])],
    },
    {
        "key": "rim_segments", "group": "Black hole",
        "label": "Photon-ring segments",
        "desc": "Tessellation (width & height) of the photon-ring rim sphere.",
        "type": "int", "min": 12, "max": 256, "default": 64,
        "patterns": [(_pat(
            r"(new THREE\.SphereGeometry\(R_EH \* 1\.035, )(\d+)(, )(\d+)(\))"),
            [2, 4])],
    },

    # ---- Supernova iris ----
    {
        "key": "iris_width_segments", "group": "Supernova",
        "label": "Iris width segments",
        "desc": "Latitudinal segments of the remnant-iris sphere. Keep high (>=160) so the displaced shell reads smooth, not faceted.",
        "type": "int", "min": 8, "max": 256, "default": 160,
        "patterns": [(_pat(
            r"(new THREE\.SphereGeometry\(1, )(\d+)(, )(\d+)(\))"), [2])],
    },
    {
        "key": "iris_height_segments", "group": "Supernova",
        "label": "Iris height segments",
        "desc": "Longitudinal segments of the remnant-iris sphere. Keep high (>=128) so the displaced shell reads smooth, not faceted.",
        "type": "int", "min": 8, "max": 192, "default": 128,
        "patterns": [(_pat(
            r"(new THREE\.SphereGeometry\(1, )(\d+)(, )(\d+)(\))"), [4])],
    },

    # ---- Accretion disk ----
    {
        "key": "disk_layer_count", "group": "Disk",
        "label": "Disk stack layers",
        "desc": "Volumetric sheets the accretion disk is built from.",
        "type": "int", "min": 1, "max": 80, "default": 20,
        "patterns": [],  # handled specially (array regeneration)
    },
    {
        "key": "disk_angular_segments", "group": "Disk",
        "label": "Disk angular segments",
        "desc": "Smoothness of each disk sheet around the ring.",
        "type": "int", "min": 24, "max": 512, "default": 200,
        "patterns": [(_pat(
            r"(new THREE\.RingGeometry\(R_IN, R_OUT, )(\d+)(, )(\d+)(\))"),
            [2])],
    },
    {
        "key": "disk_radial_segments", "group": "Disk",
        "label": "Disk radial segments",
        "desc": "Smoothness of each disk sheet toward the hole.",
        "type": "int", "min": 1, "max": 96, "default": 28,
        "patterns": [(_pat(
            r"(new THREE\.RingGeometry\(R_IN, R_OUT, )(\d+)(, )(\d+)(\))"),
            [4])],
    },

    # ---- Polar halo bands ----
    {
        "key": "halo_radial_segments", "group": "Halos",
        "label": "Halo radial segments",
        "desc": "Cross-section smoothness of each halo torus.",
        "type": "int", "min": 8, "max": 64, "default": 20,
        "patterns": [(_pat(
            r"(new THREE\.TorusGeometry\(R_EH \* radiusF, R_EH \* tubeF, )"
            r"(\d+)(, )(\d+)(\))"), [2])],
    },
    {
        "key": "halo_tubular_segments", "group": "Halos",
        "label": "Halo tubular segments",
        "desc": "Along-the-ring smoothness of each halo torus.",
        "type": "int", "min": 24, "max": 512, "default": 220,
        "patterns": [(_pat(
            r"(new THREE\.TorusGeometry\(R_EH \* radiusF, R_EH \* tubeF, )"
            r"(\d+)(, )(\d+)(\))"), [4])],
    },

    # ---- Supernova ejecta ----
    {
        "key": "nova_particles_big", "group": "Supernova",
        "label": "Big supernova particles",
        "desc": "Particle count for big (ambient/click) supernovae.",
        "type": "int", "min": 100, "max": 8000, "default": 1400,
        "patterns": [(_pat(
            r"(const count = big \? )(\d+)( : )(\d+)(;)"), [2])],
    },
    {
        "key": "nova_particles_small", "group": "Supernova",
        "label": "Small supernova particles",
        "desc": "Particle count for small supernovae.",
        "type": "int", "min": 50, "max": 4000, "default": 900,
        "patterns": [(_pat(
            r"(const count = big \? )(\d+)( : )(\d+)(;)"), [4])],
    },

    # ---- Post pass ----
    {
        "key": "geodesic_steps", "group": "Post pass",
        "label": "Geodesic RK4 steps",
        "desc": "Per-pixel photon ray-tracing steps. The single biggest cost.",
        "type": "int", "min": 40, "max": 400, "default": 200,
        "patterns": [(_pat(r"(const int STEPS = )(\d+)(;)"), [2])],
    },
    {
        "key": "bloom_taps", "group": "Post pass",
        "label": "Bloom taps",
        "desc": "Wide-radius bloom samples per pixel.",
        "type": "int", "min": 32, "max": 1024, "default": 512,
        "patterns": [(_pat(r"(const int TAPS = )(\d+)(;)"), [2])],
    },
    {
        "key": "max_novae", "group": "Post pass",
        "label": "Max simultaneous supernovae",
        "desc": "How many live explosions the ray tracer treats as obstacles.",
        "type": "int", "min": 1, "max": 10, "default": 6,
        "patterns": [
            (_pat(r"(const MAX_NOVAE = )(\d+)(;)"), [2]),
            (_pat(r"(#define MAX_NOVAE )(\d+)"), [2]),
        ],
    },

    # ---- Meteor showers ----
    {
        "key": "meteor_first_delay", "group": "Meteors",
        "label": "First meteor shower delay (s)",
        "desc": "Seconds before the first shower arrives.",
        "type": "float", "min": 0.0, "max": 60.0, "default": 1.5,
        "patterns": [(_pat(
            r"(let meteorTimer = )([\d.]+)( \+ Math\.random\(\) \* [\d.]+;)"),
            [2])],
    },
    {
        "key": "meteor_interval", "group": "Meteors",
        "label": "Meteor shower interval (s)",
        "desc": "Base seconds between showers (plus a random spread).",
        "type": "float", "min": 0.2, "max": 100000.0, "default": 3.5,
        "patterns": [(_pat(
            r"(if \(meteorTimer > )([\d.]+)( \+ Math\.random\(\) \* [\d.]+\) \{)"),
            [2])],
    },
    {
        "key": "meteors_per_shower", "group": "Meteors",
        "label": "Meteors per shower",
        "desc": "Base streak count per shower (plus variance).",
        "type": "int", "min": 1, "max": 200, "default": 8,
        "patterns": [(_pat(
            r"(const shellR = 65 \+ Math\.random\(\) \* 45;\n"
            r"\s+const count = )(\d+)( \+ Math\.floor\(Math\.random\(\) \* )"
            r"(\d+)(\);)"), [2])],
    },
    {
        "key": "meteor_shower_variance", "group": "Meteors",
        "label": "Extra meteors per shower (rand)",
        "desc": "Random extra streaks added on top of the base count.",
        "type": "int", "min": 0, "max": 200, "default": 10,
        "patterns": [(_pat(
            r"(const shellR = 65 \+ Math\.random\(\) \* 45;\n"
            r"\s+const count = )(\d+)( \+ Math\.floor\(Math\.random\(\) \* )"
            r"(\d+)(\);)"), [4])],
    },

    # ---- Comets ----
    {
        "key": "comet_first_delay", "group": "Comets",
        "label": "First comet delay (s)",
        "desc": "Seconds before the first comet arrives.",
        "type": "float", "min": 0.0, "max": 600.0, "default": 14,
        "patterns": [(_pat(
            r"(let cometTimer = )([\d.]+)( \+ Math\.random\(\) \* [\d.]+;)"),
            [2])],
    },
    {
        "key": "comet_interval", "group": "Comets",
        "label": "Comet interval (s)",
        "desc": "Base seconds between comets (plus a random spread).",
        "type": "float", "min": 1.0, "max": 100000.0, "default": 30,
        "patterns": [(_pat(
            r"(if \(cometTimer > )([\d.]+)( \+ Math\.random\(\) \* [\d.]+\) \{)"),
            [2])],
    },
    {
        "key": "comet_trail_points", "group": "Comets",
        "label": "Comet trail points",
        "desc": "Particles in each comet's trailing wisp.",
        "type": "int", "min": 4, "max": 400, "default": 26,
        "patterns": [(_pat(r"(const N = )(\d+)(;)"), [2])],
    },

    # ---- Ambient supernovae ----
    {
        "key": "ambient_nova_interval", "group": "Supernova",
        "label": "Ambient supernova interval (s)",
        "desc": "Base seconds between spontaneous background supernovae.",
        "type": "float", "min": 2.0, "max": 600.0, "default": 9,
        "patterns": [(_pat(
            r"(if \(ambientTimer > )([\d.]+)( \+ Math\.random\(\) \* [\d.]+\) \{)"),
            [2])],
    },
]

KEYS = {t["key"] for t in TUNING}

# Regex used to locate the disk-offsets array (for disk_layer_count).
# group(1) = leading whitespace + declaration, group(2) = array contents,
# group(3) = trailing ";".
_DISK_OFFSETS_RE = re.compile(
    r"^( *const DISK_LAYER_OFFSETS = )\[([^\]]*)\](;)$", re.MULTILINE)


def fmt_num(value: float | int, typ: str) -> str:
    """Format a value the way the .tsx writes it (ints bare, floats trimmed)."""
    if typ == "int":
        return str(int(value))
    s = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return s


def validate(tun: dict, value: float | int) -> float | int:
    """Coerce to the knob's type and enforce its [min, max] range."""
    key = tun["key"]
    if tun["type"] == "int":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{key}: expected an integer, got {value!r}")
        v: float | int = int(value)
        if v != value:
            raise ConfigError(f"{key}: expected an integer, got {value!r}")
    else:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ConfigError(f"{key}: expected a number, got {value!r}")
        v = float(value)
    if not (tun["min"] <= v <= tun["max"]):
        raise ConfigError(
            f"{key}: {value} is out of range [{tun['min']}, {tun['max']}]")
    return v


def rebuild(match: re.Match, groups: list[int], value: float | int, typ: str) -> str:
    """Rebuild a matched literal, replacing each captured group with `value`.

    Positional (uses match.span) so it is immune to a number that appears more
    than once inside the match as a plain substring.
    """
    parts: list[str] = []
    pos = match.start()
    for g in sorted(groups):
        s, e = match.span(g)
        parts.append(match.string[pos:s])
        parts.append(fmt_num(value, typ))
        pos = e
    parts.append(match.string[pos:match.end()])
    return "".join(parts)


def patch_disk_offsets(text: str, count: int) -> str:
    """Replace the whole DISK_LAYER_OFFSETS array with `count` layers."""
    vals = generate_disk_offsets(count)
    array_literal = "[" + ", ".join(f"{v:.6f}" for v in vals) + "]"
    m = _DISK_OFFSETS_RE.search(text)
    if not m:
        raise PatchError("disk_layer_count: DISK_LAYER_OFFSETS array not found")
    return text[: m.start()] + m.group(1) + array_literal + m.group(3) + text[m.end():]


def generate_disk_offsets(count: int) -> list[float]:
    """Return `count` disk-layer offsets.

    count == len(reference) reproduces the reference array verbatim (same
    order).  Any other count is resampled from the *sorted* reference
    distribution, so a thinner/fatter disk keeps the same thickness profile.
    """
    if count == len(REF_DISK_LAYER_OFFSETS):
        return list(REF_DISK_LAYER_OFFSETS)
    if count == 1:
        return [0.0]
    s = sorted(REF_DISK_LAYER_OFFSETS)
    out: list[float] = []
    for k in range(count):
        x = k * (len(s) - 1) / (count - 1)
        i0 = int(x)
        frac = x - i0
        i1 = min(i0 + 1, len(s) - 1)
        v = s[i0] + (s[i1] - s[i0]) * frac
        out.append(round(v, 6))
    return out


def apply_tunable(text: str, tun: dict, value: float | int) -> str:
    """Patch every anchor of one knob. Raises PatchError if any anchor fails."""
    for pat, groups in tun["patterns"]:
        matches = pat.findall(text)
        if len(matches) != 1:
            raise PatchError(
                f"{tun['key']}: expected exactly 1 anchor, found {len(matches)} "
                f"(pattern: {pat.pattern[:70]!r}...)")

        def repl(m: re.Match, groups: list[int] = groups,
                 value: float | int = value, tun: dict = tun) -> str:
            return rebuild(m, groups, value, tun["type"])

        text = pat.sub(repl, text, count=1)
    return text


def apply_all(text: str, config: dict) -> str:
    """Validate every knob, then patch all anchors. All-or-nothing."""
    unknown = set(config) - KEYS
    if unknown:
        raise ConfigError("unknown config key(s): " + ", ".join(sorted(unknown)))
    for tun in TUNING:
        value = validate(tun, config[tun["key"]])
        if tun["key"] == "disk_layer_count":
            text = patch_disk_offsets(text, int(value))
        else:
            text = apply_tunable(text, tun, value)
    return text


def read_current(text: str) -> dict:
    """Return {key: [current literal(s) in the file]} for display/diff."""
    out: dict = {}
    for tun in TUNING:
        if tun["key"] == "disk_layer_count":
            m = _DISK_OFFSETS_RE.search(text)
            out[tun["key"]] = (
                [str(len(re.findall(r"-?[\d.]+", m.group(2))))] if m else None)
            continue
        vals: list[str] | None = []
        for pat, groups in tun["patterns"]:
            m = pat.search(text)
            if not m:
                vals = None
                break
            vals.extend(m.group(g) for g in groups)
        out[tun["key"]] = vals
    return out


def looks_like_target(text: str) -> bool:
    return "CosmosWebGL" in text and "Deep-space background" in text


def render_current(vals: list[str] | None) -> str:
    if vals is None:
        return "<anchor missing!>"
    if len(set(vals)) == 1:
        return vals[0]
    return "mixed: " + ", ".join(vals)


def print_summary(cur: dict, config: dict, old_text: str, new_text: str) -> None:
    rows = []
    for tun in TUNING:
        key = tun["key"]
        new = config[key]
        new_str = fmt_num(new, tun["type"])
        c = cur.get(key)
        cur_str = render_current(c) if c is not None else "<anchor missing!>"
        marker = " " if cur_str == new_str else "*"
        rows.append((tun["group"], key, cur_str, new_str, marker))
    groups = list(dict.fromkeys(r[0] for r in rows))
    width_key = max(len(r[1]) for r in rows)
    print("\nChange summary ( * = will change / changed ):")
    for g in groups:
        print(f"\n  {g}:")
        for grp, key, old, new, marker in rows:
            if grp != g:
                continue
            print(f"   {marker} {key:<{width_key}}  {old}  ->  {new}")


def do_show(text: str, target: Path) -> None:
    cur = read_current(text)
    print(f"Current values in {target}\n")
    groups = list(dict.fromkeys(t["group"] for t in TUNING))
    width_key = max(len(t["key"]) for t in TUNING)
    width_lab = max(len(t["label"]) for t in TUNING)
    for g in groups:
        print(f"  {g}:")
        for t in TUNING:
            if t["group"] != g:
                continue
            print(f"    {t['key']:<{width_key}}  {t['label']:<{width_lab}}  "
                  f"= {render_current(cur.get(t['key']))}  "
                  f"(default {fmt_num(t['default'], t['type'])})")
    print("\nCopy-paste the block below into CONFIG, edit, then run the script:")
    print("\nCONFIG = {")
    for t in TUNING:
        key = t["key"]
        val = cur.get(key)
        shown = val[0] if val and len(set(val)) == 1 else fmt_num(t["default"], t["type"])
        print(f'    "{key}": {shown},')
    print("}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="value_patcher.py",
        description="Tune the deep-space background's beauty/performance knobs "
                    "and patch them into components/CosmosWebGL.tsx.",
        epilog="With no command, the script applies CONFIG. See VALUE_PATCHER.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--file", metavar="PATH", default=None,
                   help="target CosmosWebGL.tsx (default: this repo's copy)")
    p.add_argument("--show", action="store_true",
                   help="print the values currently in the file, then exit")
    p.add_argument("--diff", action="store_true",
                   help="dry run: print the unified diff without writing")
    p.add_argument("--apply", action="store_true",
                   help="patch the file (the default action)")
    p.add_argument("--restore", action="store_true",
                   help="restore the pre-patch original from the backup")
    p.add_argument("--preset", choices=sorted(PRESETS), default=None,
                   help="start from a preset instead of CONFIG")
    args = p.parse_args(argv)

    target = Path(args.file).resolve() if args.file else DEFAULT_TARGET
    if not target.exists():
        p.error(f"target not found: {target}")

    if args.restore:
        backup = target.with_name(target.name + BACKUP_SUFFIX)
        if not backup.exists():
            print(f"No backup found at {backup} — nothing to restore.")
            return 1
        target.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
        backup.unlink()
        print(f"Restored {target} from {backup} (backup removed).")
        return 0

    text = target.read_text(encoding="utf-8")
    if not looks_like_target(text):
        p.error(f"{target} does not look like CosmosWebGL.tsx — refusing to touch it.")

    if args.show:
        do_show(text, target)
        return 0

    config = dict(PRESETS[args.preset]) if args.preset else dict(CONFIG)
    missing = KEYS - set(config)
    extra = set(config) - KEYS
    if missing:
        p.error("CONFIG is missing keys: " + ", ".join(sorted(missing)))
    if extra:
        p.error("CONFIG has unknown keys: " + ", ".join(sorted(extra)))

    new_text = apply_all(text, config)

    if new_text == text:
        if args.diff:
            print("No changes — the file already matches.")
        else:
            print("No changes needed — the file already matches CONFIG.")
        return 0

    diff = "".join(difflib.unified_diff(
        text.splitlines(keepends=True), new_text.splitlines(keepends=True),
        fromfile=str(target), tofile="<patched>"))
    if args.diff:
        print(diff)
        print("\n(dry run — nothing was written.)")
        return 0

    cur = read_current(text)
    backup = target.with_name(target.name + BACKUP_SUFFIX)
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"Backup created: {backup}")
    else:
        print(f"Backup already exists (kept as-is): {backup}")

    target.write_text(new_text, encoding="utf-8")
    print_summary(cur, config, text, new_text)
    print("\nPatched. The dev server (localhost:3210) hot-reloads the change;")
    print("for production run `npm run build` and type-check with `npx tsc --noEmit`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
