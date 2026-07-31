"use client";

import {
  createContext,
  useContext,
  useEffect,
  useState,
  ReactNode,
  useCallback,
  useMemo,
  useRef,
} from "react";
import { Palette, ChevronDown, Settings, X, Sparkles, Sun, Moon, Sliders, RotateCcw } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

// ── Accent palettes ────────────────────────────────────────────────────────
//
// Each palette only carries the accent colors (primary/secondary/tertiary/
// accent and their glows). Base structural colors (background, foreground,
// card, border, muted) are owned by the active theme — see THEMES below.

export interface ColorPalette {
  id: string;
  name: string;
  description: string;
  primary: string;
  primaryGlow: string;
  secondary: string;
  secondaryGlow: string;
  tertiary: string;
  tertiaryGlow: string;
  accent: string;
  accentGlow: string;
}

export const PALETTES: ColorPalette[] = [
  {
    id: "neon-cyber",
    name: "Neon Cyber",
    description: "Classic cyan/blue/purple tech aesthetic",
    primary: "199 89% 48%",
    primaryGlow: "199 100% 60%",
    secondary: "220 100% 50%",
    secondaryGlow: "220 100% 65%",
    tertiary: "262 83% 58%",
    tertiaryGlow: "262 100% 70%",
    accent: "330 81% 60%",
    accentGlow: "330 100% 72%",
  },
  {
    id: "aurora-borealis",
    name: "Aurora Borealis",
    description: "Green/teal/blue northern lights",
    primary: "173 80% 40%",
    primaryGlow: "173 100% 55%",
    secondary: "199 89% 48%",
    secondaryGlow: "199 100% 60%",
    tertiary: "220 100% 50%",
    tertiaryGlow: "220 100% 65%",
    accent: "142 76% 36%",
    accentGlow: "142 100% 50%",
  },
  {
    id: "solar-flare",
    name: "Solar Flare",
    description: "Orange/red/yellow energy burst",
    primary: "24 95% 53%",
    primaryGlow: "24 100% 65%",
    secondary: "0 84% 60%",
    secondaryGlow: "0 100% 70%",
    tertiary: "51 100% 50%",
    tertiaryGlow: "51 100% 65%",
    accent: "32 95% 44%",
    accentGlow: "32 100% 58%",
  },
  {
    id: "cosmic-void",
    name: "Cosmic Void",
    description: "Deep purple/violet/indigo space theme",
    primary: "270 95% 65%",
    primaryGlow: "270 100% 75%",
    secondary: "262 83% 58%",
    secondaryGlow: "262 100% 70%",
    tertiary: "238 90% 60%",
    tertiaryGlow: "238 100% 72%",
    accent: "300 81% 60%",
    accentGlow: "300 100% 72%",
  },
  {
    id: "matrix-green",
    name: "Matrix Green",
    description: "Terminal green/amber monochrome",
    primary: "142 76% 36%",
    primaryGlow: "142 100% 50%",
    secondary: "120 100% 30%",
    secondaryGlow: "120 100% 45%",
    tertiary: "84 100% 25%",
    tertiaryGlow: "84 100% 40%",
    accent: "45 100% 51%",
    accentGlow: "45 100% 65%",
  },
  {
    id: "ocean-depths",
    name: "Ocean Depths",
    description: "Deep blue/teal/cyan underwater",
    primary: "189 94% 43%",
    primaryGlow: "189 100% 55%",
    secondary: "199 89% 48%",
    secondaryGlow: "199 100% 60%",
    tertiary: "173 80% 40%",
    tertiaryGlow: "173 100% 55%",
    accent: "166 76% 42%",
    accentGlow: "166 100% 55%",
  },
  {
    id: "rose-quartz",
    name: "Rose Quartz",
    description: "Soft pink/rose/magenta elegance",
    primary: "330 81% 60%",
    primaryGlow: "330 100% 72%",
    secondary: "340 82% 52%",
    secondaryGlow: "340 100% 65%",
    tertiary: "300 81% 60%",
    tertiaryGlow: "300 100% 72%",
    accent: "355 100% 66%",
    accentGlow: "355 100% 78%",
  },
  {
    id: "golden-hour",
    name: "Golden Hour",
    description: "Warm amber/orange/gold sunset",
    primary: "38 92% 50%",
    primaryGlow: "38 100% 65%",
    secondary: "24 95% 53%",
    secondaryGlow: "24 100% 65%",
    tertiary: "45 100% 51%",
    tertiaryGlow: "45 100% 65%",
    accent: "32 95% 44%",
    accentGlow: "32 100% 58%",
  },
  {
    // Deep Space: nearly-black/grey UI; the only color comes from the cosmos
    // background (stars, galaxies, black holes, hypernovas).
    id: "deep-space",
    name: "Deep Space",
    description: "Realistic cosmos — black holes, galaxies, hypernovas",
    primary: "200 20% 35%",
    primaryGlow: "200 30% 50%",
    secondary: "0 0% 18%",
    secondaryGlow: "0 0% 28%",
    tertiary: "240 10% 30%",
    tertiaryGlow: "240 15% 45%",
    accent: "45 10% 60%",
    accentGlow: "45 20% 75%",
  },
];

// Deep Space is the default palette — it is the one wired to the WebGL cosmos
// background (InteractiveBackground → CosmosWebGL), so fresh visitors land on
// the real astrophysics scene instead of a flat 2D canvas. Index-based lookup
// keeps the default in sync with the order of PALETTES (deep-space is last).
const DEFAULT_PALETTE = PALETTES[PALETTES.length - 1];

// ── Themes ────────────────────────────────────────────────────────────────
//
// Themes own structural colors (background, foreground, card, border, muted).
// One theme is active at a time and is independent of the accent palette.

export type Theme = "dark" | "light";

export interface ThemeColors {
  background: string;
  foreground: string;
  card: string;
  cardForeground: string;
  popover: string;
  popoverForeground: string;
  border: string;
  input: string;
  ring: string;
  muted: string;
  mutedForeground: string;
  secondary: string;
  secondaryForeground: string;
  accent: string;
  accentForeground: string;
  primaryForeground: string;
  destructive: string;
  destructiveForeground: string;
}

export const THEMES: Record<Theme, ThemeColors> = {
  dark: {
    background: "240 10% 3.9%",
    foreground: "0 0% 98%",
    card: "240 10% 5.9%",
    cardForeground: "0 0% 98%",
    popover: "240 10% 5.9%",
    popoverForeground: "0 0% 98%",
    border: "240 3.7% 15.9%",
    input: "240 3.7% 15.9%",
    ring: "199 89% 48%",
    muted: "240 3.7% 15.9%",
    mutedForeground: "240 5% 64.9%",
    secondary: "240 3.7% 15.9%",
    secondaryForeground: "0 0% 98%",
    accent: "240 3.7% 15.9%",
    accentForeground: "0 0% 98%",
    primaryForeground: "240 5.9% 10%",
    destructive: "0 62.8% 50.6%",
    destructiveForeground: "0 0% 98%",
  },
  light: {
    background: "0 0% 100%",
    foreground: "240 10% 3.9%",
    card: "0 0% 98%",
    cardForeground: "240 10% 3.9%",
    popover: "0 0% 100%",
    popoverForeground: "240 10% 3.9%",
    border: "240 5.9% 90%",
    input: "240 5.9% 90%",
    ring: "199 89% 48%",
    muted: "240 4.8% 95.9%",
    mutedForeground: "240 3.8% 46.1%",
    secondary: "240 4.8% 95.9%",
    secondaryForeground: "240 5.9% 10%",
    accent: "240 4.8% 95.9%",
    accentForeground: "240 5.9% 10%",
    primaryForeground: "0 0% 98%",
    destructive: "0 84.2% 60.2%",
    destructiveForeground: "0 0% 98%",
  },
};

// Deep Space overrides the structural tokens regardless of theme — the cosmos
// background is the only color the user sees, so UI chrome must be muted.
const DEEP_SPACE_OVERRIDES: Partial<ThemeColors> = {
  background: "0 0% 2%",
  card: "0 0% 5%",
  cardForeground: "0 0% 82%",
  popover: "0 0% 4%",
  popoverForeground: "0 0% 82%",
  border: "0 0% 12%",
  input: "0 0% 12%",
  ring: "200 20% 35%",
  muted: "0 0% 8%",
  mutedForeground: "0 0% 55%",
  secondary: "0 0% 8%",
  secondaryForeground: "0 0% 82%",
  accent: "0 0% 8%",
  accentForeground: "0 0% 82%",
};

// ── HSL parsing helpers ────────────────────────────────────────────────────

type HSL = { h: number; s: number; l: number };

function parseHSL(triple: string): HSL {
  const [h, s, l] = triple
    .trim()
    .split(/\s+/)
    .map((v) => parseFloat(v));
  return { h, s, l };
}

function formatHSL({ h, s, l }: HSL): string {
  return `${Math.round(h)} ${Math.round(s)}% ${Math.round(l)}%`;
}

function withLuminance(triple: string, delta: number): string {
  const { h, s, l } = parseHSL(triple);
  return formatHSL({ h, s, l: Math.min(100, Math.max(0, l + delta)) });
}

function withSaturation(triple: string, delta: number): string {
  const { h, s, l } = parseHSL(triple);
  return formatHSL({ h, s: Math.min(100, Math.max(0, s + delta)), l });
}

// ── Context ───────────────────────────────────────────────────────────────

interface PaletteContextType {
  currentPalette: ColorPalette;
  effectivePalette: ColorPalette;
  setPalette: (paletteId: string) => void;
  palettes: ColorPalette[];
  theme: Theme;
  setTheme: (theme: Theme) => void;
  isDark: boolean;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  editingPaletteId: string | null;
  setEditingPaletteId: (id: string | null) => void;
  overrides: Record<string, Partial<ColorPalette>>;
  updateOverride: (paletteId: string, key: keyof ColorPalette, value: string) => void;
  resetOverride: (paletteId: string) => void;
  hasOverride: (paletteId: string) => boolean;
}

const PaletteContext = createContext<PaletteContextType | undefined>(undefined);

// ── Provider ──────────────────────────────────────────────────────────────

export function PaletteProvider({ children }: { children: ReactNode }) {
  const [currentPalette, setCurrentPalette] = useState<ColorPalette>(DEFAULT_PALETTE);
  const [theme, setThemeState] = useState<Theme>("dark");
  const [isOpen, setIsOpen] = useState(false);
  const [editingPaletteId, setEditingPaletteId] = useState<string | null>(null);
  const [overrides, setOverrides] = useState<Record<string, Partial<ColorPalette>>>({});

  const effectivePalette = useMemo<ColorPalette>(() => {
    const o = overrides[currentPalette.id];
    return o ? { ...currentPalette, ...o } : currentPalette;
  }, [currentPalette, overrides]);

  // Apply a palette + theme combination to the document root
  const apply = useCallback((palette: ColorPalette, nextTheme: Theme) => {
    const root = document.documentElement;
    const baseColors = THEMES[nextTheme];
    const colors: ThemeColors =
      palette.id === "deep-space"
        ? { ...baseColors, ...DEEP_SPACE_OVERRIDES }
        : baseColors;

    // Theme class drives :root vs :root.dark CSS variable selection
    root.classList.toggle("dark", nextTheme === "dark");

    // Accent colors (from effective palette — includes any user overrides)
    root.style.setProperty("--palette-primary", palette.primary);
    root.style.setProperty("--palette-primary-glow", palette.primaryGlow);
    root.style.setProperty("--palette-secondary", palette.secondary);
    root.style.setProperty("--palette-secondary-glow", palette.secondaryGlow);
    root.style.setProperty("--palette-tertiary", palette.tertiary);
    root.style.setProperty("--palette-tertiary-glow", palette.tertiaryGlow);
    root.style.setProperty("--palette-accent", palette.accent);
    root.style.setProperty("--palette-accent-glow", palette.accentGlow);

    // Structural colors
    root.style.setProperty("--background", colors.background);
    root.style.setProperty("--foreground", colors.foreground);
    root.style.setProperty("--card", colors.card);
    root.style.setProperty("--card-foreground", colors.cardForeground);
    root.style.setProperty("--popover", colors.popover);
    root.style.setProperty("--popover-foreground", colors.popoverForeground);
    root.style.setProperty("--border", colors.border);
    root.style.setProperty("--input", colors.input);
    root.style.setProperty("--ring", colors.ring);
    root.style.setProperty("--muted", colors.muted);
    root.style.setProperty("--muted-foreground", colors.mutedForeground);
    root.style.setProperty("--secondary", colors.secondary);
    root.style.setProperty("--secondary-foreground", colors.secondaryForeground);
    root.style.setProperty("--accent", colors.accent);
    root.style.setProperty("--accent-foreground", colors.accentForeground);
    root.style.setProperty("--primary-foreground", colors.primaryForeground);
    root.style.setProperty("--destructive", colors.destructive);
    root.style.setProperty("--destructive-foreground", colors.destructiveForeground);

    // Primary maps to the active accent so Tailwind utilities follow it
    root.style.setProperty("--primary", palette.primary);

    localStorage.setItem("selected-palette", palette.id);
    localStorage.setItem("selected-theme", nextTheme);
  }, []);

  const setPalette = useCallback(
    (paletteId: string) => {
      const palette = PALETTES.find((p) => p.id === paletteId);
      if (!palette) return;
      setCurrentPalette(palette);
      const o = overrides[palette.id];
      apply(o ? { ...palette, ...o } : palette, theme);
    },
    [apply, overrides, theme]
  );

  const setTheme = useCallback(
    (next: Theme) => {
      setThemeState(next);
      apply(effectivePalette, next);
    },
    [apply, effectivePalette]
  );

  const updateOverride = useCallback(
    (paletteId: string, key: keyof ColorPalette, value: string) => {
      setOverrides((prev) => {
        const base = PALETTES.find((p) => p.id === paletteId);
        if (!base) return prev;
        const existing = prev[paletteId] || {};
        const nextOverride: Partial<ColorPalette> = { ...existing, [key]: value };
        // Auto-keep the glow variants in sync if the user changes the base color
        if (key === "primary") nextOverride.primaryGlow = withLuminance(value, 12);
        if (key === "secondary") nextOverride.secondaryGlow = withLuminance(value, 15);
        if (key === "tertiary") nextOverride.tertiaryGlow = withLuminance(value, 12);
        if (key === "accent") {
          nextOverride.accentGlow = withSaturation(withLuminance(value, 12), 19);
        }
        return { ...prev, [paletteId]: nextOverride };
      });
    },
    []
  );

  // Apply live overrides to the active palette
  useEffect(() => {
    apply(effectivePalette, theme);
  }, [effectivePalette, theme, apply]);

  const resetOverride = useCallback((paletteId: string) => {
    setOverrides((prev) => {
      const next = { ...prev };
      delete next[paletteId];
      return next;
    });
  }, []);

  const hasOverride = useCallback(
    (paletteId: string) => Object.keys(overrides[paletteId] || {}).length > 0,
    [overrides]
  );

  // Hydrate from localStorage on mount
  useEffect(() => {
    // URL ?palette=<id> overrides everything — handy for previewing a palette.
    const urlPalette = new URLSearchParams(window.location.search).get("palette");
    const savedPaletteId = localStorage.getItem("selected-palette");
    const savedTheme = localStorage.getItem("selected-theme");
    const palette =
      (urlPalette && PALETTES.find((p) => p.id === urlPalette)) ||
      (savedPaletteId && PALETTES.find((p) => p.id === savedPaletteId)) ||
      DEFAULT_PALETTE;
    const nextTheme: Theme = savedTheme === "light" ? "light" : "dark";
    setCurrentPalette(palette);

    const rawOverrides = localStorage.getItem("custom-palette-overrides");
    if (rawOverrides) {
      try {
        const parsed = JSON.parse(rawOverrides);
        if (parsed && typeof parsed === "object") setOverrides(parsed);
      } catch {
        /* ignore corrupt localStorage */
      }
    }
    setThemeState(nextTheme);
    apply(palette, nextTheme);
  }, [apply]);

  // Persist overrides
  useEffect(() => {
    if (Object.keys(overrides).length > 0) {
      localStorage.setItem("custom-palette-overrides", JSON.stringify(overrides));
    } else {
      localStorage.removeItem("custom-palette-overrides");
    }
  }, [overrides]);

  const value = useMemo(
    () => ({
      currentPalette,
      effectivePalette,
      setPalette,
      palettes: PALETTES,
      theme,
      setTheme,
      isDark: theme === "dark",
      isOpen,
      setIsOpen,
      editingPaletteId,
      setEditingPaletteId,
      overrides,
      updateOverride,
      resetOverride,
      hasOverride,
    }),
    [
      currentPalette,
      effectivePalette,
      isOpen,
      overrides,
      editingPaletteId,
      hasOverride,
      resetOverride,
      setPalette,
      setTheme,
      theme,
      updateOverride,
    ]
  );

  return <PaletteContext.Provider value={value}>{children}</PaletteContext.Provider>;
}

export function usePalette() {
  const context = useContext(PaletteContext);
  if (!context) {
    throw new Error("usePalette must be used within a PaletteProvider");
  }
  return context;
}

// ── HSL slider primitive ──────────────────────────────────────────────────

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  trackStyle: React.CSSProperties;
  onChange: (next: number) => void;
  format?: (n: number) => string;
}

function HslSlider({ label, value, min, max, trackStyle, onChange, format }: SliderProps) {
  const ref = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const applyFromPointer = (clientX: number) => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    const raw = min + ratio * (max - min);
    const next = Math.round(raw);
    if (next !== value) onChange(next);
  };

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    dragging.current = true;
    (e.target as Element).setPointerCapture?.(e.pointerId);
    applyFromPointer(e.clientX);
  };
  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragging.current) return;
    applyFromPointer(e.clientX);
  };
  const onPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    dragging.current = false;
    (e.target as Element).releasePointerCapture?.(e.pointerId);
  };

  const pct = ((value - min) / (max - min)) * 100;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-muted-foreground">
        <span>{label}</span>
        <span className="font-mono text-foreground/80">{format ? format(value) : value}</span>
      </div>
      <div
        ref={ref}
        role="slider"
        aria-label={label}
        aria-valuenow={value}
        aria-valuemin={min}
        aria-valuemax={max}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        className="relative h-3 rounded-full cursor-pointer select-none touch-none"
        style={trackStyle}
      >
        <div
          className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3.5 h-3.5 rounded-full border-2 border-white/90 shadow-md pointer-events-none"
          style={{
            left: `${pct}%`,
            backgroundColor: "currentColor",
          }}
        />
      </div>
    </div>
  );
}

// ── Customize editor ──────────────────────────────────────────────────────

function PaletteEditor() {
  const { editingPaletteId, effectivePalette, updateOverride, resetOverride, setEditingPaletteId } =
    usePalette();
  if (!editingPaletteId) return null;

  const base = PALETTES.find((p) => p.id === editingPaletteId);
  if (!base) return null;

  const targets: { key: keyof ColorPalette; label: string; effective: string }[] = [
    { key: "primary", label: "Primary", effective: effectivePalette.primary },
    { key: "secondary", label: "Secondary", effective: effectivePalette.secondary },
    { key: "tertiary", label: "Tertiary", effective: effectivePalette.tertiary },
    { key: "accent", label: "Accent", effective: effectivePalette.accent },
  ];

  return (
    <motion.div
      key="editor"
      initial={{ opacity: 0, x: 8 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 8 }}
      transition={{ duration: 0.15 }}
      className="border-t border-border/30 mt-2 pt-3"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Sliders className="w-3.5 h-3.5 text-[hsl(var(--palette-primary))]" />
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Customize {base.name}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => resetOverride(base.id)}
            className="flex items-center gap-1 px-2 py-1 rounded-md text-[10px] uppercase tracking-wider text-muted-foreground hover:text-foreground hover:bg-accent/30 transition-colors"
            aria-label="Reset to default"
          >
            <RotateCcw className="w-3 h-3" />
            Reset
          </button>
          <button
            onClick={() => setEditingPaletteId(null)}
            className="p-1 rounded-lg hover:bg-accent/30 transition-colors"
            aria-label="Close editor"
          >
            <X className="w-3.5 h-3.5 text-muted-foreground" />
          </button>
        </div>
      </div>

      <div className="space-y-4 max-h-[320px] overflow-y-auto pr-1">
        {targets.map(({ key, label, effective }) => {
          const { h, s, l } = parseHSL(effective);
          const setOne = (channel: keyof HSL, val: number) =>
            updateOverride(base.id, key, formatHSL({ ...parseHSL(effective), [channel]: val }));

          return (
            <div
              key={key}
              className="rounded-lg border border-border/30 p-3 bg-accent/15"
            >
              <div className="flex items-center gap-2 mb-2">
                <div
                  className="w-5 h-5 rounded-full border border-border/50 shadow-sm"
                  style={{ backgroundColor: `hsl(${effective})` }}
                />
                <span className="text-xs font-medium">{label}</span>
                <span className="ml-auto text-[10px] font-mono text-muted-foreground">
                  hsl({Math.round(h)}, {Math.round(s)}%, {Math.round(l)}%)
                </span>
              </div>
              <div className="space-y-2">
                <HslSlider
                  label="Hue"
                  value={h}
                  min={0}
                  max={360}
                  trackStyle={{
                    background:
                      "linear-gradient(to right, hsl(0,100%,50%), hsl(60,100%,50%), hsl(120,100%,50%), hsl(180,100%,50%), hsl(240,100%,50%), hsl(300,100%,50%), hsl(360,100%,50%))",
                  }}
                  onChange={(v) => setOne("h", v)}
                  format={(v) => `${v}°`}
                />
                <HslSlider
                  label="Saturation"
                  value={s}
                  min={0}
                  max={100}
                  trackStyle={{
                    background: `linear-gradient(to right, hsl(${h}, 0%, ${l}%), hsl(${h}, 100%, ${l}%))`,
                  }}
                  onChange={(v) => setOne("s", v)}
                  format={(v) => `${v}%`}
                />
                <HslSlider
                  label="Lightness"
                  value={l}
                  min={0}
                  max={100}
                  trackStyle={{
                    background: `linear-gradient(to right, hsl(${h}, ${s}%, 0%), hsl(${h}, ${s}%, 50%), hsl(${h}, ${s}%, 100%))`,
                  }}
                  onChange={(v) => setOne("l", v)}
                  format={(v) => `${v}%`}
                />
              </div>
            </div>
          );
        })}
      </div>
    </motion.div>
  );
}

// ── Selector dropdown ─────────────────────────────────────────────────────

export function PaletteSelector() {
  const {
    currentPalette,
    effectivePalette,
    setPalette,
    palettes,
    theme,
    setTheme,
    isOpen,
    setIsOpen,
    editingPaletteId,
    setEditingPaletteId,
    hasOverride,
  } = usePalette();

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="relative flex items-center gap-2 px-3 py-2 rounded-lg glass border border-border/50 hover:bg-accent/30 transition-all group"
        aria-label="Select color palette"
        aria-expanded={isOpen}
        aria-haspopup="dialog"
      >
        <div className="flex items-center gap-1.5">
          <Palette className="w-4 h-4 text-muted-foreground group-hover:text-foreground transition-colors" />
          <span className="text-sm font-medium hidden sm:inline">
            {currentPalette.name}
          </span>
          {hasOverride(currentPalette.id) && (
            <span className="w-1.5 h-1.5 rounded-full bg-[hsl(var(--palette-primary))] shadow-[0_0_6px_hsl(var(--palette-primary)/0.8)]" />
          )}
        </div>
        <motion.div
          animate={{ rotate: isOpen ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          className="w-4 h-4 flex items-center justify-center"
        >
          <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />
        </motion.div>
        {/* Preview dots — show EFFECTIVE colors so overrides are visible */}
        <div className="flex items-center gap-1 ml-1 hidden md:flex">
          <div
            className="w-2.5 h-2.5 rounded-full border border-border/50"
            style={{ backgroundColor: `hsl(${effectivePalette.primary})` }}
          />
          <div
            className="w-2.5 h-2.5 rounded-full border border-border/50"
            style={{ backgroundColor: `hsl(${effectivePalette.secondary})` }}
          />
          <div
            className="w-2.5 h-2.5 rounded-full border border-border/50"
            style={{ backgroundColor: `hsl(${effectivePalette.tertiary})` }}
          />
        </div>
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="absolute right-0 top-full mt-2 z-50"
          >
            <div className="palette-panel rounded-xl p-2 min-w-[320px] shadow-glass-strong">
              <div className="flex items-center justify-between px-3 py-2 mb-2">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-[hsl(var(--palette-primary))]" />
                  <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                    Color Palettes
                  </span>
                </div>
                <button
                  onClick={() => setIsOpen(false)}
                  className="p-1 rounded-lg hover:bg-accent/30 transition-colors"
                  aria-label="Close palette selector"
                >
                  <X className="w-3.5 h-3.5 text-muted-foreground" />
                </button>
              </div>

              {/* Theme toggle (light/dark) */}
              <div className="px-2 pb-2">
                <div className="flex items-center gap-1 p-1 rounded-lg bg-accent/30 border border-border/30">
                  <button
                    onClick={() => setTheme("light")}
                    aria-label="Light mode"
                    aria-pressed={theme === "light"}
                    className={`flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-md text-xs font-medium transition-all ${
                      theme === "light"
                        ? "bg-[hsl(var(--palette-primary))]/15 text-[hsl(var(--palette-primary))] shadow-sm"
                        : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                    }`}
                  >
                    <Sun className="w-3.5 h-3.5" />
                    <span>Light</span>
                  </button>
                  <button
                    onClick={() => setTheme("dark")}
                    aria-label="Dark mode"
                    aria-pressed={theme === "dark"}
                    className={`flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-md text-xs font-medium transition-all ${
                      theme === "dark"
                        ? "bg-[hsl(var(--palette-primary))]/15 text-[hsl(var(--palette-primary))] shadow-sm"
                        : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                    }`}
                  >
                    <Moon className="w-3.5 h-3.5" />
                    <span>Dark</span>
                  </button>
                </div>
              </div>

              <div className="max-h-[450px] overflow-y-auto space-y-1">
                {palettes.map((palette) => (
                  <div
                    key={palette.id}
                    className={`w-full flex items-center gap-3 p-3 rounded-lg transition-all relative overflow-hidden ${
                      palette.id === currentPalette.id
                        ? "bg-accent/50 border border-border/30"
                        : "hover:bg-accent/30"
                    }`}
                  >
                    {/* Animated glow indicator for active palette */}
                    {palette.id === currentPalette.id && (
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: "100%" }}
                        className="absolute inset-0 bg-gradient-to-r from-[hsl(var(--palette-primary))]/10 to-transparent pointer-events-none"
                      />
                    )}
                    <button
                      onClick={() => {
                        setPalette(palette.id);
                        setIsOpen(false);
                      }}
                      className="flex items-center gap-3 flex-1 min-w-0 relative z-10 text-left"
                      role="option"
                      aria-selected={palette.id === currentPalette.id}
                    >
                      <div className="flex items-center gap-1.5">
                        <div
                          className="w-3.5 h-3.5 rounded-full shadow-sm"
                          style={{ backgroundColor: `hsl(${palette.primary})`, boxShadow: `0 0 8px hsl(${palette.primaryGlow} / 0.6)` }}
                        />
                        <div
                          className="w-3.5 h-3.5 rounded-full shadow-sm"
                          style={{ backgroundColor: `hsl(${palette.secondary})`, boxShadow: `0 0 8px hsl(${palette.secondaryGlow} / 0.5)` }}
                        />
                        <div
                          className="w-3.5 h-3.5 rounded-full shadow-sm"
                          style={{ backgroundColor: `hsl(${palette.tertiary})`, boxShadow: `0 0 8px hsl(${palette.tertiaryGlow} / 0.5)` }}
                        />
                      </div>
                      <div className="flex-1 min-w-0 text-left">
                        <p className="text-sm font-medium truncate">{palette.name}</p>
                        <p className="text-xs text-muted-foreground truncate">{palette.description}</p>
                      </div>
                    </button>
                    <div className="flex items-center gap-2 relative z-10">
                      {hasOverride(palette.id) && (
                        <span
                          className="w-1.5 h-1.5 rounded-full bg-[hsl(var(--palette-primary))] shadow-[0_0_6px_hsl(var(--palette-primary)/0.8)]"
                          aria-label="Customized"
                        />
                      )}
                      {palette.id === currentPalette.id && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setEditingPaletteId(
                              editingPaletteId === palette.id ? null : palette.id
                            );
                          }}
                          className="flex items-center justify-center w-7 h-7 rounded-md hover:bg-accent/40 transition-colors"
                          aria-label="Customize palette"
                          aria-expanded={editingPaletteId === palette.id}
                        >
                          <motion.div
                            animate={{ rotate: editingPaletteId === palette.id ? 90 : 0 }}
                            transition={{ duration: 0.2 }}
                          >
                            <Settings className="w-4 h-4 text-[hsl(var(--palette-primary))]" />
                          </motion.div>
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              <AnimatePresence>
                {editingPaletteId && <PaletteEditor />}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}