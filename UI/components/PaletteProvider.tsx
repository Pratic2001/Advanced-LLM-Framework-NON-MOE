"use client";

import { createContext, useContext, useEffect, useState, ReactNode, useCallback, useMemo } from "react";
import { Palette, ChevronDown, ChevronUp, Settings, X, Sparkles, Sun, Moon } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export interface ColorPalette {
  id: string;
  name: string;
  description: string;
  primary: string;      // Main accent color (hsl format)
  primaryGlow: string;  // Glow variant
  secondary: string;    // Secondary accent
  secondaryGlow: string;
  tertiary: string;     // Tertiary accent
  tertiaryGlow: string;
  accent: string;       // Accent color for highlights
  accentGlow: string;
  background: string;   // Base background
  foreground: string;   // Base foreground
  card: string;         // Card background
  border: string;       // Border color
  muted: string;        // Muted foreground
}

export const PALETTES: ColorPalette[] = [
  {
    id: "neon-cyber",
    name: "Neon Cyber",
    description: "Classic cyan/blue/purple tech aesthetic",
    primary: "199 89% 48%",      // #00f0ff
    primaryGlow: "199 100% 60%",
    secondary: "220 100% 50%",   // #0088ff
    secondaryGlow: "220 100% 65%",
    tertiary: "262 83% 58%",     // #7c3aed
    tertiaryGlow: "262 100% 70%",
    accent: "330 81% 60%",       // #ec4899
    accentGlow: "330 100% 72%",
    background: "240 10% 3.9%",
    foreground: "0 0% 98%",
    card: "240 10% 5.9%",
    border: "240 3.7% 15.9%",
    muted: "240 5% 64.9%",
  },
  {
    id: "aurora-borealis",
    name: "Aurora Borealis",
    description: "Green/teal/blue northern lights",
    primary: "173 80% 40%",      // #0ec7a7
    primaryGlow: "173 100% 55%",
    secondary: "199 89% 48%",    // #00f0ff
    secondaryGlow: "199 100% 60%",
    tertiary: "220 100% 50%",    // #0088ff
    tertiaryGlow: "220 100% 65%",
    accent: "142 76% 36%",       // #22c55e
    accentGlow: "142 100% 50%",
    background: "200 20% 4%",
    foreground: "0 0% 98%",
    card: "200 15% 6%",
    border: "200 10% 15%",
    muted: "200 5% 60%",
  },
  {
    id: "solar-flare",
    name: "Solar Flare",
    description: "Orange/red/yellow energy burst",
    primary: "24 95% 53%",       // #f97316
    primaryGlow: "24 100% 65%",
    secondary: "0 84% 60%",      // #ef4444
    secondaryGlow: "0 100% 70%",
    tertiary: "51 100% 50%",     // #eab308
    tertiaryGlow: "51 100% 65%",
    accent: "32 95% 44%",        // #f59e0b
    accentGlow: "32 100% 58%",
    background: "25 15% 4%",
    foreground: "0 0% 98%",
    card: "25 10% 6%",
    border: "25 8% 15%",
    muted: "25 5% 60%",
  },
  {
    id: "cosmic-void",
    name: "Cosmic Void",
    description: "Deep purple/violet/indigo space theme",
    primary: "270 95% 65%",      // #c084fc
    primaryGlow: "270 100% 75%",
    secondary: "262 83% 58%",    // #7c3aed
    secondaryGlow: "262 100% 70%",
    tertiary: "238 90% 60%",     // #6366f1
    tertiaryGlow: "238 100% 72%",
    accent: "300 81% 60%",       // #d946ef
    accentGlow: "300 100% 72%",
    background: "260 15% 3.5%",
    foreground: "0 0% 98%",
    card: "260 12% 5.5%",
    border: "260 8% 14%",
    muted: "260 5% 58%",
  },
  {
    id: "matrix-green",
    name: "Matrix Green",
    description: "Terminal green/amber monochrome",
    primary: "142 76% 36%",      // #22c55e
    primaryGlow: "142 100% 50%",
    secondary: "120 100% 30%",   // #009900
    secondaryGlow: "120 100% 45%",
    tertiary: "84 100% 25%",     // #008000
    tertiaryGlow: "84 100% 40%",
    accent: "45 100% 51%",       // #eab308
    accentGlow: "45 100% 65%",
    background: "120 10% 3%",
    foreground: "120 100% 90%",
    card: "120 8% 5%",
    border: "120 6% 12%",
    muted: "120 5% 55%",
  },
  {
    id: "ocean-depths",
    name: "Ocean Depths",
    description: "Deep blue/teal/cyan underwater",
    primary: "189 94% 43%",      // #06b6d4
    primaryGlow: "189 100% 55%",
    secondary: "199 89% 48%",    // #00f0ff
    secondaryGlow: "199 100% 60%",
    tertiary: "173 80% 40%",     // #0ec7a7
    tertiaryGlow: "173 100% 55%",
    accent: "166 76% 42%",       // #0d9488
    accentGlow: "166 100% 55%",
    background: "200 25% 3.5%",
    foreground: "0 0% 98%",
    card: "200 20% 5.5%",
    border: "200 15% 14%",
    muted: "200 8% 58%",
  },
  {
    id: "rose-quartz",
    name: "Rose Quartz",
    description: "Soft pink/rose/magenta elegance",
    primary: "330 81% 60%",      // #ec4899
    primaryGlow: "330 100% 72%",
    secondary: "340 82% 52%",    // #f43f5e
    secondaryGlow: "340 100% 65%",
    tertiary: "300 81% 60%",     // #d946ef
    tertiaryGlow: "300 100% 72%",
    accent: "355 100% 66%",      // #f87171
    accentGlow: "355 100% 78%",
    background: "340 15% 4%",
    foreground: "0 0% 98%",
    card: "340 12% 6%",
    border: "340 10% 15%",
    muted: "340 8% 60%",
  },
  {
    id: "golden-hour",
    name: "Golden Hour",
    description: "Warm amber/orange/gold sunset",
    primary: "38 92% 50%",       // #f59e0b
    primaryGlow: "38 100% 65%",
    secondary: "24 95% 53%",     // #f97316
    secondaryGlow: "24 100% 65%",
    tertiary: "45 100% 51%",     // #eab308
    tertiaryGlow: "45 100% 65%",
    accent: "32 95% 44%",        // #d97706
    accentGlow: "32 100% 58%",
    background: "40 15% 4%",
    foreground: "40 10% 98%",
    card: "40 12% 6%",
    border: "40 10% 15%",
    muted: "40 8% 60%",
  },
];

interface PaletteContextType {
  currentPalette: ColorPalette;
  setPalette: (paletteId: string) => void;
  palettes: ColorPalette[];
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}

const PaletteContext = createContext<PaletteContextType | undefined>(undefined);

export function PaletteProvider({ children }: { children: ReactNode }) {
  const [currentPalette, setCurrentPalette] = useState<ColorPalette>(PALETTES[0]);
  const [isOpen, setIsOpen] = useState(false);

  const setPalette = useCallback((paletteId: string) => {
    const palette = PALETTES.find((p) => p.id === paletteId);
    if (palette) {
      setCurrentPalette(palette);
      applyPalette(palette);
    }
  }, []);

  useEffect(() => {
    const saved = localStorage.getItem("selected-palette");
    if (saved) {
      const palette = PALETTES.find((p) => p.id === saved);
      if (palette) {
        setCurrentPalette(palette);
        applyPalette(palette);
      }
    }
  }, []);

  const applyPalette = (palette: ColorPalette) => {
    const root = document.documentElement;
    root.style.setProperty("--palette-primary", palette.primary);
    root.style.setProperty("--palette-primary-glow", palette.primaryGlow);
    root.style.setProperty("--palette-secondary", palette.secondary);
    root.style.setProperty("--palette-secondary-glow", palette.secondaryGlow);
    root.style.setProperty("--palette-tertiary", palette.tertiary);
    root.style.setProperty("--palette-tertiary-glow", palette.tertiaryGlow);
    root.style.setProperty("--palette-accent", palette.accent);
    root.style.setProperty("--palette-accent-glow", palette.accentGlow);
    root.style.setProperty("--background", palette.background);
    root.style.setProperty("--foreground", palette.foreground);
    root.style.setProperty("--card", palette.card);
    root.style.setProperty("--border", palette.border);
    root.style.setProperty("--muted-foreground", palette.muted);
    root.style.setProperty("--muted", palette.border);
    root.style.setProperty("--ring", palette.primary);
    root.style.setProperty("--primary", palette.primary);
    root.style.setProperty("--primary-foreground", palette.foreground);
    root.style.setProperty("--secondary", palette.card);
    root.style.setProperty("--secondary-foreground", palette.foreground);
    root.style.setProperty("--accent", palette.card);
    root.style.setProperty("--accent-foreground", palette.foreground);
    root.style.setProperty("--destructive", palette.secondary);
    root.style.setProperty("--destructive-foreground", palette.foreground);
    root.style.setProperty("--popover", palette.card);
    root.style.setProperty("--popover-foreground", palette.foreground);
    root.style.setProperty("--card-foreground", palette.foreground);
    root.style.setProperty("--input", palette.border);
    localStorage.setItem("selected-palette", palette.id);
  };

  const value = useMemo(
    () => ({
      currentPalette,
      setPalette,
      palettes: PALETTES,
      isOpen,
      setIsOpen,
    }),
    [currentPalette, isOpen, setPalette]
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

export function PaletteSelector() {
  const { currentPalette, setPalette, palettes, isOpen, setIsOpen } = usePalette();

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="relative flex items-center gap-2 px-3 py-2 rounded-lg glass border border-border/50 hover:bg-accent/30 transition-all group"
        aria-label="Select color palette"
        aria-expanded={isOpen}
        aria-haspopup="listbox"
      >
        <div className="flex items-center gap-1.5">
          <Palette className="w-4 h-4 text-muted-foreground group-hover:text-foreground transition-colors" />
          <span className="text-sm font-medium hidden sm:inline">
            {currentPalette.name}
          </span>
        </div>
        <motion.div
          animate={{ rotate: isOpen ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          className="w-4 h-4 flex items-center justify-center"
        >
          <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />
        </motion.div>
        {/* Preview dots */}
        <div className="flex items-center gap-1 ml-1 hidden md:flex">
          <div
            className="w-2.5 h-2.5 rounded-full border border-border/50"
            style={{ backgroundColor: `hsl(${currentPalette.primary})` }}
          />
          <div
            className="w-2.5 h-2.5 rounded-full border border-border/50"
            style={{ backgroundColor: `hsl(${currentPalette.secondary})` }}
          />
          <div
            className="w-2.5 h-2.5 rounded-full border border-border/50"
            style={{ backgroundColor: `hsl(${currentPalette.tertiary})` }}
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
            <div className="glass-strong rounded-xl border border-border/30 p-2 min-w-[300px] shadow-glass-strong">
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
              <div className="max-h-[450px] overflow-y-auto space-y-1">
                {palettes.map((palette) => (
                  <button
                    key={palette.id}
                    onClick={() => {
                      setPalette(palette.id);
                      setIsOpen(false);
                    }}
                    className={`w-full flex items-center gap-3 p-3 rounded-lg transition-all relative overflow-hidden ${
                      palette.id === currentPalette.id
                        ? "bg-accent/50 border border-border/30"
                        : "hover:bg-accent/30"
                    }`}
                    role="option"
                    aria-selected={palette.id === currentPalette.id}
                  >
                    {/* Animated glow indicator for active palette */}
                    {palette.id === currentPalette.id && (
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: "100%" }}
                        className="absolute inset-0 bg-gradient-to-r from-[hsl(var(--palette-primary))]/10 to-transparent"
                      />
                    )}
                    <div className="flex items-center gap-1.5 relative z-10">
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
                    <div className="flex-1 min-w-0 text-left relative z-10">
                      <p className="text-sm font-medium truncate">{palette.name}</p>
                      <p className="text-xs text-muted-foreground truncate">{palette.description}</p>
                    </div>
                    {palette.id === currentPalette.id && (
                      <motion.div
                        initial={{ scale: 0, rotate: -45 }}
                        animate={{ scale: 1, rotate: 0 }}
                        className="flex items-center justify-center w-6 h-6 relative z-10"
                      >
                        <Settings className="w-4 h-4" style={{ color: `hsl(${palette.primary})` }} />
                      </motion.div>
                    )}
                  </button>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}