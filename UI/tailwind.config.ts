import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./hooks/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        // Palette colors - controlled by CSS variables
        palette: {
          primary: "hsl(var(--palette-primary))",
          "primary-glow": "hsl(var(--palette-primary-glow))",
          secondary: "hsl(var(--palette-secondary))",
          "secondary-glow": "hsl(var(--palette-secondary-glow))",
          tertiary: "hsl(var(--palette-tertiary))",
          "tertiary-glow": "hsl(var(--palette-tertiary-glow))",
          accent: "hsl(var(--palette-accent))",
          "accent-glow": "hsl(var(--palette-accent-glow))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "-apple-system", "sans-serif"],
        mono: ["var(--font-mono)", "JetBrains Mono", "Fira Code", "monospace"],
      },
      animation: {
        "pulse-glow": "pulse-glow 2s ease-in-out infinite",
        "float": "float 6s ease-in-out infinite",
        "grid-scroll": "grid-scroll 20s linear infinite",
        "shimmer": "shimmer 2s ease-in-out infinite",
        "particle-drift": "particle-drift 20s ease-in-out infinite",
      },
      keyframes: {
        "pulse-glow": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.5" },
        },
        "float": {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-10px)" },
        },
        "grid-scroll": {
          "0%": { transform: "translateY(0)" },
          "100%": { transform: "translateY(-50%)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "200% 0" },
          "100%": { backgroundPosition: "-200% 0" },
        },
        "particle-drift": {
          "0%, 100%": { transform: "translate(0, 0) rotate(0deg)" },
          "25%": { transform: "translate(30px, -30px) rotate(90deg)" },
          "50%": { transform: "translate(-20px, 40px) rotate(180deg)" },
          "75%": { transform: "translate(40px, 20px) rotate(270deg)" },
        },
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic": "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
      },
      boxShadow: {
        "glow-primary": "0 0 20px hsl(var(--palette-primary) / 0.3), 0 0 40px hsl(var(--palette-primary) / 0.1), inset 0 0 20px hsl(var(--palette-primary) / 0.05)",
        "glow-secondary": "0 0 20px hsl(var(--palette-secondary) / 0.3), 0 0 40px hsl(var(--palette-secondary) / 0.1), inset 0 0 20px hsl(var(--palette-secondary) / 0.05)",
        "glow-tertiary": "0 0 20px hsl(var(--palette-tertiary) / 0.3), 0 0 40px hsl(var(--palette-tertiary) / 0.1), inset 0 0 20px hsl(var(--palette-tertiary) / 0.05)",
        "glow-accent": "0 0 20px hsl(var(--palette-accent) / 0.3), 0 0 40px hsl(var(--palette-accent) / 0.1), inset 0 0 20px hsl(var(--palette-accent) / 0.05)",
        "glass": "0 8px 32px 0 rgba(0, 0, 0, 0.37), inset 0 1px 0 rgba(255, 255, 255, 0.1)",
        "glass-strong": "0 12px 40px 0 rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.15)",
      },
      backdropBlur: {
        xs: "2px",
        "4xl": "72px",
      },
    },
  },
  plugins: [],
};

export default config;