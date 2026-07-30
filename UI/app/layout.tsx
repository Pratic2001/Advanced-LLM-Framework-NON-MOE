import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";

// NOTE: We intentionally do NOT use next/font/google here. That loader
// fetches font files from fonts.googleapis.com at build/dev-compile time,
// which throws inside the root layout (and therefore triggers
// app/global-error.tsx, the "Critical Error" screen) on networks that can't
// reach the public internet — e.g. this app tunnelled over Tailscale with
// no outbound access. --font-inter / --font-mono already have system-font
// fallbacks defined in globals.css and tailwind.config.ts, so we just
// declare them as static variables here instead of fetching anything.
const fontVariables = "font-vars";

export const metadata: Metadata = {
  metadataBase: new URL("https://llmforge.dev"),
  title: {
    default: "LLMForge — LLM Training Pipeline",
    template: "%s | LLMForge",
  },
  description: "Advanced LLM Training Pipeline UI — Pretrain, SFT, GRPO, DPO",
  icons: {
    icon: "/favicon.svg",
    apple: "/apple-touch-icon.png",
  },
  openGraph: {
    title: "LLMForge",
    description: "Orchestrate LLM training from browser to cluster",
    url: "https://llmforge.dev",
    siteName: "LLMForge",
    images: [{ url: "/og-image.png", width: 1200, height: 630 }],
    locale: "en_US",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${fontVariables} min-h-screen bg-background antialiased`}
      >
        <div className="grid-overlay" />
        <canvas id="particle-canvas" className="particle-network" />
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
