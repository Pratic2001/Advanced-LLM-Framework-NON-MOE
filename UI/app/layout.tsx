import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

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
        className={`${inter.variable} ${jetbrainsMono.variable} min-h-screen bg-background antialiased`}
      >
        <div className="grid-overlay" />
        <canvas id="particle-canvas" className="particle-network" />
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
