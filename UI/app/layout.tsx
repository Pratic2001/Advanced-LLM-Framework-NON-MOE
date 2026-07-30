import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "LLM Training Pipeline",
  description: "Advanced LLM Training Pipeline UI - Pretrain, SFT, GRPO, DPO",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-background antialiased">
        <div className="grid-overlay" />
        <canvas id="particle-canvas" className="particle-network" />
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
