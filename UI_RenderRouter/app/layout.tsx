import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LLMForge Render Router",
  description:
    "Entry point that audits this device and routes it to the heavy WebGL UI or the lightweight canvas UI.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
