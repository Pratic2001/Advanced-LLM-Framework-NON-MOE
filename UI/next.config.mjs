/** @type {import('next').NextConfig} */
const nextConfig = {
  // Security: remove X-Powered-By header
  poweredByHeader: false,

  // Build output for Docker
  output: "standalone",

  // Keep native modules out of webpack's bundle. node-pty ships a `.node`
  // binary that webpack can't bundle — marking it external makes Next leave
  // it as a runtime `require` so the build doesn't try to resolve
  // `pty.node` against the host filesystem. The route handlers under
  // `app/api/shell/*` import this transitively through `pty-manager`.
  // (Next.js 15+ moved this out of `experimental`.)
  serverExternalPackages: ["@homebridge/node-pty-prebuilt-multiarch"],

  // Pin the Turbopack workspace root so it stops picking up
  // /home/pratic/package-lock.json as the project root.
  turbopack: {
    root: process.cwd(),
  },

  // Security headers at the config level
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          {
            key: "Content-Security-Policy",
            value:
              process.env.NODE_ENV === "production"
                ? [
                    "default-src 'self'",
                    "script-src 'self' 'unsafe-eval' 'unsafe-inline'",
                    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
                    "font-src 'self' https://fonts.gstatic.com",
                    "img-src 'self' data: blob:",
                    "connect-src 'self' ws: wss:",
                    "frame-ancestors 'none'",
                    "base-uri 'self'",
                  ].join("; ")
                : "",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
