/** @type {import('next').NextConfig} */
const nextConfig = {
  // Security: remove X-Powered-By header
  poweredByHeader: false,

  // Build output for Docker
  output: "standalone",

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
