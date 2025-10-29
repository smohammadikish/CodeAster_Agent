// next.config.ts
// Disable type checking for this config file (to avoid red lines)
// @ts-nocheck

/** @type {import('next').NextConfig} */
const nextConfig = {
  // ✅ Disable Turbopack and use Webpack instead
  experimental: {
    turbo: false,
  },

  // ✅ Keep Webpack active
  webpack(config) {
    return config;
  },
};

export default nextConfig;