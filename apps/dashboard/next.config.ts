import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Minimal production image: infra/dashboard.Dockerfile copies only
  // .next/standalone + .next/static + public/, not the full node_modules.
  output: "standalone",
};

export default nextConfig;
