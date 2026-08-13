import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Minimal production image: infra/dashboard.Dockerfile copies only
  // .next/standalone + .next/static + public/, not the full node_modules.
  // Skipped on Vercel (which sets VERCEL=1 in every build automatically) --
  // Vercel's own builder packages serverless functions from the normal
  // build output and doesn't support standalone mode's shape; forcing it
  // there is what produced the missing `.next/next-server.js.nft.json`
  // build error. Railway's deploy (railway.json) only builds the API, not
  // this Dockerfile, so nothing else needs this set unconditionally.
  output: process.env.VERCEL ? undefined : "standalone",
};

export default nextConfig;
