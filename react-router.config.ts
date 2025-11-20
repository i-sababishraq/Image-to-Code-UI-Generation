import type { Config } from "@react-router/dev/config";

export default {
  // Re-enable SSR so API routes (/api/*) and actions run on the dev server.
  ssr: true,
  routeDiscovery: {
    mode: "initial",
  },
} satisfies Config;
