import react from "@vitejs/plugin-react";
import { loadEnv } from "vite";
import { defineConfig } from "vitest/config";

export default defineConfig(({ mode }) => {
  const environment = loadEnv(mode, ".", "VITE_");
  const apiBaseUrl = process.env.VITE_API_BASE_URL ?? environment.VITE_API_BASE_URL ?? "";
  return {
    plugins: [
      react(),
      {
        name: "fhp-api-identity",
        transformIndexHtml: {
          order: "pre",
          handler: () => [{
            tag: "meta",
            attrs: { name: "fhp-api-base", content: apiBaseUrl },
            injectTo: "head",
          }],
        },
      },
    ],
    server: {
      host: "127.0.0.1",
      port: 5173,
      proxy: {
        "/api": {
          target: "http://127.0.0.1:8000",
          changeOrigin: false,
        },
      },
    },
    test: {
      environment: "jsdom",
      setupFiles: "./src/test/setup.ts",
      css: true,
    },
  };
});
