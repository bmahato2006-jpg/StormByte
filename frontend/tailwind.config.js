/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        command: {
          bg: "#f8fafc",
          card: "#ffffff",
          cardBorder: "#e2e8f0",
          panel: "#f1f5f9",
          header: "#ffffff",
          accent: "#0284c7",
          accentGlow: "rgba(2, 132, 199, 0.12)",
          amber: "#d97706",
          danger: "#dc2626",
          success: "#16a34a",
          muted: "#64748b",
        },
        cyclone: {
          depression: "#2563eb",
          cyclonic: "#0284c7",
          severe: "#d97706",
          verySevere: "#ea580c",
          extremelySevere: "#dc2626",
          super: "#db2777",
        }
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
};

