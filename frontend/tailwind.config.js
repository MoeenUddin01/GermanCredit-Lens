/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: "#121214",
          light: "#1A1A1E",
          card: "#1E1E24",
          border: "rgba(255,255,255,0.08)",
        },
        neon: {
          DEFAULT: "#CCFF00",
          hover: "#B8E600",
          glow: "rgba(204,255,0,0.25)",
          muted: "rgba(204,255,0,0.1)",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      boxShadow: {
        "3d": "0 50px 100px -20px rgba(0,0,0,0.7), 0 30px 60px -30px rgba(0,0,0,0.5)",
        glass: "0 8px 32px rgba(0,0,0,0.4)",
        "neon-glow": "0 0 40px rgba(204,255,0,0.15)",
      },
      backdropBlur: {
        glass: "24px",
      },
    },
  },
  plugins: [],
}
