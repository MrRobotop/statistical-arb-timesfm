import type { Config } from 'tailwindcss';

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bloomberg: {
          bg: '#000000',
          text: '#FFB800',       // Brighter Amber
          green: '#00FF00',      // Neon Green
          red: '#FF4444',        // Brighter Red
          blue: '#00FFFF',       // Cyan/Blue
          gray: '#888888',       // Lighter gray for better contrast
        }
      },
      fontFamily: {
        mono: ['"Courier New"', 'Courier', 'monospace'],
      }
    },
  },
  plugins: [],
} satisfies Config;