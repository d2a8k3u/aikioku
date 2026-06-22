import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      colors: {
        copper: {
          DEFAULT: '#B87333',
          50: '#FBF5EF',
          100: '#F5E6D6',
          200: '#EBC9A8',
          300: '#E1AC7A',
          400: '#D78F4C',
          500: '#B87333',
          600: '#9A5F2B',
          700: '#7C4B23',
          800: '#5E371B',
          900: '#402313',
        },
        dark: {
          50: '#F7F8F8',
          100: '#E2E4E7',
          200: '#D0D6E0',
          300: '#8A8F98',
          400: '#62666D',
          500: '#3E3E44',
          600: '#28282C',
          700: '#191A1B',
          800: '#0F1011',
          900: '#08090A',
          950: '#010102',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'monospace'],
      },
      letterSpacing: {
        'display-xl': '-1.584px',
        'display-lg': '-1.408px',
        display: '-1.056px',
        heading: '-0.704px',
        subheading: '-0.288px',
        feature: '-0.24px',
        body: '-0.165px',
      },
      borderRadius: {
        micro: '2px',
        standard: '4px',
        comfortable: '6px',
        card: '8px',
        panel: '12px',
        large: '22px',
      },
      boxShadow: {
        surface: 'rgba(255,255,255,0.05) 0px 0px 0px 1px',
        elevated: 'rgba(0,0,0,0.4) 0px 2px 4px',
        dialog:
          'rgba(0,0,0,0) 0px 8px 2px, rgba(0,0,0,0.01) 0px 5px 2px, rgba(0,0,0,0.04) 0px 3px 2px, rgba(0,0,0,0.07) 0px 1px 1px, rgba(0,0,0,0.08) 0px 0px 1px',
      },
    },
  },
  plugins: [],
};

export default config;
