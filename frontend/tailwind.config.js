/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#2563EB',
        'risk-low': '#10B981',
        'risk-moderate': '#F59E0B',
        'risk-high': '#EF4444',
        'risk-urgent': '#DC2626',
      },
    },
  },
  plugins: [],
};
