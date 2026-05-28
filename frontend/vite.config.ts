import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '^/translate/': 'http://localhost:8000',
      '^/animation/': 'http://localhost:8000',
      '^/system/':    'http://localhost:8000',
      '^/api/':       'http://localhost:8000',
      '^/realtime/':  { target: 'ws://localhost:8000', ws: true },
    },
  },
})
