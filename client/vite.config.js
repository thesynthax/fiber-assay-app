import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxy all /upload and /preview requests to FastAPI
      '/upload': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/preview': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/convert': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Optionally proxy other API routes:
      // '/api': { target: 'http://localhost:8000', changeOrigin: true },
    }
  },
})
