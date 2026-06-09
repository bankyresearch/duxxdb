import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In dev, proxy the two backend APIs so the SPA can call them same-origin
// (no CORS). For production, set VITE_CONTROL_URL / VITE_STUDIO_URL to the real
// origins (and enable CORS on the backends, or serve them behind one proxy).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api/control': {
        target: 'http://localhost:7070',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api\/control/, ''),
      },
      '/api/studio': {
        target: 'http://localhost:7072',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api\/studio/, '/studio'),
      },
    },
  },
})
