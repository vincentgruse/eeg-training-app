import { defineConfig } from 'vite'
import electron from 'vite-plugin-electron'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [
    react(),
    electron([
      {
        // Main process entry point
        entry: 'electron/main.ts',
        vite: {
          build: {
            outDir: 'dist-electron',
            rollupOptions: {
              external: ['brainflow', 'koffi']
            }
          },
        },
      },
      {
        // Preload script entry point
        entry: 'electron/preload.ts',
        onstart(options) {
          options.reload()
        },
        vite: {
          build: {
            outDir: 'dist-electron',
            rollupOptions: {
              external: ['brainflow', 'koffi']
            }
          },
        },
      }
    ]),
  ],
})