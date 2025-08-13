// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  base: './',
  plugins: [react()],
  server: {
    proxy: {
      // call fetch('/chat', ...) in the frontend
      '/chat': { target: 'http://127.0.0.1:8000', changeOrigin: true }
    }
  }
})
