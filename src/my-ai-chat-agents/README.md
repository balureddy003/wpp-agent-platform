# Agents Chat (Next.js + Vercel AI SDK)

- Agent picker + tool toggles
- Realm/Tenant headers forwarded to backend
- Edge route adapts UI payload to your API and handles SSE or non-stream

Quickstart:
1) `pnpm install`  2) copy `.env.example` to `.env.local`  3) `pnpm dev`

Env (.env.local):
BACKEND_URL=http://localhost:8000
BACKEND_PATH=/llm/stream
BACKEND_NON_STREAMING=false
