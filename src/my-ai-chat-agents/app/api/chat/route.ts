import { NextRequest } from 'next/server';

export const runtime = 'edge';
export const maxDuration = 60;

const BACKEND_URL = process.env.BACKEND_URL || '';
const BACKEND_PATH = process.env.BACKEND_PATH ?? '/llm/stream';
const NON_STREAMING = String(process.env.BACKEND_NON_STREAMING ?? 'false') === 'true';

type Body = { messages?: Array<{ id?: string; role: string; content: string }>; agent?: string; tools?: string[]; model?: string };

function buildPayload(body: Body, headers: Headers) {
  const messages = body?.messages ?? [];
  const input = messages.at(-1)?.content ?? '';
  const history = messages.slice(0, -1).map((m) => ({ role: m.role, content: m.content }));
  const agent = body?.agent ?? 'default';
  const tools = Array.isArray(body?.tools) ? body.tools : [];
  const realm = headers.get('x-realm') ?? undefined;
  const tenant = headers.get('x-tenant') ?? undefined;
  return { input, history, agent, tools, stream: !NON_STREAMING, meta: { realm, tenant } };
}

function forwardHeaders(req: NextRequest) {
  const allow = ['authorization','x-realm','x-tenant','x-request-id','x-correlation-id','cookie'];
  const out: Record<string, string> = { 'content-type': 'application/json' };
  for (const k of allow) { const v = req.headers.get(k); if (v) out[k] = v; }
  if (process.env.X_API_KEY) out['x-api-key'] = process.env.X_API_KEY;
  return out;
}

function sseFromTextStream(textStream: ReadableStream<Uint8Array>) {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const ts = new TransformStream<Uint8Array, Uint8Array>();
  const writer = ts.writable.getWriter();
  const reader = textStream.getReader();
  (async () => { try { while (true) { const { done, value } = await reader.read(); if (done) break; const chunkText = decoder.decode(value, { stream: true }); await writer.write(encoder.encode(`data: ${JSON.stringify({ content: chunkText })}\n\n`)); } await writer.write(encoder.encode(`data: [DONE]\n\n`)); } catch (e) { await writer.write(encoder.encode(`data: ${JSON.stringify({ error: String(e) })}\n\n`)); } finally { await writer.close(); } })();
  return new Response(ts.readable, { headers: { 'Content-Type': 'text/event-stream' } });
}

function sseFromSingleText(text: string) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({ start(controller) { controller.enqueue(encoder.encode(`data: ${JSON.stringify({ content: text })}\n\n`)); controller.enqueue(encoder.encode(`data: [DONE]\n\n`)); controller.close(); } });
  return new Response(stream, { headers: { 'Content-Type': 'text/event-stream' } });
}

export async function POST(req: NextRequest) {
  if (!BACKEND_URL) { return new Response(JSON.stringify({ error: 'BACKEND_URL not configured' }), { status: 500, headers: { 'Content-Type': 'application/json' } }); }
  const raw = await req.text(); let body: Body = {}; try { body = raw ? JSON.parse(raw) as Body : {}; } catch {}
  const payload = buildPayload(body, req.headers);
  const url = `${BACKEND_URL}${BACKEND_PATH}`;
  const headers = forwardHeaders(req);
  const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload) });
  const ct = res.headers.get('content-type') || '';
  if (!NON_STREAMING && ct.includes('text/event-stream') && res.body) return new Response(res.body, { status: res.status, headers: { 'Content-Type': 'text/event-stream' } });
  if (!NON_STREAMING && res.body && (ct.startsWith('text/') || ct === '')) return sseFromTextStream(res.body);
  const text = ct.includes('application/json') ? JSON.stringify(await res.json()) : await res.text();
  return sseFromSingleText(text);
}
