'use client';

import './globals.css';
import { useMemo, useState } from 'react';
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import AgentToolPanel from './components/AgentToolPanel';

const AGENTS = ['PlannerAgent', 'DiagnoserAgent', 'OperatorAgent', 'ERPAgent'];
const TOOLS = ['web_search', 'kb_lookup', 'get_order_status', 'create_ticket'];

export default function Page() {
  const [agent, setAgent] = useState(AGENTS[0]);
  const [toolState, setToolState] = useState<Record<string, boolean>>(
    Object.fromEntries(TOOLS.map((t) => [t, t === 'web_search' || t === 'kb_lookup']))
  );
  const [realm, setRealm] = useState('ikea-prod');
  const [tenant, setTenant] = useState('storeA');

  const enabledTools = useMemo(() => Object.entries(toolState).filter(([,v]) => v).map(([k]) => k), [toolState]);

  const transport = useMemo(() => new DefaultChatTransport({
    api: '/api/chat',
    headers: { 'x-realm': realm, 'x-tenant': tenant },
    body: { agent, tools: enabledTools }
  }), [agent, enabledTools, realm, tenant]);

  const [input, setInput] = useState('');
  const { messages, status, sendMessage, stop } = useChat({ transport });

  return (
    <main className="container">
      <div className="header">
        <div className="h1">Agents Chat</div>
        <div className="badge">{status === 'streaming' ? 'Streaming' : 'Idle'}</div>
      </div>

      <AgentToolPanel
        agents={AGENTS}
        tools={TOOLS}
        agent={agent}
        setAgent={setAgent}
        toolState={toolState}
        setToolState={setToolState}
        realm={realm}
        setRealm={setRealm}
        tenant={tenant}
        setTenant={setTenant}
      />

      <div className="card" style={{ minHeight: 300 }}>
        <div className="list">
          {messages.map((m) => (
            <div key={m.id} style={{ textAlign: m.role === 'user' ? 'right' : 'left' }}>
              <div className={`bubble ${m.role}`}>{m.content}</div>
            </div>
          ))}
          {messages.length === 0 && (
            <div className="help">Tip: set your agent/tools above. This UI sends them in the request body and forwards realm/tenant headers.</div>
          )}
        </div>
      </div>

      <form onSubmit={(e) => { e.preventDefault(); if (input.trim()) { sendMessage(input); setInput(''); } }} style={{ marginTop: 12 }} className="row">
        <input className="input" placeholder={status === 'streaming' ? 'Generating…' : 'Type a message'} value={input} onChange={(e) => setInput(e.target.value)} />
        {status === 'streaming' ? (
          <button type="button" onClick={() => stop()} className="button secondary">Stop</button>
        ) : (
          <button className="button">Send</button>
        )}
      </form>
    </main>
  );
}
