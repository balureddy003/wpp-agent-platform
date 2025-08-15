'use client';

type AgentToolPanelProps = {
  agents: string[];
  tools: string[];
  agent: string;
  setAgent: (v: string) => void;
  toolState: Record<string, boolean>;
  setToolState: (v: Record<string, boolean> | ((s: Record<string, boolean>) => Record<string, boolean>)) => void;
  realm: string;
  setRealm: (v: string) => void;
  tenant: string;
  setTenant: (v: string) => void;
};

export default function AgentToolPanel(props: AgentToolPanelProps) {
  const { agents, tools, agent, setAgent, toolState, setToolState, realm, setRealm, tenant, setTenant } = props;

  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="grid">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <div className="sectionTitle">Runtime Config</div>
          <div className="help">Choose an agent, toggle tools, and set realm/tenant headers.</div>
        </div>
        <div className="row">
          <label className="sectionTitle" style={{ width: 80 }}>Agent</label>
          <select className="select" value={agent} onChange={(e) => setAgent(e.target.value)}>
            {agents.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
        </div>

        <div>
          <div className="sectionTitle" style={{ marginBottom: 6 }}>Tools</div>
          <div className="row" style={{ flexWrap: 'wrap' }}>
            {tools.map((t) => (
              <label key={t} style={{ display: 'inline-flex', alignItems: 'center', border: '1px solid var(--border)', borderRadius: 10, padding: '6px 10px', marginRight: 6, marginBottom: 6, background: 'white' }}>
                <input
                  type="checkbox"
                  className="checkbox"
                  checked={!!toolState[t]}
                  onChange={(e) => setToolState((prev) => ({ ...prev, [t]: e.target.checked }))}
                />
                {t}
              </label>
            ))}
          </div>
        </div>

        <div className="row">
          <label className="sectionTitle" style={{ width: 80 }}>Realm</label>
          <input className="input" placeholder="ikea-prod" value={realm} onChange={(e) => setRealm(e.target.value)} />
          <label className="sectionTitle" style={{ width: 80 }}>Tenant</label>
          <input className="input" placeholder="storeA" value={tenant} onChange={(e) => setTenant(e.target.value)} />
        </div>
      </div>
    </div>
  );
}
