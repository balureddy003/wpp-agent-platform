import { useState } from "react";

function normalizeAssistantContent(payload) {
  try {
    if (typeof payload === "string") return payload;
    if (!payload || typeof payload !== "object") return "";
    if (payload.content) return payload.content;
    if (payload.message) return payload.message;
    if (Array.isArray(payload.messages)) {
      // messages can be ["text", ...] or [{content:"..."}|{text:"..."}|"..."]
      const parts = payload.messages.map((m) => {
        if (typeof m === "string") return m;
        if (!m || typeof m !== "object") return "";
        return m.content ?? m.text ?? "";
      }).filter(Boolean);
      return parts.join("\n\n");
    }
    if (Array.isArray(payload.responses)) return payload.responses.filter(Boolean).join("\n\n");
    if (payload.choices?.[0]?.message?.content) return payload.choices[0].message.content;
    if (payload.output_text) return payload.output_text;
    if (payload.answer) return payload.answer;
    if (payload.data?.content) return payload.data.content;
    // last resort: show JSON so we can see the shape
    return JSON.stringify(payload, null, 2);
  } catch {
    return "";
  }
}

const API_BASE = import.meta.env.DEV ? "" : (import.meta.env.VITE_API_BASE ?? "");

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  async function onSubmit(e) {
    e.preventDefault();
    const userMsg = { role: "user", content: input };
    setMessages((m) => [...m, userMsg]);
    setInput("");

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMsg.content,
          language: "en",
          session_id: crypto.randomUUID()
        })
      });

      const contentType = res.headers.get("content-type") || "";
      const payload = contentType.includes("application/json") ? await res.json() : await res.text();

      if (!res.ok) {
        const errMsg = typeof payload === "string" ? payload : (payload?.detail || payload?.error || JSON.stringify(payload));
        throw new Error(errMsg || `HTTP ${res.status}`);
      }

      const assistantText = normalizeAssistantContent(payload);
      setMessages((m) => [...m, { role: "assistant", content: assistantText }]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `Error: ${err?.message || String(err)}` }
      ]);
    }
  }

  return (
    <div className="chat-container">
      <div className="chat-header">Conversational Agent</div>
      <div className="chat-disclaimer">This is a demo. Responses may be inaccurate.</div>

      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-header">{msg.role === "user" ? "You" : "Agent"}</div>
            <p className="message-content">{msg.content}</p>
          </div>
        ))}
      </div>

      <form className="chat-input-form" onSubmit={onSubmit}>
        <input
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message…"
        />
        <button className="chat-submit-button" type="submit">Send</button>
      </form>
    </div>
  );
}