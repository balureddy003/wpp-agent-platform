import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// ---------- helpers: payload normalization & sources ----------

function normalizeAssistantContent(payload) {
  try {
    if (typeof payload === "string") return payload;
    if (!payload || typeof payload !== "object") return "";
    if (payload.content) return payload.content;
    if (payload.message) return payload.message;

    if (Array.isArray(payload.messages)) {
      const parts = payload.messages
        .map((m) => {
          if (typeof m === "string") return m;
          if (!m || typeof m !== "object") return "";
          return m.content ?? m.text ?? "";
        })
        .filter(Boolean);
      return parts.join("\n\n");
    }

    if (Array.isArray(payload.responses)) {
      return payload.responses.filter(Boolean).join("\n\n");
    }

    if (payload.choices?.[0]?.message?.content) {
      return payload.choices[0].message.content;
    }

    if (payload.output_text) return payload.output_text;
    if (payload.answer) return payload.answer;
    if (payload.data?.content) return payload.data.content;

    return JSON.stringify(payload, null, 2);
  } catch {
    return "";
  }
}

function extractUrlsFromText(text) {
  if (!text) return [];
  const rx = /\bhttps?:\/\/[^\s)>\]]+/gi;
  const found = text.match(rx) || [];
  const cleaned = found.map((u) => u.replace(/[)\].,]+$/g, ""));
  return Array.from(new Set(cleaned));
}

function normalizeAssistant(payload) {
  const content = normalizeAssistantContent(payload);
  let sources = [];
  try {
    if (Array.isArray(payload?.sources)) sources = payload.sources;
    else if (Array.isArray(payload?.data?.sources)) sources = payload.data.sources;
    else if (Array.isArray(payload?.citations)) sources = payload.citations;
    else if (Array.isArray(payload?.references)) sources = payload.references;
    else if (Array.isArray(payload?.source_urls)) sources = payload.source_urls;
  } catch {}
  if (!sources || sources.length === 0) {
    sources = extractUrlsFromText(content);
  }
  const uniq = Array.from(new Set((sources || []).filter(Boolean)));
  return { content, sources: uniq.slice(0, 8) };
}

// ---------- helpers: rendering ----------

function niceUrl(u) {
  try {
    const { origin, pathname } = new URL(u);
    const short = pathname.length > 40 ? pathname.slice(0, 37).replace(/\/?$/, "") + "…" : pathname || "/";
    return `${origin}${short}`;
  } catch {
    return u;
  }
}

function faviconUrl(u) {
  try {
    const { origin } = new URL(u);
    return `${origin}/favicon.ico`;
  } catch {
    return null;
  }
}

const API_BASE = import.meta.env.DEV ? "" : (import.meta.env.VITE_API_BASE ?? "");

// ---------- component ----------

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState(null);
  const scrollRef = useRef(null);

  const sessionId = useMemo(() => crypto.randomUUID(), []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    }
  }, [messages]);

  async function onSubmit(e) {
    e?.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isSending) return;

    setError(null);
    const userMsg = { role: "user", content: trimmed };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setIsSending(true);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMsg.content,
          language: "en",
          session_id: sessionId,
        }),
      });

      const contentType = res.headers.get("content-type") || "";
      const payload = contentType.includes("application/json") ? await res.json() : await res.text();

      if (!res.ok) {
        const errMsg =
          typeof payload === "string"
            ? payload
            : payload?.detail || payload?.error || JSON.stringify(payload);
        throw new Error(errMsg || `HTTP ${res.status}`);
      }

      const { content: assistantText, sources } = normalizeAssistant(payload);
      setMessages((m) => [...m, { role: "assistant", content: assistantText, sources }]);
    } catch (err) {
      const msg = err?.message || String(err);
      setError(msg);
      setMessages((m) => [...m, { role: "assistant", content: `Error: ${msg}` }]);
    } finally {
      setIsSending(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSubmit();
    }
  }

  function onClear() {
    setMessages([]);
    setError(null);
  }

  return (
    <div className="chat-container mx-auto max-w-3xl p-4">
      <div className="chat-header text-xl font-semibold">Conversational Agent</div>
      <div className="chat-disclaimer text-sm text-gray-500 mb-3">
        This is a demo. Responses may be inaccurate.
      </div>

      <div
        ref={scrollRef}
        className="chat-messages border rounded-md p-3 h-[60vh] overflow-y-auto bg-white/70"
      >
        {messages.length === 0 && (
          <div className="text-gray-500 text-sm">
            Ask a question about Cyclone Rake, website content, or internal data sources.
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`message my-3 ${msg.role}`}>
            <div className="message-header mb-1 text-xs uppercase tracking-wide text-gray-500">
              {msg.role === "user" ? "You" : "Agent"}
            </div>

            <div className="message-content prose prose-sm max-w-none">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  a: (props) => <a {...props} target="_blank" rel="noreferrer" />,
                  ul: (props) => <ul className="list-disc pl-5" {...props} />,
                  ol: (props) => <ol className="list-decimal pl-5" {...props} />,
                  code: ({ inline, className, children, ...rest }) =>
                    inline ? (
                      <code className={className} {...rest}>{children}</code>
                    ) : (
                      <pre className={className}><code {...rest}>{children}</code></pre>
                    ),
                }}
              >
                {msg.content || ""}
              </ReactMarkdown>

              {Array.isArray(msg.sources) && msg.sources.length > 0 && (
                <div className="message-sources mt-3">
                  <div className="message-sources-title text-xs font-semibold mb-1">Sources</div>
                  <ul className="message-sources-list grid gap-2 sm:grid-cols-2">
                    {msg.sources.map((u, j) => (
                      <li key={j} className="border rounded p-2 bg-white/60 hover:bg-white">
                        <a href={u} target="_blank" rel="noreferrer" className="flex items-start gap-2" title={u}>
                          {faviconUrl(u) && (
                            <img
                              src={faviconUrl(u)}
                              alt=""
                              className="w-4 h-4 mt-1 rounded-sm"
                              onError={(e) => { e.currentTarget.style.visibility = "hidden"; }}
                            />
                          )}
                          <span className="text-xs break-all">{niceUrl(u)}</span>
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        ))}

        {isSending && (
          <div className="mt-2 text-sm text-gray-500 animate-pulse">Agent is thinking…</div>
        )}
      </div>

      <form className="chat-input-form mt-3 flex gap-2" onSubmit={onSubmit}>
        <input
          className="chat-input flex-1 border rounded-md px-3 py-2"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Type your message… (Enter to send, Shift+Enter for newline)"
          disabled={isSending}
        />
        <button className="chat-submit-button bg-black text-white px-4 py-2 rounded-md disabled:opacity-50" type="submit" disabled={isSending}>
          {isSending ? "Sending…" : "Send"}
        </button>
        <button className="chat-clear-button border px-3 py-2 rounded-md" type="button" onClick={onClear} disabled={isSending || messages.length === 0} title="Clear conversation">
          Clear
        </button>
      </form>

      {error && <div className="mt-2 text-sm text-red-600">{error}</div>}
    </div>
  );
}