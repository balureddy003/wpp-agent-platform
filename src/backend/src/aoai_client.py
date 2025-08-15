# Copyright (c) Microsoft.
# Licensed under the MIT License.

import logging
import json
import os
import re
from typing import Callable, List, Dict, Any, Optional
from pathlib import Path
from urllib.parse import urlparse

from openai import AzureOpenAI
from azure.core.credentials import TokenCredential
from azure.identity import get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery

from utils import get_azure_credential
from search_multi import MultiIndexSearch

# ---------------------------------------------------------------------
# Grounding prompt (file-backed with a safe built-in fallback)
# ---------------------------------------------------------------------

DEFAULT_RAG_GROUNDING_PROMPT = (
    "Use the SOURCES below to answer the USER QUERY.\n"
    "Only use information from SOURCES. If the answer is not present, reply \"I don't know\".\n\n"
    "USER QUERY:\n{query}\n\nSOURCES:\n{sources}\n"
)

def get_prompt(
    prompt: str,
    path: str = "prompts/"
) -> str:
    """
    Load prompt text. Tries several common locations and falls back to a built-in default.
    """
    logger = logging.getLogger(__name__)
    candidates = []
    if path:
        candidates.append(Path(path) / prompt)

    here = Path(__file__).resolve().parent
    candidates.append(here / "prompts" / prompt)
    candidates.append(Path.cwd() / "src" / "prompts" / prompt)
    candidates.append(Path.cwd() / "prompts" / prompt)

    for p in candidates:
        try:
            with open(p, "r", encoding="utf-8") as fp:
                return fp.read()
        except FileNotFoundError:
            continue

    logger.warning(f"Prompt file not found: {prompt}; using built-in default.")
    return DEFAULT_RAG_GROUNDING_PROMPT

RAG_GROUNDING_PROMPT = get_prompt("rag_grounding.txt")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _get_first(doc: Any, keys: List[str], default: str = "") -> str:
    """Return the first non-empty value for any of the candidate keys."""
    for k in keys:
        try:
            if isinstance(doc, dict) and doc.get(k):
                v = doc.get(k)
                return v if isinstance(v, str) else str(v)
            v = getattr(doc, k, None)
            if v:
                return v if isinstance(v, str) else str(v)
            if hasattr(doc, "__getitem__"):  # some SDK objects
                v = doc[k]
                if v:
                    return v if isinstance(v, str) else str(v)
        except Exception:
            continue
    return default


def _escape_odata_value(v: str) -> str:
    """Escape single quotes for OData string literal."""
    return (v or "").replace("'", "''")


def _item_to_dict(item: Any) -> Dict[str, Any]:
    """Best-effort: turn a Search result into a plain dict of fields."""
    try:
        if isinstance(item, dict):
            return dict(item)
        if hasattr(item, "keys"):
            return {k: item[k] for k in item.keys()}
        out = {}
        for k in dir(item):
            if k.startswith("_"):
                continue
            try:
                v = getattr(item, k)
                if callable(v):
                    continue
                out[k] = v
            except Exception:
                continue
        return out
    except Exception:
        return {}


def _format_field_dump(d: Dict[str, Any], max_pairs: int = 40, max_val_len: int = 400) -> str:
    """Render key/value pairs as markdown list, trimming long values."""
    skip = {"@search.score", "text_vector"}
    pairs = []
    for k, v in d.items():
        if k in skip:
            continue
        try:
            s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
        except Exception:
            s = str(v)
        s = s.strip()
        if len(s) > max_val_len:
            s = s[:max_val_len] + "…"
        pairs.append(f"- **{k}**: {s}")
        if len(pairs) >= max_pairs:
            break
    return "\n".join(pairs)


def parse_structured_filters(query: str) -> (Dict[str, str], str):
    """
    Extract simple field filters from the user query.

    Supported forms:
      - "collection: Case record_id: 66bc...56d2"
      - "collection=Case, record_id=...."
      - "case id 66bc...56d2" (maps to record_id)
      - A bare 24-hex string → record_id
    Returns (filters_dict, residual_text)
    """
    q = query.strip()
    filters: Dict[str, str] = {}

    # 1) key[:=]value pairs
    pair_re = re.compile(r"(?P<key>[A-Za-z_][\w\-]*)\s*[:=]\s*(?P<val>'[^']*'|\"[^\"]*\"|[^\s,]+)")
    consumed = []
    for m in pair_re.finditer(q):
        key = m.group("key").strip()
        val = m.group("val").strip().strip('\'"')
        filters[key] = val
        consumed.append(m.span())

    # 2) "case id <hex>" → record_id
    m2 = re.search(r"(case\s*id|case\s*#|id)\s*[:=]?\s*([0-9a-fA-F]{24})", q)
    if m2 and "record_id" not in filters:
        filters["record_id"] = m2.group(2)
        consumed.append(m2.span())

    # 3) bare 24-hex → record_id
    m3 = re.search(r"\b([0-9a-fA-F]{24})\b", q)
    if m3 and "record_id" not in filters:
        filters["record_id"] = m3.group(1)
        consumed.append(m3.span())

    # residual = query with consumed spans removed
    if consumed:
        consumed.sort()
        pieces = []
        last = 0
        for a, b in consumed:
            pieces.append(q[last:a])
            last = b
        pieces.append(q[last:])
        residual = " ".join("".join(pieces).split())
    else:
        residual = q

    return filters, residual


# ---------------------------------------------------------------------
# AOAI Client with optional multi-index RAG
# ---------------------------------------------------------------------

class AOAIClient(AzureOpenAI):
    """
    AzureOpenAI wrapper with function-calling and RAG support.
    Supports either a single SearchClient or a MultiIndexSearch helper.
    """

    def __init__(
        self,
        endpoint: str,
        deployment: str,
        api_version: str = "2023-12-01-preview",
        scope: str = "https://cognitiveservices.azure.com/.default",
        azure_credential: Optional[TokenCredential] = None,
        system_message: Optional[str] = None,
        function_calling: bool = False,
        tools: Optional[list] = None,
        functions: Optional[Dict[str, Callable]] = None,
        return_functions: bool = False,
        use_rag: bool = False,
        search_client: Optional[SearchClient] = None,
        multi_search: Optional[MultiIndexSearch] = None,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        if not azure_credential:
            azure_credential = get_azure_credential()

        token_provider = get_bearer_token_provider(azure_credential, scope)
        AzureOpenAI.__init__(
            self,
            api_version=api_version,
            azure_ad_token_provider=token_provider,
            azure_endpoint=endpoint,
        )

        # Function-calling:
        self.function_calling = function_calling
        self.tools = tools or []
        self.functions = functions or {}
        self.return_functions = return_functions

        # RAG:
        self.use_rag = use_rag
        # prefer explicit multi-search if passed
        self.search_client = multi_search if multi_search is not None else search_client

        # Optional allow-lists (env):
        self.allowed_sources = set(
            s.strip().lower()
            for s in os.getenv("RAG_ALLOWED_SOURCES", "").split(",")
            if s.strip()
        )
        self.allowed_domains = set(
            d.strip().lower()
            for d in os.getenv("RAG_ALLOWED_DOMAINS", "").split(",")
            if d.strip()
        )

        # For UI/debug
        self.last_sources: List[str] = []
        self.last_debug: Dict[str, Any] = {}

        # General:
        self.deployment = self.model_name = deployment
        self.api_version = api_version
        self.chat_api = True
        self.messages: List[Dict[str, Any]] = []

        if system_message:
            self.messages = [{"role": "system", "content": system_message}]

    # -----------------------------------------------------------------
    # Function calling path
    # -----------------------------------------------------------------
    def call_functions(
        self,
        language: Optional[str],
        id: Optional[str]
    ) -> list:
        response = self.chat.completions.create(
            model=self.deployment,
            messages=self.messages,
            tools=self.tools,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        self.messages.append(response_message)
        self.logger.info(f"Model response: {response_message}")

        function_responses = []
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(tool_call.function.arguments or "{}")
                except Exception:
                    function_args = {}
                self.logger.info(f"Function call: {function_name}")
                self.logger.info(f"Function arguments: {function_args}")

                if function_name in self.functions:
                    try:
                        func_input = next(iter(function_args.values()))
                    except StopIteration:
                        func_input = None
                    func = self.functions[function_name]
                    func_response = func(func_input, language, id)
                else:
                    func_response = json.dumps({"error": "Unknown function"})

                function_responses.append(func_response)
                self.logger.info(f"Function response: {str(func_response)}")
                self.messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(func_response)
                })
        else:
            self.logger.info("No tool calls made by model.")

        return function_responses

    # -----------------------------------------------------------------
    # RAG helpers
    # -----------------------------------------------------------------
    def _allow_row(self, source: str, url: str) -> bool:
        """Env-driven filters: RAG_ALLOWED_SOURCES and RAG_ALLOWED_DOMAINS."""
        ok_src = True
        ok_dom = True
        if self.allowed_sources:
            ok_src = (source or "").lower() in self.allowed_sources
        if self.allowed_domains and url:
            try:
                host = (urlparse(url).hostname or "").lower()
                ok_dom = any(host.endswith(d) for d in self.allowed_domains)
            except Exception:
                ok_dom = False
        return ok_src and ok_dom

    # -----------------------------------------------------------------
    # RAG grounding prompt
    # -----------------------------------------------------------------
    def generate_rag_prompt(self, query: str) -> str:
        """
        Build a grounding prompt. First tries structured filters (exact lookups),
        then hybrid semantic search. Avoids hard-coded $select to be multi-index safe.
        """
        if not self.search_client:
            self.logger.warning("RAG enabled but no SearchClient; falling back to empty sources.")
            self.last_sources = []
            self.last_debug = {"path": "no_search_client"}
            return RAG_GROUNDING_PROMPT.format(query=query, sources="")

        top_k = int(os.getenv("RAG_TOP_K", "5"))

        # -------- 1) Structured filter path (generic) --------
        filters, residual = parse_structured_filters(query)
        if filters:
            try:
                rows: List[str] = []
                urls: List[str] = []
                used = 0
                if isinstance(self.search_client, MultiIndexSearch):
                    print(f"[aoai-client] RAG structured (multi) filters={filters} residual='{residual}'")
                    results = self.search_client.search_by_filters(
                        filters, top_k=top_k, select=None, search_text=residual or "*"
                    )
                else:
                    flt = " and ".join([f"{k} eq '{_escape_odata_value(v)}'" for k, v in filters.items()])
                    print(f"[aoai-client] RAG structured (native) filter=\"{flt}\" residual='{residual}'")
                    results = self.search_client.search(
                        search_text=(residual or "*"),
                        top=top_k,
                        filter=flt
                        # no select → all retrievable fields
                    )

                for doc in results:
                    raw = doc.get("__raw__") if isinstance(doc, dict) else None
                    doc_map = _item_to_dict(raw or doc)

                    title = _get_first(doc_map, ["title", "page_title", "name", "heading", "subject", "Title"]) or ""
                    url   = _get_first(doc_map, ["url", "source_url", "uri", "link", "permalink", "SourceUrl"]) or ""
                    source = _get_first(doc_map, ["source", "Source", "source_type"]) or ""
                    content = _get_first(doc_map, ["chunk", "text", "content", "body", "passage", "abstract", "description", "snippet"]) or ""

                    # Build a rich details section from all returned fields
                    field_dump = _format_field_dump(doc_map)
                    if not field_dump and not content:
                        try:
                            field_dump = json.dumps(doc_map, ensure_ascii=False, default=str)[:1200] + "…"
                        except Exception:
                            field_dump = str(doc_map)

                    if len(content) > 1200:
                        content = content[:1200] + "…"

                    parts = []
                    if title: parts.append(f"TITLE: {title}")
                    if url:   parts.append(f"URL: {url}")
                    if "collection" in filters: parts.append(f"COLLECTION: {filters['collection']}")
                    for k, v in filters.items():
                        if k != "collection":
                            parts.append(f"{k.upper()}: {v}")
                    if content:
                        parts.append(f"CONTENT:\n{content}")
                    if field_dump:
                        parts.append(f"FIELDS:\n{field_dump}")

                    if parts:
                        rows.append("\n".join(parts))
                        if url:
                            urls.append(url)
                        used += 1

                self.last_sources = urls[:top_k]
                self.last_debug = {
                    "path": "structured_filters",
                    "hits": used,
                    "filters": filters,
                    "residual": residual
                }
                print(f"[aoai-client] RAG structured gathered rows={used} sources={len(self.last_sources)}")
                if used > 0:
                    return RAG_GROUNDING_PROMPT.format(query=query, sources="\n=================\n".join(rows))
            except Exception as e:
                print(f"[aoai-client] structured filter search failed: {type(e).__name__}: {e}")

        # -------- 2) Hybrid (vector + keyword) similarity search --------
        results = None
        try:
            vq = VectorizableTextQuery(
                text=query,
                k_nearest_neighbors=max(10, top_k),
                fields="text_vector"
            )
            print("[aoai-client] RAG native vector search top_k={}".format(top_k))
            results = self.search_client.search(
                search_text=query,
                vector_queries=[vq],
                top=top_k
            )
        except Exception as e:
            print(f"[aoai-client] vector search failed → keyword-only: {e}")
            try:
                results = self.search_client.search(search_text=query, top=top_k, query_type="full")
            except Exception as e2:
                print(f"[aoai-client] keyword search also failed: {e2}")
                self.last_sources = []
                self.last_debug = {"path": "hybrid_failed", "err": str(e2)}
                return RAG_GROUNDING_PROMPT.format(query=query, sources="")

        rows: List[str] = []
        urls: List[str] = []
        used = 0
        for doc in results:
            title = _get_first(doc, ["title", "page_title", "name", "heading", "subject", "Title"]) or ""
            url   = _get_first(doc, ["url", "source_url", "uri", "link", "permalink", "SourceUrl"]) or ""
            source = _get_first(doc, ["source", "Source", "source_type"]) or ""
            content = _get_first(doc, ["chunk", "text", "content", "body", "passage", "abstract", "description", "snippet"]) or ""

            if not self._allow_row(source, url):
                continue

            if len(content) > 1200:
                content = content[:1200] + "…"

            parts = []
            if title: parts.append(f"TITLE: {title}")
            if url:   parts.append(f"URL: {url}")
            if content: parts.append(f"CONTENT:\n{content}")

            if parts:
                rows.append("\n".join(parts))
                if url:
                    urls.append(url)
                used += 1

        self.last_sources = urls[:top_k]
        self.last_debug = {"path": "hybrid", "hits": used}
        print(f"[aoai-client] RAG native gathered rows={used} sources={len(self.last_sources)}")

        sources_formatted = "\n=================\n".join(rows)
        return RAG_GROUNDING_PROMPT.format(query=query, sources=sources_formatted)

    # -----------------------------------------------------------------
    # Chat completion
    # -----------------------------------------------------------------
    def chat_completion(
        self,
        message: str,
        language: Optional[str] = None,
        id: Optional[str] = None
    ) -> Any:
        """
        If RAG is enabled, returns: {"content": <str>, "sources": [<url>...]}.
        Otherwise returns just the assistant content string.
        """
        prompt = self.generate_rag_prompt(message) if self.use_rag else message
        self.messages.append({"role": "user", "content": prompt})

        if self.function_calling:
            function_results = self.call_functions(language=language, id=id)
            if self.return_functions:
                return function_results

        response = self.chat.completions.create(
            model=self.deployment,
            messages=self.messages
        )
        response_message = response.choices[0].message
        self.logger.info(f"Model response: {response_message}")
        self.messages.append(response_message)

        if self.use_rag:
            return {"content": response_message.content, "sources": self.last_sources}
        return response_message.content