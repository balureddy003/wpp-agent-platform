# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import json
import importlib
from json import JSONDecodeError
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from azure.search.documents import SearchClient

import pii_redacter
from aoai_client import AOAIClient, get_prompt
from router.router_type import RouterType
from unified_conversation_orchestrator import UnifiedConversationOrchestrator
from utils import get_azure_credential

# Load local .env (same folder as this file)
load_dotenv(Path(__file__).with_name(".env"))

# =============================================================================
# Multi-index helper (kept local to this file)
# =============================================================================

from typing import Any as _Any  # avoid shadowing

def _parse_search_index_names() -> List[str]:
    """
    Read SEARCH_INDEX_NAMES (comma-separated) or fallback to SEARCH_INDEX_NAME.
    """
    raw = os.environ.get("SEARCH_INDEX_NAMES") or os.environ.get("SEARCH_INDEX_NAME")
    if not raw:
        return []
    if "," in raw:
        return [n.strip() for n in raw.split(",") if n.strip()]
    return [raw.strip()]

class MultiIndexSearch:
    """
    Very small helper that fans out a search across multiple SearchClient instances
    and merges the results by @search.score (desc).
    """
    def __init__(self, endpoint: str, index_names: List[str], credential):
        self.clients = [SearchClient(endpoint=endpoint, index_name=name, credential=credential) for name in index_names]
        self.indexes = index_names

    @staticmethod
    def _score(item: _Any) -> float:
        try:
            if isinstance(item, dict) and "@search.score" in item:
                return float(item["@search.score"])
            return float(getattr(item, "@search.score", 0.0))
        except Exception:
            return 0.0

    def search(self, *args, **kwargs):
        """
        Signature compatible with SearchClient.search.
        We’ll merge and then trim to the 'top' requested.
        """
        top = int(kwargs.get("top", 5) or 5)
        merged = []
        for client in self.clients:
            try:
                results = client.search(*args, **kwargs)
                for r in results:
                    merged.append(r)
            except Exception as e:
                print(f"[multi-index] search error on {client.index_name}: {e}")
                continue
        try:
            merged.sort(key=self._score, reverse=True)
        except Exception:
            pass
        return merged[:top]

# =============================================================================
# Static files / app bootstrap
# =============================================================================

DIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "dist"))
print(f"[startup] DIST_DIR: {DIST_DIR}")

app = FastAPI()

# CORS: allow localhost/127.0.0.1 on any port (Vite, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# If you want to serve /assets, uncomment when dist/assets exists
# assets_dir = os.path.join(DIST_DIR, "assets")
# if os.path.isdir(assets_dir):
#     app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

# =============================================================================
# RAG AOAI client (single or multi index)
# =============================================================================

_search_endpoint = os.environ.get("SEARCH_ENDPOINT")
_index_names = _parse_search_index_names()
_credential = get_azure_credential()

if not _search_endpoint:
    raise ValueError("SEARCH_ENDPOINT is not set.")

if not _index_names:
    raise ValueError("No search index configured. Set SEARCH_INDEX_NAME or SEARCH_INDEX_NAMES.")

print(f"[startup] Search endpoint:  {_search_endpoint}")
print(f"[startup] Search indexes:   {_index_names}")

# Build either a single SearchClient or MultiIndexSearch
rag_multi_search: Optional[MultiIndexSearch] = None
rag_single_search: Optional[SearchClient] = None

if len(_index_names) == 1:
    rag_single_search = SearchClient(endpoint=_search_endpoint, index_name=_index_names[0], credential=_credential)
else:
    rag_multi_search = MultiIndexSearch(endpoint=_search_endpoint, index_names=_index_names, credential=_credential)

rag_client = AOAIClient(
    endpoint=os.environ.get("AOAI_ENDPOINT"),
    deployment=os.environ.get("AOAI_DEPLOYMENT"),
    use_rag=True,
    search_client=rag_single_search,
    multi_search=rag_multi_search,   # used when multiple indexes
)

# =============================================================================
# Utterance extraction client (no RAG)
# =============================================================================

extract_prompt = get_prompt("extract_utterances.txt")
extract_client = AOAIClient(
    endpoint=os.environ.get("AOAI_ENDPOINT"),
    deployment=os.environ.get("AOAI_DEPLOYMENT"),
    system_message=extract_prompt
)

# =============================================================================
# PII toggle
# =============================================================================

PII_ENABLED = os.environ.get("PII_ENABLED", "false").lower() == "true"

# =============================================================================
# Fallback = RAG
# =============================================================================

def fallback_function(
    query: str,
    language: str,
    id: int
) -> str:
    if PII_ENABLED:
        query = pii_redacter.redact(text=query, id=id, language=language, cache=True)
    return rag_client.chat_completion(query)

# =============================================================================
# Orchestrator
# =============================================================================

router_type = RouterType(os.environ.get("ROUTER_TYPE", "BYPASS"))
orchestrator = UnifiedConversationOrchestrator(
    router_type=router_type,
    fallback_function=fallback_function
)

# If you want to track sessions, you can increment this per request.
chat_id = 0

def orchestrate_chat(message: str) -> List[str]:
    if PII_ENABLED:
        message = pii_redacter.redact(text=message, id=chat_id, cache=True)

    # Break user message into separate utterances (string or JSON list)
    utterances = extract_client.chat_completion(message)
    print(f"Utterances: {utterances}")

    if not isinstance(utterances, list):
        try:
            utterances = json.loads(utterances)
        except JSONDecodeError:
            if PII_ENABLED:
                pii_redacter.remove(id=chat_id)
            return ['I am unable to respond or participate in this conversation.']

    responses: List[str] = []
    for query in utterances:
        if PII_ENABLED:
            query = pii_redacter.reconstruct(text=query, id=chat_id, cache=True)

        orchestration_response = orchestrator.orchestrate(message=query, id=chat_id)
        print(f"Orchestration response: {orchestration_response}")

        response = None
        if orchestration_response["route"] == "fallback":
            response = orchestration_response["result"]

        elif orchestration_response["route"] == "clu":
            intent = orchestration_response["result"]["intent"]
            entities = orchestration_response["result"]["entities"]
            hooks_module = importlib.import_module("clu_hooks")
            hook_func = getattr(hooks_module, intent)
            response = hook_func(entities)

        elif orchestration_response["route"] == "cqa":
            answer = orchestration_response["result"]["answer"]
            response = answer

        print(f"Parsed response: {response}")
        responses.append(response)

    if PII_ENABLED:
        pii_redacter.remove(id=chat_id)

    return responses

# =============================================================================
# Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def home_page():
    """
    Serve index.html from dist/. If missing, return a small placeholder.
    """
    index_path = os.path.join(DIST_DIR, "index.html")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    # Fallback placeholder
    return HTMLResponse(
        "<!doctype html><html><head><meta charset='utf-8'><title>App</title></head>"
        "<body><h3>dist/index.html not found</h3><p>Build your frontend and place it under src/backend/src/dist</p></body></html>",
        status_code=200
    )

@app.post("/chat")
async def chat(request: Request):
    payload = await request.json()
    message = payload.get("message", "")
    responses = orchestrate_chat(message)
    print(f"responses: {responses}")
    return JSONResponse({"messages": responses})

@app.get("/healthz")
async def healthz():
    return {"ok": True}