# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import json
import logging
import pii_redacter
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from semantic_kernel_orchestrator import SemanticKernelOrchestrator
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AzureAIAgent
from utils import get_azure_credential
from aoai_client import AOAIClient, get_prompt
from azure.search.documents import SearchClient

from typing import List, Dict, Any

# ---- Multi-index support helpers -------------------------------------------------

def _parse_search_index_names() -> List[str]:
    """Read SEARCH_INDEX_NAMES (comma-separated) or fallback to SEARCH_INDEX_NAME."""
    raw = os.environ.get("SEARCH_INDEX_NAMES") or os.environ.get("SEARCH_INDEX_NAME")
    if not raw:
        return []
    if "," in raw:
        return [n.strip() for n in raw.split(",") if n.strip()]
    return [raw.strip()]

class MultiIndexSearch:
    """Minimal adapter exposing a SearchClient-like .search() over multiple indexes.
    It calls each index and merges the top results by score.
    """
    def __init__(self, endpoint: str, index_names: List[str], credential):
        from azure.search.documents import SearchClient
        self.clients = [SearchClient(endpoint=endpoint, index_name=name, credential=credential) for name in index_names]
        self.index_names = index_names

    def _score(self, item: Any) -> float:
        try:
            # SearchResult behaves like a dict
            return float(item.get("@search.score", 0.0))
        except Exception:
            try:
                return float(item["@search.score"])  # type: ignore[index]
            except Exception:
                return 0.0

    def search(self, *args, **kwargs):
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
        # sort by score desc if available
        try:
            merged.sort(key=self._score, reverse=True)
        except Exception:
            pass
        return merged[:top]

# Initialize the Azure Search client (multi-index aware)
_search_endpoint = os.environ.get("SEARCH_ENDPOINT")
_index_names = _parse_search_index_names()
_credential = get_azure_credential()
if not _index_names:
    raise ValueError("No search index configured. Set SEARCH_INDEX_NAME or SEARCH_INDEX_NAMES.")
if len(_index_names) == 1:
    search_client = SearchClient(endpoint=_search_endpoint, index_name=_index_names[0], credential=_credential)
else:
    search_client = MultiIndexSearch(endpoint=_search_endpoint, index_names=_index_names, credential=_credential)
print(f"Search client initialized for indexes: {', '.join(_index_names)}")

# RAG AOAI client:
rag_client = AOAIClient(
    endpoint=os.environ.get("AOAI_ENDPOINT"),
    deployment=os.environ.get("AOAI_DEPLOYMENT"),
    use_rag=True,
    search_client=search_client
)
print("RAG client initialized.")

# Extract-utterances AOAI client:
extract_prompt = get_prompt("extract_utterances.txt")
extract_client = AOAIClient(
    endpoint=os.environ.get("AOAI_ENDPOINT"),
    deployment=os.environ.get("AOAI_DEPLOYMENT"),
    system_message=extract_prompt
)

# PII:
PII_ENABLED = os.environ.get("PII_ENABLED", "false").lower() == "true"
print(f"PII_ENABLED: {PII_ENABLED}")


# Fallback function (RAG) definition:
def fallback_function(
    query: str,
    language: str,
    id: int
) -> str:
    """
    Call RAG client for grounded chat completion.
    """
    if PII_ENABLED:
        # Redact PII:
        query = pii_redacter.redact(
            text=query,
            id=id,
            language=language,
            cache=True
        )

    return rag_client.chat_completion(query)


# Function to handle processing and orchestrating a chat message with utterance extraction, fallback handling, and PII redaction
async def orchestrate_chat(
    message: str,
    history: list[ChatMessage],
    orchestrator: SemanticKernelOrchestrator,
    chat_id: int
) -> tuple[list[str], bool]:

    responses = []
    need_more_info = False

    # Reshaping system input into proper backend format
    task = f"query: {message}"

    history_str = ", ".join(f"{msg.role} - {msg.content}" for msg in history)
    if history_str:
        task = f"query: {message}, {history_str}"

    print(f"Processing message: {task} with chat_id: {chat_id}")
    try:
        # Handle PII redaction if enabled
        if PII_ENABLED:
            print(f"Redacting PII for message: {task} with chat_id: {chat_id}")
            task = pii_redacter.redact(
                text=task,
                id=chat_id,
                cache=True
            )

        try:
            # Try semantic kernel orchestration first
            orchestrator = app.state.orchestrator
            response, need_more_info = await orchestrator.process_message(task)

            if isinstance(response, dict) and response.get("error"):
                # If semantic kernel fails, use fallback
                print(f"Semantic kernel failed, using fallback for: {message}")
                response = fallback_function(
                    message,
                    "en",  # Assuming English for simplicity, adjust as needed
                    chat_id
                )
            responses.append(response)

        except Exception as e:
            logging.error(f"Error processing utterance: {e}")
            responses.append("I encountered an error processing part of your message.")

    except Exception as e:
        logging.error(f"Error in message processing: {e}")
        responses = ["I apologize, but I'm having trouble processing your request. Please try again."]

    finally:
        # Clean up PII cache if enabled
        if PII_ENABLED:
            pii_redacter.remove(id=chat_id)

    return responses, need_more_info


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup app
    try:
        logging.basicConfig(level=logging.WARNING)

        print("Setting up Azure credentials and client...")
        print(f"Using PROJECT_ENDPOINT: {PROJECT_ENDPOINT}")
        print(f"Using MODEL_NAME: {MODEL_NAME}")

        async with DefaultAzureCredential(exclude_interactive_browser_credential=False) as creds:
            async with AzureAIAgent.create_client(credential=creds, endpoint=PROJECT_ENDPOINT) as client:
                orchestrator = SemanticKernelOrchestrator(
                    client,
                    MODEL_NAME,
                    PROJECT_ENDPOINT,
                    AGENT_IDS,
                    fallback_function,
                    3
                )
                await orchestrator.create_agent_group_chat()

                # Store in app state
                app.state.creds = creds
                app.state.client = client
                app.state.orchestrator = orchestrator

                # Yield control back to FastAPI lifespan
                yield

    except Exception as e:
        logging.error(f"Error during setup: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=os.path.join(DIST_DIR, "assets")), name="assets")


# In order to test uvicorn app locally:
# 1) run `npm run build` in the frontend directory to generate the static files
# 2) move the `dist` directory to `src/backend/src/`
@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(DIST_DIR, "index.html"))


# Define the chat endpoint
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Grab the orchestrator from app state and orchestrate chat message
        orchestrator = app.state.orchestrator
        # pass in message and history
        responses, need_more_info = await orchestrate_chat(request.message, request.history, orchestrator, chat_id=0)
        print("[APP]: need_more_info:", need_more_info)
        return JSONResponse(
            content={
                "messages": responses,
                "need_more_info": need_more_info
            }, status_code=200)

    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}")
        return JSONResponse(
            content={"error": "An unexpected error occurred"},
            status_code=500
        )
