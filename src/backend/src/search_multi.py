# Copyright (c) Microsoft.
# Licensed under the MIT License.

from typing import List, Dict, Any, Optional
import os
import time
import json

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery

from utils import get_azure_credential


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _get(v: Any, key: str, default=""):
    """Tolerant getter that works for dict-like or SDK result objects."""
    try:
        if isinstance(v, dict):
            return v.get(key, default)
        if hasattr(v, key):
            val = getattr(v, key)
            return val if val is not None else default
        if hasattr(v, "__getitem__"):
            return v[key]
    except Exception:
        pass
    return default


def _first(d: Any, keys: List[str], default: str = "") -> str:
    """Return first non-empty string-like value for any key in keys."""
    for k in keys:
        val = _get(d, k, None)
        if val:
            return val if isinstance(val, str) else str(val)
    return default


def _escape_odata_value(v: str) -> str:
    """Escape single quotes for OData string literal."""
    return (v or "").replace("'", "''")


def _build_filter_string(filters: Dict[str, str]) -> str:
    """Build an OData filter from a simple field->value map using 'eq'."""
    parts = []
    for k, v in filters.items():
        if not k or v is None:
            continue
        parts.append(f"{k} eq '{_escape_odata_value(str(v))}'")
    return " and ".join(parts)


class MultiIndexSearch:
    """Run vector/keyword search across multiple Azure AI Search indexes and merge results."""
    is_multi = True

    def __init__(self, endpoint: str, indexes: List[str], credential=None):
        self.endpoint = endpoint
        self.indexes = indexes
        if credential is None:
            key = os.getenv("SEARCH_API_KEY")
            credential = AzureKeyCredential(key) if key else get_azure_credential()
        self.credential = credential
        self.clients: Dict[str, SearchClient] = {
            ix: SearchClient(endpoint=endpoint, index_name=ix, credential=credential)
            for ix in indexes
        }
        print(f"[multi-search {_now()}] init endpoint={endpoint} indexes={indexes}")

    # -----------------------------
    # Hybrid similarity search
    # -----------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        select: Optional[List[str]] = None,
        vector_field: str = "text_vector",
        use_hybrid: bool = True,
    ) -> List[Dict[str, Any]]:
        select = select or ["title", "chunk", "text", "url"]
        per_index_hits: Dict[str, int] = {}
        merged: List[Dict[str, Any]] = []

        for ix, client in self.clients.items():
            try:
                if use_hybrid:
                    vq = VectorizableTextQuery(text=query, k_nearest_neighbors=max(10, top_k), fields=vector_field)
                    results = client.search(
                        search_text=query,
                        vector_queries=[vq],
                        top=top_k,
                        select=select
                    )
                else:
                    results = client.search(
                        search_text=query,
                        top=top_k,
                        select=select,
                        query_type="full"
                    )

                ix_rows = 0
                for doc in results:
                    ix_rows += 1
                    merged.append({
                        "index": ix,
                        "score": float(_get(doc, "@search.score", 0.0)),
                        "title": _first(doc, ["title", "page_title", "name", "Title"]),
                        "url": _first(doc, ["url", "source_url", "uri", "link", "permalink", "SourceUrl"]),
                        "content": _first(doc, ["chunk", "text", "content", "body", "snippet", "description"]),
                    })
                per_index_hits[ix] = ix_rows
            except Exception as e:
                print(f"[multi-search {_now()}] index='{ix}' error={type(e).__name__}: {e}")

        merged.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        out = merged[:top_k]
        print(f"[multi-search {_now()}] query='{query}' indexes={self.indexes} hits_per_index={per_index_hits} returned={len(out)}")
        return out

    # -----------------------------
    # Structured (exact) filter search
    # -----------------------------
    def search_by_filters(
        self,
        filters: Dict[str, str],
        top_k: int = 5,
        select: Optional[List[str]] = None,
        search_text: str = "*",
    ) -> List[Dict[str, Any]]:
        """
        Exact lookup across indexes using OData filter on provided fields.
        select=None → return all retrievable fields so downstream can format details.
        """
        flt = _build_filter_string(filters)
        per_index_hits: Dict[str, int] = {}
        merged: List[Dict[str, Any]] = []

        for ix, client in self.clients.items():
            try:
                res = client.search(
                    search_text=search_text,
                    top=top_k,
                    select=select,   # None → all retrievable fields
                    filter=flt
                )
                cnt = 0
                for doc in res:
                    cnt += 1
                    merged.append({
                        "index": ix,
                        "score": float(_get(doc, "@search.score", 0.0)),
                        "title": _first(doc, ["title", "page_title", "name", "Title"], default=""),
                        "url": _first(doc, ["url", "source_url", "uri", "link", "permalink", "SourceUrl"], default=""),
                        "content": _first(doc, ["chunk", "text", "content", "body", "snippet", "description"], default=""),
                        "record_id": _first(doc, ["record_id"], default=""),
                        "source": _first(doc, ["source", "Source", "source_type"], default=""),
                        "collection": _first(doc, ["collection"], default=""),
                        "__raw__": doc  # keep full doc for downstream formatting
                    })
                per_index_hits[ix] = cnt
            except Exception as e:
                print(f"[multi-search {_now()}] filter search index='{ix}' error={type(e).__name__}: {e}")

        merged.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        out = merged[:top_k]
        print(f"[multi-search {_now()}] filter='{flt}' hits_per_index={per_index_hits} returned={len(out)} (select={'ALL' if select is None else select})")
        return out


def build_multi_from_env() -> Optional[MultiIndexSearch]:
    names = os.getenv("SEARCH_INDEX_NAMES", "").strip()
    if not names:
        return None
    indexes = [n.strip() for n in names.split(",") if n.strip()]
    if not indexes:
        return None
    endpoint = os.environ["SEARCH_ENDPOINT"]
    return MultiIndexSearch(endpoint=endpoint, indexes=indexes)