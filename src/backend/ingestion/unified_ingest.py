# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Unified, extensible ingestion pipeline for multiple data sources (Web (Cyclopedia + Site),
MongoDB (GrandNode), Airtable), that pushes chunked + embedded text into Azure AI Search.

- Reuses your identity pattern (USE_MI_AUTH + MI_CLIENT_ID or API keys)
- One config, many connectors; per-source index supported
- Shared index schema with `url` field and vector
- Detailed, always-on print logs (no buffering issues)
"""

from __future__ import annotations

import os, re, sys, json, time, hashlib, contextlib, urllib.parse
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Dict, Any, Iterable, Iterator, List, Optional, Protocol
from xml.etree import ElementTree as ET

# --- Optional deps (guarded) -------------------------------------------------
with contextlib.suppress(Exception):
    from bson.objectid import ObjectId   # type: ignore
with contextlib.suppress(Exception):
    from bson.decimal128 import Decimal128  # type: ignore
with contextlib.suppress(Exception):
    from bs4 import BeautifulSoup  # type: ignore
with contextlib.suppress(Exception):
    from pymongo import MongoClient  # type: ignore
with contextlib.suppress(Exception):
    from pyairtable import Table, Api  # type: ignore

# --- Third-party Azure / OpenAI ---------------------------------------------
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchField, SearchFieldDataType, SimpleField, SearchableField,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    AzureOpenAIVectorizer, AzureOpenAIVectorizerParameters, SearchIndex
)
from openai import AzureOpenAI

# --- Constants & toggles -----------------------------------------------------
UA = "wpp-agent-ingestor/1.0 (+github.com/balureddy003/wpp-agent-platform)"
USE_BROWSER = os.getenv("INGEST_USE_BROWSER", "0").lower() not in ("0","false","no","")
REINDEX_ALL = os.getenv("INGEST_REINDEX", "0").lower() not in ("0","false","no","")
ASSET_EXT = {".png",".jpg",".jpeg",".gif",".webp",".svg",".pdf",".zip",".gz",".mp4",".mov",".css",".js",".ico",".woff",".woff2",".ttf"}

DEFAULT_INDEX_NAME = os.environ.get('SEARCH_INDEX_NAME_UNIFIED', 'unified-knowledge')
EMBED_MODEL_NAME   = os.environ.get('EMBEDDING_MODEL_NAME', 'text-embedding-3-small')
EMBED_DEPLOYMENT   = os.environ.get('EMBEDDING_DEPLOYMENT_NAME', 'text-embedding-3-small')
EMBED_DIMS         = int(os.environ.get('EMBEDDING_MODEL_DIMENSIONS', '1536'))

# --- Logging (always on prints) ---------------------------------------------
def dbg(*a):
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[unified-ingest {ts}]", *a, flush=True)
    except Exception:
        try:
            print("[unified-ingest]", *a, flush=True)
        except Exception:
            pass

# --- Identity / OpenAI -------------------------------------------------------
def get_azure_credential():
    use_mi_auth = os.environ.get('USE_MI_AUTH', 'false').lower() == 'true'
    if use_mi_auth:
        mi_client_id = os.environ.get('MI_CLIENT_ID')
        if not mi_client_id:
            raise RuntimeError("USE_MI_AUTH=true but MI_CLIENT_ID is not set")
        return ManagedIdentityCredential(client_id=mi_client_id)
    return DefaultAzureCredential(
        exclude_interactive_browser_credential=True,
        exclude_visual_studio_code_credential=True,
        exclude_shared_token_cache_credential=True,
        exclude_powershell_credential=True,
    )

def get_openai_client() -> AzureOpenAI:
    endpoint = os.getenv("AOAI_ENDPOINT")
    api_version = os.getenv("AOAI_API_VERSION", "2024-02-15-preview")
    api_key = os.getenv("AOAI_API_KEY")
    if api_key:
        return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
    cred = get_azure_credential()
    def _token_provider():
        return cred.get_token("https://cognitiveservices.azure.com/.default").token
    return AzureOpenAI(azure_endpoint=endpoint, api_version=api_version, azure_ad_token_provider=_token_provider)

# --- Helpers -----------------------------------------------------------------
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def chunk_text(s: str, size: int, overlap: int) -> List[str]:
    if not s:
        return []
    out, i = [], 0
    step = max(1, size - overlap)
    while i < len(s):
        out.append(s[i:i+size])
        i += step
    return out

def to_plain_text(value: object) -> str:
    try:
        if value is None: return ""
        if isinstance(value, (datetime, date)): return value.isoformat()
        if 'ObjectId' in globals() and isinstance(value, ObjectId): return str(value)
        if 'Decimal128' in globals() and isinstance(value, Decimal128):
            try: return str(value.to_decimal())
            except Exception: return str(value)
        if isinstance(value, bytes):
            try: return value.decode('utf-8', 'ignore')
            except Exception: return str(value)
        if isinstance(value, (list, tuple)): return "; ".join(to_plain_text(v) for v in value)
        if isinstance(value, dict):
            try: return json.dumps(value, ensure_ascii=False, default=str)
            except Exception: return str(value)
        return str(value)
    except Exception:
        return str(value)

def coerce_doc_for_index(p: dict) -> dict:
    scalar = ["doc_id","source","collection","record_id","title","text","url","updated_at"]
    for k in scalar: p[k] = to_plain_text(p.get(k, ""))
    tags = p.get("tags", [])
    if not isinstance(tags, list): tags = [to_plain_text(tags)] if tags else []
    else: tags = [to_plain_text(t) for t in tags]
    p["tags"] = tags
    vec = p.get("text_vector", None)
    if isinstance(vec, list):
        try: p["text_vector"] = [float(x) for x in vec]
        except Exception: p["text_vector"] = []
    else:
        p["text_vector"] = [] if vec is None else [float(vec)]
    return p

# --- Core doc model ----------------------------------------------------------
@dataclass
class ChunkDoc:
    doc_id: str
    source: str
    collection: str
    record_id: str
    title: str
    text: str
    url: str = ""
    tags: List[str] = field(default_factory=list)
    updated_at: str = ""
    text_vector: Optional[List[float]] = None

# --- Connectors protocol & registry -----------------------------------------
class Connector(Protocol):
    name: str
    def iter_docs(self) -> Iterator[ChunkDoc]: ...

CONNECTORS: Dict[str, Any] = {}
def register_connector(kind: str):
    def _wrap(cls):
        CONNECTORS[kind] = cls
        return cls
    return _wrap

# --- Azure Search sink -------------------------------------------------------
class SearchSink:
    def __init__(self, index_name: str = DEFAULT_INDEX_NAME):
        self.endpoint = os.environ['SEARCH_ENDPOINT']
        self.index_name = index_name
        key = os.getenv("SEARCH_API_KEY")
        self.credential = AzureKeyCredential(key) if key else get_azure_credential()
        self.client = SearchClient(endpoint=self.endpoint, index_name=self.index_name, credential=self.credential)
        self.index_client = SearchIndexClient(endpoint=self.endpoint, credential=self.credential)
        self._existing_types: Dict[str, str] = {}
        dbg(f"SearchSink init endpoint={self.endpoint} index={self.index_name}")

    def ensure_index(self):
        fields = [
            SimpleField(name="doc_id", type=SearchFieldDataType.String, key=True, filterable=True, sortable=True),
            SearchableField(name="source",     type=SearchFieldDataType.String, filterable=True, facetable=True, analyzer_name="keyword"),
            SearchableField(name="collection", type=SearchFieldDataType.String, filterable=True, facetable=True, analyzer_name="keyword"),
            SearchableField(name="record_id",  type=SearchFieldDataType.String, filterable=True, analyzer_name="keyword"),
            SearchableField(name="title",      type=SearchFieldDataType.String),
            SearchableField(name="text",       type=SearchFieldDataType.String),
            SimpleField(name="url",            type=SearchFieldDataType.String),
            SearchableField(name="tags",       type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True, facetable=True, analyzer_name="keyword"),
            SimpleField(name="updated_at",     type=SearchFieldDataType.String, filterable=True, sortable=True),
            SearchField(
                name="text_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=EMBED_DIMS,
                vector_search_profile_name="hnswSearch",
            ),
        ]
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnswConfig")],
            profiles=[VectorSearchProfile(name="hnswSearch", algorithm_configuration_name="hnswConfig", vectorizer_name="aoaiVec")],
            vectorizers=[AzureOpenAIVectorizer(
                vectorizer_name="aoaiVec", kind="azureOpenAI",
                parameters=AzureOpenAIVectorizerParameters(
                    resource_url=os.environ['AOAI_ENDPOINT'],
                    deployment_name=EMBED_DEPLOYMENT,
                    model_name=EMBED_MODEL_NAME,
                ),
            )],
        )
        expected_types = {
            "doc_id":"Edm.String","source":"Edm.String","collection":"Edm.String","record_id":"Edm.String",
            "title":"Edm.String","text":"Edm.String","url":"Edm.String","tags":"Collection(Edm.String)",
            "updated_at":"Edm.String","text_vector":"Collection(Edm.Single)"
        }
        force_recreate = os.getenv("SEARCH_FORCE_RECREATE","false").lower()=="true"
        try:
            idx = self.index_client.get_index(self.index_name)
            existing_types = {f.name: str(f.type) for f in idx.fields}
            self._existing_types = existing_types
            dbg(f"index {self.index_name}: existing field types -> {existing_types}")
            mismatches = [(k, existing_types.get(k), v) for k,v in expected_types.items()
                          if k in existing_types and existing_types.get(k)!=v]
            if mismatches:
                dbg(f"index {self.index_name}: schema mismatches -> {mismatches}")
                if force_recreate:
                    dbg(f"index {self.index_name}: deleting to recreate (SEARCH_FORCE_RECREATE=true)")
                    self.index_client.delete_index(self.index_name)
                    idx = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)
                    self.index_client.create_or_update_index(idx)
                    dbg(f"index {self.index_name}: recreated")
                    return
                else:
                    dbg("Set SEARCH_FORCE_RECREATE=true to drop & recreate; upserts may fail until schema matches.")
            existing = {f.name for f in idx.fields}
            added = False
            for f in fields:
                if f.name not in existing:
                    idx.fields.append(f); added=True
            if added:
                self.index_client.create_or_update_index(idx)
                dbg(f"index {self.index_name}: added missing fields")
            else:
                dbg(f"index {self.index_name}: already compatible")
        except ResourceNotFoundError:
            idx = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)
            self.index_client.create_or_update_index(idx)
            self._existing_types = expected_types
            dbg(f"index {self.index_name}: created")

    def _coerce_for_index_types(self, payload: List[dict]) -> None:
        et = getattr(self, "_existing_types", {}) or {}
        if et.get("tags") == "Edm.String":
            for p in payload:
                if isinstance(p.get("tags"), list):
                    p["tags"] = "; ".join(str(t) for t in p["tags"])
        if et.get("text_vector") in ("Edm.Single","Edm.String"):
            for p in payload:
                if isinstance(p.get("text_vector"), list):
                    p.pop("text_vector", None)
        if payload:
            sample = {k: payload[0].get(k) for k in ("tags","text_vector")}
            if isinstance(sample.get("text_vector"), list):
                sample["text_vector_len"] = len(sample["text_vector"])
                sample["text_vector"] = "<list>"
            dbg("coercion based on existing index types:", et, "sample:", sample)

    def upsert(self, docs: List[ChunkDoc]):
        if not docs:
            dbg("upsert: 0 docs – nothing to do")
            return
        payload = [coerce_doc_for_index(d.__dict__.copy()) for d in docs]
        dbg("index type map:", self._existing_types)
        self._coerce_for_index_types(payload)
        sample = payload[0].copy()
        if len(sample.get("text",""))>160: sample["text"]=sample["text"][:160]+"…"
        sample["text_vector_len"] = len(sample.get("text_vector",[]) or [])
        dbg("upsert first doc sample:", {k: sample[k] for k in ("doc_id","title","url","tags","text_vector_len") if k in sample})
        total_ok = total_fail = 0
        for i in range(0, len(payload), 1000):
            chunk = payload[i:i+1000]
            try:
                results = self.client.merge_or_upload_documents(documents=chunk)
                ok = sum(1 for r in results if getattr(r, "succeeded", False))
                fail = len(results) - ok
                total_ok += ok; total_fail += fail
                if fail:
                    bad = next((r for r in results if not getattr(r, "succeeded", False)), None)
                    dbg(f"upsert chunk: ok={ok} fail={fail} sample_error={getattr(bad,'error_message',None)} key={getattr(bad,'key',None)}")
                else:
                    dbg(f"upsert chunk: ok={ok}")
            except Exception as e:
                total_fail += len(chunk)
                dbg("upsert chunk exception:", repr(e))
        dbg(f"upsert summary: ok={total_ok} fail={total_fail}")

# --- Embeddings --------------------------------------------------------------
class Embedder:
    def __init__(self):
        self.client = get_openai_client()
        self.model = EMBED_DEPLOYMENT
    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            dbg("embed: 0 texts – skipping"); return []
        try:
            resp = self.client.embeddings.create(model=self.model, input=texts)
            vecs = [d.embedding for d in resp.data]
            dbg(f"embed: inputs={len(texts)} outputs={len(vecs)}")
            return vecs
        except Exception as e:
            dbg("embed exception:", repr(e)); raise

# --- Mongo (GrandNode) -------------------------------------------------------
@register_connector("mongo")
class MongoConnector:
    name = "mongo"
    def __init__(self, cfg: Dict[str, Any]):
        if 'pymongo' not in sys.modules:
            raise RuntimeError("pymongo not installed. pip install pymongo")
        self.uri = cfg.get("uri") or os.environ.get("MONGO_URI")
        self.db_name = cfg.get("db") or os.environ.get("MONGO_DB", "grandnode")
        self.include = re.compile(cfg.get("include", os.environ.get("MONGO_INCLUDE", ".*")))
        self.exclude = re.compile(cfg.get("exclude", os.environ.get("MONGO_EXCLUDE", "^(Order|Customer|Address|Log|_.*)$")))
        self.title_fields = cfg.get("title_fields") or ["Name","Title","name","title","SystemName","SeName","Sku","SKU"]
        self.text_fields  = cfg.get("text_fields") or [
            "ShortDescription","FullDescription","Description","Body","Text","Overview","rawHtml","content",
            "ShortDescriptionRaw","FullDescriptionRaw"
        ]
        self.url_fields   = cfg.get("url_fields") or ["url","Url","ProductUrl","Slug","SeName"]
        self.base_url = cfg.get("base_url") or "https://cyclonerake.com"
        self.tags = cfg.get("tags", [])
        self.chunk_size = int(cfg.get("chunk_size", os.environ.get("MONGO_CHUNK_SIZE", 2000)))
        self.chunk_overlap = int(cfg.get("chunk_overlap", os.environ.get("MONGO_CHUNK_OVERLAP", 200)))
        self.client = MongoClient(self.uri); self.db = self.client[self.db_name]
        dbg(f"mongo init uri_set={'yes' if bool(self.uri) else 'no'} db={self.db_name} include={self.include.pattern} exclude={self.exclude.pattern}")

    def _pick_first(self, d: Dict[str, Any], keys: List[str]) -> str:
        for k in keys:
            if k in d and d[k]:
                return to_plain_text(d[k])
        return ""

    def _build_url(self, raw: str) -> str:
        if not raw: return self.base_url
        if raw.startswith("http://") or raw.startswith("https://"): return raw
        base = self.base_url.rstrip("/") + "/"
        return base + raw.strip("/")

    def _doc_to_chunks(self, coll: str, doc: Dict[str, Any]) -> List[ChunkDoc]:
        mongo_id = to_plain_text(doc.get("_id",""))
        title = self._pick_first(doc, self.title_fields) or coll
        url   = self._build_url(self._pick_first(doc, self.url_fields))
        parts: List[str] = []
        for k in self.text_fields:
            if k in doc and doc[k]:
                parts.append(to_plain_text(doc[k]))
        if not parts:
            shallow = {k: v for k,v in doc.items() if k != "_id"}
            try: parts.append(json.dumps(shallow, ensure_ascii=False, default=str))
            except Exception: parts.append(to_plain_text(shallow))
        text = "\n".join(p for p in parts if p).strip()
        if not text:
            dbg(f"mongo skip (empty text) coll={coll} id={mongo_id}"); return []
        updated_at = to_plain_text(doc.get("UpdatedOnUtc") or doc.get("updated_at") or doc.get("LastUpdated") or "")
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        out: List[ChunkDoc] = []
        for i,c in enumerate(chunks):
            doc_id = sha1(f"mongo:{coll}:{mongo_id}:{i}")
            out.append(ChunkDoc(doc_id, self.name, coll, mongo_id, title, c, url, self.tags, updated_at))
        return out

    def iter_docs(self) -> Iterator[ChunkDoc]:
        all_names = self.db.list_collection_names()
        names = [c for c in all_names if self.include.search(c) and not self.exclude.search(c)]
        dbg("mongo collections (all):", all_names)
        dbg("mongo collections (selected):", names)
        for coll_name in names:
            coll = self.db[coll_name]
            try:
                try:
                    est = coll.estimated_document_count()
                    dbg(f"mongo processing collection={coll_name} approx_docs={est}")
                except Exception:
                    dbg(f"mongo processing collection={coll_name}")
                cursor = coll.find({}, no_cursor_timeout=True)
                try:
                    cnt = 0
                    for doc in cursor:
                        try:
                            for ch in self._doc_to_chunks(coll_name, doc):
                                cnt += 1; yield ch
                        except Exception as e_doc:
                            dbg(f"mongo doc skipped coll={coll_name} id={doc.get('_id','')} err={repr(e_doc)}")
                            continue
                    dbg(f"mongo collection done {coll_name}: yielded_chunks={cnt}")
                finally:
                    cursor.close()
            except Exception as e:
                dbg(f"mongo collection exception {coll_name}:", repr(e))

# --- Airtable ---------------------------------------------------------------
@register_connector("airtable")
class AirtableConnector:
    name = "airtable"
    def __init__(self, cfg: Dict[str, Any]):
        if 'pyairtable' not in sys.modules:
            raise RuntimeError("pyairtable not installed. pip install pyairtable")
        # Airtable uses Personal Access Tokens (PAT). We accept multiple env var names for convenience.
        # Preferred: AIRTABLE_PAT or AIRTABLE_PERSONAL_ACCESS_TOKEN. We still fall back to AIRTABLE_API_KEY for legacy setups.
        self.token = (
            cfg.get("token")
            or os.environ.get("AIRTABLE_PAT")
            or os.environ.get("AIRTABLE_PERSONAL_ACCESS_TOKEN")
            or os.environ.get("AIRTABLE_TOKEN")
            or os.environ.get("AIRTABLE_API_KEY")  # legacy
        )
        self.base_id = cfg.get("base_id") or os.environ.get("AIRTABLE_BASE_ID")
        self.tables  = cfg.get("tables") or (os.environ.get("AIRTABLE_TABLES","").split(",") if os.environ.get("AIRTABLE_TABLES") else [])
        self.title_fields = cfg.get("title_fields") or ["Name","Title","name","title"]
        self.text_fields  = cfg.get("text_fields")  or ["Description","Notes","Body","Text","Content"]
        self.url_fields   = cfg.get("url_fields")   or ["Url","URL","Link"]
        self.base_url = cfg.get("base_url") or ""
        self.tags = cfg.get("tags", ["airtable"])
        self.chunk_size = int(cfg.get("chunk_size", 2000))
        self.chunk_overlap = int(cfg.get("chunk_overlap", 200))

        if not self.token:
            raise RuntimeError(
                "Airtable PAT not configured. Set one of: AIRTABLE_PAT, AIRTABLE_PERSONAL_ACCESS_TOKEN, AIRTABLE_TOKEN, or provide `token` in the source config."
            )
        if not self.base_id:
            raise RuntimeError("Airtable BASE_ID missing. Set AIRTABLE_BASE_ID or provide `base_id` in the source config.")

        # Sanity checks: PATs start with 'pat', base IDs start with 'app'
        if self.token and not str(self.token).startswith("pat"):
            dbg("WARNING: Airtable token does not look like a Personal Access Token (should start with 'pat').")
        if self.base_id and not str(self.base_id).startswith("app"):
            raise RuntimeError(
                f"Airtable BASE_ID looks incorrect: '{self.base_id}'. It should start with 'app'.\n"
                "Ensure AIRTABLE_BASE_ID is set to your base ID (not the token)."
            )

        dbg(f"airtable init base={self.base_id} tables={self.tables} token_set={'yes' if bool(self.token) else 'no'}")

    def _pick_first(self, fields: Dict[str, Any], keys: List[str]) -> str:
        for k in keys:
            if k in fields and fields[k]:
                v = fields[k]; return v if isinstance(v,str) else json.dumps(v, ensure_ascii=False)
        return ""

    def _build_url(self, raw: str) -> str:
        if not raw: return self.base_url
        return raw

    def iter_docs(self) -> Iterator[ChunkDoc]:
        dbg("airtable tables:", self.tables)
        for table_name in self.tables:
            api = Api(self.token)
            table = api.table(self.base_id, table_name)

            # ---- Preflight probe: fetch up to 1 record to validate PAT/base/table ----
            try:
                probe = table.all(max_records=1)
                if isinstance(probe, list) and len(probe) > 0 and isinstance(probe[0], dict):
                    dbg(f"[airtable] preflight ok base={self.base_id} table={table_name}")
                else:
                    dbg(f"[airtable] preflight ok but table is empty base={self.base_id} table={table_name}")
            except Exception as e_probe:
                dbg(f"[airtable] preflight failed base={self.base_id} table={table_name}: {repr(e_probe)}")
                dbg("Hints: ensure PAT scopes include data.records:read, PAT is granted access to this base, and the table name or table ID (starts with 'tbl') is exact. If the table is a synced view or has special permissions, use the table ID.")
                continue  # skip this table and move on

            # ---- Normal iteration with robust shape handling ----
            try:
                for page in table.iterate(page_size=100):
                    # pyairtable may yield a list of records (a page) OR individual dict records depending on version
                    if isinstance(page, list):
                        records = page
                    elif isinstance(page, dict) and "records" in page and isinstance(page["records"], list):
                        records = page["records"]
                    else:
                        records = [page]  # assume a single record dict

                    for r in records:
                        if not isinstance(r, dict):
                            continue
                        # Normalize to Airtable record shape
                        if "id" in r and "fields" in r:
                            rec_id = r.get('id', '')
                            fields = r.get('fields', {}) or {}
                        elif "id" in r and "fields" not in r:
                            rec_id = r.get('id', '')
                            fields = {k: v for k, v in r.items() if k != 'id'}
                        else:
                            # Some SDKs return {"fields": {...}} without id; use a stable hash as record_id
                            fields = r.get('fields', r)
                            rec_id = hashlib.sha1(json.dumps(fields, sort_keys=True, default=str).encode('utf-8')).hexdigest()

                        title = self._pick_first(fields, self.title_fields) or table_name
                        url = self._build_url(self._pick_first(fields, self.url_fields))

                        parts: List[str] = []
                        for k in self.text_fields:
                            if k in fields and fields[k]:
                                v = fields[k]
                                parts.append(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
                        if not parts:
                            parts.append(json.dumps(fields, ensure_ascii=False))
                        text = "\n".join(parts).strip()

                        for i, c in enumerate(chunk_text(text, self.chunk_size, self.chunk_overlap)):
                            doc_id = sha1(f"airtable:{table_name}:{rec_id}:{i}")
                            yield ChunkDoc(
                                doc_id=doc_id,
                                source=self.name,
                                collection=table_name,
                                record_id=rec_id,
                                title=title,
                                text=c,
                                url=url,
                                tags=self.tags,
                                updated_at="",
                            )

                dbg(f"[airtable] table done base={self.base_id} table={table_name}")
            except Exception as e:
                dbg(f"airtable error table={table_name} base={self.base_id}: {repr(e)}")
                dbg("Hints: (1) Verify AIRTABLE_BASE_ID starts with 'app' and matches the base that the PAT can access; (2) Confirm the table name (or use the table ID 'tbl...'); (3) Ensure the PAT scope includes data.records:read.")
                continue

# --- Web: Cyclopedia (Document360) ------------------------------------------
@register_connector("web_cyclopedia")
class WebCyclopediaConnector:
    name = "cyclopedia"
    RM_SELECTORS = [
        "header","footer","nav",".navbar",".menu",".sidebar",".breadcrumbs",".cookie",".cookie-banner",
        ".consent",".skip-link",".toc",".search",".ad"
    ]
    MAIN_SELECTORS = [
        "main article","article","main .markdown",".markdown",".md-content",".md-typeset",".docs-content",
        ".docMainContainer",".theme-doc-markdown",".docItemContainer","#doc-content","main","#content",".content"
    ]
    def __init__(self, cfg: Dict[str, Any]):
        self.site = cfg.get("site", "https://cyclopedia.cyclonerake.com")
        self.sitemap_url = cfg.get("sitemap_url", f"{self.site}/sitemap-en.xml")
        self.allow_prefixes = cfg.get("allow_prefixes", ["/docs/"])
        self.max_pages = int(cfg.get("max_pages", 1000))
        self.sleep_sec = float(cfg.get("sleep_sec", 0.15))
        self.state_path = cfg.get("state_path", os.path.join(os.path.dirname(__file__), "..", "state_cyclopedia.json"))
        self.tags = cfg.get("tags", ["wiki","cyclopedia"])
        self.chunk_size = int(cfg.get("chunk_size", 2000))
        self.chunk_overlap = int(cfg.get("chunk_overlap", 200))
        self.wait_selector = cfg.get("wait_selector", "body")
        import requests
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA})
        self.state: Dict[str, Any] = self._load_state()
        self.pages: Dict[str, Any] = self.state.get("pages", {})
        dbg(f"web_cyclopedia init site={self.site} sitemap={self.sitemap_url} allow={self.allow_prefixes}")

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.state_path):
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"pages": {}}

    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def _norm_url(self, href: str) -> str:
        u = urllib.parse.urljoin(self.site, href)
        p = urllib.parse.urlparse(u)
        path = p.path
        if path == "/docs": path = "/docs/"
        if path.endswith("/") and path != "/docs/": path = path.rstrip("/")
        return urllib.parse.urlunparse((p.scheme, p.netloc, path, "", "", ""))

    def _is_allowed(self, url: str) -> bool:
        p = urllib.parse.urlparse(url); base = urllib.parse.urlparse(self.site)
        if p.scheme != base.scheme or p.netloc != base.netloc: return False
        return any(p.path.startswith(prefix) or p.path == prefix.rstrip("/") for prefix in self.allow_prefixes)

    def _discover_from_sitemap(self) -> List[str]:
        urls: List[str] = []
        try:
            r = self.session.get(self.sitemap_url, timeout=20)
            dbg("sitemap fetch:", r.status_code, r.headers.get("content-type"))
            if r.ok and (("xml" in r.headers.get("content-type","")) or r.text.lstrip().startswith("<")):
                root = ET.fromstring(r.text)
                for loc in root.findall(".//{*}loc"):
                    href = (loc.text or "").strip()
                    if not href: continue
                    u = self._norm_url(href)
                    if self._is_allowed(u): urls.append(u)
        except Exception as e:
            dbg("sitemap error:", repr(e))
        dbg("sitemap urls:", len(urls))
        return sorted(set(urls))

    def _render(self, url: str, timeout_ms: int = 15000) -> Optional[str]:
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                ctx = browser.new_context(user_agent=UA)
                page = ctx.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                try: page.wait_for_selector(self.wait_selector, timeout=timeout_ms)
                except Exception: pass
                html = page.content()
                ctx.close(); browser.close()
                dbg("rendered", url, "html-len:", len(html))
                return html
        except Exception as e:
            dbg("render-fail", url, repr(e)); return None

    def _fetch_html(self, url: str) -> Optional[str]:
        r = self.session.get(url, timeout=30, allow_redirects=True)
        if not r.ok or "text/html" not in r.headers.get("content-type",""):
            dbg("fetch-skip", url, r.status_code, r.headers.get("content-type")); return None
        dbg("fetch", url, r.status_code); return r.text

    def _extract_title_and_main(self, html: str):
        soup = BeautifulSoup(html, "html.parser")
        for sel in self.RM_SELECTORS:
            for el in soup.select(sel): el.decompose()
        title = ""
        h1 = soup.select_one("h1")
        if h1: title = h1.get_text(" ", strip=True)
        elif soup.title: title = soup.title.get_text(" ", strip=True)
        main = None
        for sel in self.MAIN_SELECTORS:
            main = soup.select_one(sel)
            if main: break
        if not main: main = soup.body or soup
        for t in main(["script","style","noscript"]): t.decompose()
        return title or "", main

    def _sections_from_dom(self, main: Any) -> List[tuple[str,str]]:
        sections: List[tuple[str,str]] = []; buf: List[str] = []; current = "Overview"
        def flush():
            nonlocal buf, current
            text = "\n".join([x.strip() for x in buf if x.strip()])
            if text: sections.append((current, text)); buf.clear()
        for el in main.descendants:
            name = getattr(el, "name", None)
            if name in ("h2","h3"): flush(); current = el.get_text(" ", strip=True)
            elif name in ("p","li"): buf.append(el.get_text(" ", strip=True))
            elif name == "pre":     buf.append(el.get_text("\n", strip=True))
        flush()
        return [(h,b) for h,b in sections if b]

    def iter_docs(self) -> Iterator[ChunkDoc]:
        urls = self._discover_from_sitemap()[: self.max_pages]
        dbg("web_cyclopedia urls selected:", len(urls))
        for url in urls:
            page_state = self.pages.get(url, {})
            html = self._render(url) if USE_BROWSER else self._fetch_html(url)
            if not html: continue
            title, main = self._extract_title_and_main(html)
            sections = self._sections_from_dom(main)
            if not sections:
                self.pages[url] = {"sections": page_state.get("sections", {})}; self._save_state(); continue
            prev = page_state.get("sections", {}); new_map: Dict[str,str] = {}
            for heading, body in sections:
                bh = sha1(body); new_map[heading]=bh
                if not REINDEX_ALL and prev.get(heading)==bh: continue
                for i, piece in enumerate(chunk_text(body, self.chunk_size, self.chunk_overlap)):
                    doc_id = sha1(f"{url}#{heading}-{i}")
                    yield ChunkDoc(doc_id, "cyclopedia", "docs", f"{url}#{heading}", title or heading, piece, url, self.tags, "")
            self.pages[url] = {"sections": new_map}; self._save_state()

# --- Web: Generic Website (CycloneRake.com) with robots.txt ------------------
@register_connector("web_site")
class WebSiteConnector:
    """
    Generic sitemap + robots-aware web connector.
    Use it for https://cyclonerake.com with respect for robots.txt.
    """
    name = "website"
    def __init__(self, cfg: Dict[str, Any]):
        self.site = cfg["site"]  # e.g., https://cyclonerake.com
        self.sitemaps: List[str] = cfg.get("sitemaps") or [urllib.parse.urljoin(self.site, "/sitemap.xml")]
        self.allow_prefixes = cfg.get("allow_prefixes", ["/"])
        self.disallow_exts = cfg.get("disallow_exts", list(ASSET_EXT))
        self.respect_robots = bool(cfg.get("respect_robots", True))
        self.max_pages = int(cfg.get("max_pages", 2000))
        self.sleep_sec = float(cfg.get("sleep_sec", 0.10))
        self.tags = cfg.get("tags", ["site"])
        self.chunk_size = int(cfg.get("chunk_size", 2000))
        self.chunk_overlap = int(cfg.get("chunk_overlap", 200))
        self.state_path = cfg.get("state_path", os.path.join(os.path.dirname(__file__), "..", "state_site.json"))
        self.wait_selector = cfg.get("wait_selector", "body")

        import requests, urllib.robotparser
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA})
        self.rp = urllib.robotparser.RobotFileParser()
        self.rp.set_url(urllib.parse.urljoin(self.site, "/robots.txt"))
        try:
            if self.respect_robots:
                self.rp.read()
                dbg("robots.txt loaded; can_fetch(site root):", self.rp.can_fetch(UA, self.site))
        except Exception as e:
            dbg("robots read error:", repr(e))

        self.state: Dict[str, Any] = self._load_state()
        self.pages: Dict[str, Any] = self.state.get("pages", {})
        dbg(f"web_site init site={self.site} sitemaps={self.sitemaps} allow={self.allow_prefixes} respect_robots={self.respect_robots}")

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.state_path):
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"pages": {}}
    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def _norm_url(self, href: str) -> str:
        u = urllib.parse.urljoin(self.site, href)
        p = urllib.parse.urlparse(u)
        path = p.path
        if path.endswith("/") and path != "/": path = path.rstrip("/")
        return urllib.parse.urlunparse((p.scheme, p.netloc, path, "", "", ""))

    def _is_allowed(self, url: str) -> bool:
        p = urllib.parse.urlparse(url); base = urllib.parse.urlparse(self.site)
        if p.scheme != base.scheme or p.netloc != base.netloc: return False
        if any(p.path.startswith(prefix) or p.path == prefix.rstrip("/") for prefix in self.allow_prefixes):
            ext = os.path.splitext(p.path)[1].lower()
            if ext in self.disallow_exts: return False
            if self.respect_robots:
                try:
                    return self.rp.can_fetch(UA, url)
                except Exception:
                    return True
            return True
        return False

    def _gather_sitemaps(self) -> List[str]:
        # Accept both sitemap.xml and sitemap index with <sitemap><loc>…</loc></sitemap>
        out: List[str] = []
        for sm in self.sitemaps:
            try:
                r = self.session.get(sm, timeout=20)
                if not r.ok: continue
                root = ET.fromstring(r.text)
                if root.tag.endswith("sitemapindex"):
                    for loc in root.findall(".//{*}loc"):
                        if loc.text: out.append(loc.text.strip())
                else:
                    out.append(sm)
            except Exception as e:
                dbg("sitemap load error:", sm, repr(e))
        return list(dict.fromkeys(out))  # uniq + preserve order

    def _discover_from_sitemaps(self) -> List[str]:
        urls: List[str] = []
        for sm in self._gather_sitemaps():
            try:
                r = self.session.get(sm, timeout=30)
                dbg("sitemap fetch:", sm, r.status_code, r.headers.get("content-type"))
                if not r.ok: continue
                root = ET.fromstring(r.text)
                for loc in root.findall(".//{*}loc"):
                    href = (loc.text or "").strip()
                    if not href: continue
                    u = self._norm_url(href)
                    if self._is_allowed(u): urls.append(u)
            except Exception as e:
                dbg("sitemap parse error:", sm, repr(e))
        dbg("site sitemap urls:", len(urls))
        # Unique & cap
        return list(dict.fromkeys(urls))[: self.max_pages]

    def _render(self, url: str, timeout_ms: int = 15000) -> Optional[str]:
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                ctx = browser.new_context(user_agent=UA)
                page = ctx.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                try: page.wait_for_selector(self.wait_selector, timeout=timeout_ms)
                except Exception: pass
                html = page.content()
                ctx.close(); browser.close()
                dbg("rendered", url, "html-len:", len(html))
                return html
        except Exception as e:
            dbg("render-fail", url, repr(e)); return None

    def _fetch_html(self, url: str) -> Optional[str]:
        r = self.session.get(url, timeout=30, allow_redirects=True)
        if not r.ok or "text/html" not in r.headers.get("content-type",""):
            dbg("fetch-skip", url, r.status_code, r.headers.get("content-type")); return None
        dbg("fetch", url, r.status_code); return r.text

    def _extract(self, html: str):
        soup = BeautifulSoup(html, "html.parser")
        # Best-effort generic clean for ecommerce CMS templates:
        for sel in [
            "header","footer","nav",".navbar",".menu",".breadcrumbs",".cookie",".cookie-banner",".newsletter",
            ".consent",".skip-link",".search",".ad",".sidebar",".filters",".facets",".pagination",".reviews-summary"
        ]:
            for el in soup.select(sel): el.decompose()
        title = ""
        h1 = soup.select_one("h1")
        if h1: title = h1.get_text(" ", strip=True)
        elif soup.title: title = soup.title.get_text(" ", strip=True)
        main = soup.select_one("main") or soup.select_one("#content") or soup.body or soup
        for t in main(["script","style","noscript"]): t.decompose()
        return title or "", main

    def _sections(self, main) -> List[tuple[str,str]]:
        sections: List[tuple[str,str]] = []; buf: List[str] = []; current = "Overview"
        def flush():
            nonlocal buf, current
            text = "\n".join([x.strip() for x in buf if x.strip()])
            if text: sections.append((current, text)); buf.clear()
        for el in main.descendants:
            name = getattr(el, "name", None)
            if name in ("h2","h3"): flush(); current = el.get_text(" ", strip=True) or current
            elif name in ("p","li"): buf.append(el.get_text(" ", strip=True))
            elif name == "pre":     buf.append(el.get_text("\n", strip=True))
        flush()
        return [(h,b) for h,b in sections if b]

    def iter_docs(self) -> Iterator[ChunkDoc]:
        urls = self._discover_from_sitemaps()
        dbg("web_site urls selected:", len(urls))
        for url in urls:
            if not self._is_allowed(url): continue
            html = self._render(url) if USE_BROWSER else self._fetch_html(url)
            if not html: continue
            title, main = self._extract(html)
            secs = self._sections(main)
            if not secs: continue
            for heading, body in secs:
                body_hash = sha1(f"{url}#{heading}")
                for i, piece in enumerate(chunk_text(body, self.chunk_size, self.chunk_overlap)):
                    doc_id = sha1(f"{url}#{heading}-{i}")
                    yield ChunkDoc(doc_id, "website", urllib.parse.urlparse(self.site).netloc, body_hash, title or heading, piece, url, self.tags, "")

# --- Pipeline ----------------------------------------------------------------
@dataclass
class PipelineConfig:
    sources: List[Dict[str, Any]]
    index_name: str = DEFAULT_INDEX_NAME
    batch: int = 64
    embed_batch: int = 16

class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.embedder = Embedder()
        self.default_sink = SearchSink(index_name=cfg.index_name)
        self.default_sink.ensure_index()
        self.batch = cfg.batch; self.embed_batch = cfg.embed_batch
        self._sinks: Dict[str, SearchSink] = {}
        dbg(f"Pipeline init default_index={cfg.index_name} batch={self.batch} embed_batch={self.embed_batch}")

    def _get_sink(self, idx_name: Optional[str]) -> SearchSink:
        if not idx_name or idx_name == self.default_sink.index_name:
            return self.default_sink
        if idx_name not in self._sinks:
            s = SearchSink(index_name=idx_name); s.ensure_index(); self._sinks[idx_name] = s
        return self._sinks[idx_name]

    def _embed_and_upsert(self, sink: SearchSink, items: List[ChunkDoc]):
        dbg(f"stage: embed+upsert items={len(items)} → index={sink.index_name}")
        # embeddings in smaller sub-batches:
        for i in range(0, len(items), self.embed_batch):
            part = items[i:i+self.embed_batch]
            vecs = self.embedder.embed([d.text for d in part])
            for d, v in zip(part, vecs):
                d.text_vector = v
        sink.upsert(items)

    def run(self):
        total_chunks = 0
        dbg(f"run: sources={len(self.cfg.sources)}")
        for src_cfg in self.cfg.sources:
            kind = src_cfg.get('type')
            idx_name = src_cfg.get('index_name')  # optional per-source index
            sink = self._get_sink(idx_name)
            klass = CONNECTORS.get(kind)
            dbg(f"run: connector={kind} cfg_keys={list(src_cfg.keys())} → index={sink.index_name}")
            if not klass:
                dbg(f"ERROR unknown connector: {kind}"); continue
            connector: Connector = klass(src_cfg)
            dbg(f"running connector: {connector.name}")
            batch: List[ChunkDoc] = []; yielded = 0
            for doc in connector.iter_docs():
                batch.append(doc); yielded += 1
                if len(batch) >= self.batch:
                    self._embed_and_upsert(sink, batch)
                    total_chunks += len(batch)
                    dbg(f"run: flushed batch size={len(batch)} total_chunks={total_chunks}")
                    batch = []; time.sleep(0.05)
            if batch:
                self._embed_and_upsert(sink, batch)
                total_chunks += len(batch)
                dbg(f"run: flushed final batch size={len(batch)} total_chunks={total_chunks}")
            if yielded == 0:
                dbg(f"WARNING connector {connector.name} produced 0 items – check filters/credentials")
        dbg("ingestion complete", dict(total_chunks=total_chunks))

# --- CLI ---------------------------------------------------------------------
EXAMPLE_CONFIG_JSON = {
    "index_name": os.environ.get("SEARCH_INDEX_NAME_UNIFIED", "unified-knowledge"),
    "batch": 64,
    "embed_batch": 16,
    "sources": [
       

        # --- Airtable (PAT-based) ---
        {
            "type": "airtable",
            "token": os.environ.get("AIRTABLE_PAT", ""),   # Personal Access Token
            "base_id": os.environ.get("AIRTABLE_BASE_ID", ""),
            # Use display names OR table IDs (tblXXXXXXXXXXXXXX). IDs avoid name/space/case issues.
            "tables": [os.environ.get("AIRTABLE_TABLE", "Imported table")],
            "tags": ["questions"],
            "index_name": os.environ.get("SEARCH_INDEX_NAME_UNIFIED", "unified-knowledge")
        }
    ]
}

def _load_config(path: Optional[str]) -> PipelineConfig:
    if not path:
        dbg("no --config provided; using EXAMPLE_CONFIG_JSON")
        obj = EXAMPLE_CONFIG_JSON
    else:
        with open(path, 'r') as f:
            obj = json.load(f)
    return PipelineConfig(**obj)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Unified ingestion pipeline")
    ap.add_argument("--config", help="Path to JSON config", default=None)
    args = ap.parse_args()
    cfg = _load_config(args.config)
    Pipeline(cfg).run()