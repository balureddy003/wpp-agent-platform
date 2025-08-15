import os, time, re, hashlib, json, urllib.parse, contextlib
from xml.etree import ElementTree as ET
from typing import List, Dict, Tuple, Iterable, Optional, Set
import requests
from bs4 import BeautifulSoup
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchField, SearchFieldDataType
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

SITE = "https://cyclopedia.cyclonerake.com"
START_URL = f"{SITE}/docs/"
SITEMAP_URL = f"{SITE}/sitemap-en.xml"
STATE_PATH = os.path.join(os.path.dirname(__file__), "..", "state_cyclopedia.json")
UA = "wpp-agent-ingestor/1.0 (+github.com/balureddy003/wpp-agent-platform)"

DEBUG = str(os.getenv("INGEST_DEBUG", "0")).lower() not in ("0", "false", "no", "")
USE_BROWSER = str(os.getenv("INGEST_USE_BROWSER", "0")).lower() not in ("0", "false", "no", "")
REINDEX_ALL = str(os.getenv("INGEST_REINDEX", "0")).lower() not in ("0", "false", "no", "")

def dbg(*parts):
    if DEBUG:
        try:
            print("[ingest-debug]", *parts)
        except Exception:
            pass

# --- Azure Identity helper (match infra behavior) ---
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

def _get_aoai_client():
    endpoint = os.getenv("AOAI_ENDPOINT")
    api_version = os.getenv("AOAI_API_VERSION", "2024-02-15-preview")
    api_key = os.getenv("AOAI_API_KEY")
    if api_key:
        return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
    cred = get_azure_credential()
    def _token_provider():
        return cred.get_token("https://cognitiveservices.azure.com/.default").token
    return AzureOpenAI(azure_endpoint=endpoint, api_version=api_version, azure_ad_token_provider=_token_provider)

_openai = _get_aoai_client()
_embed_deployment = os.getenv("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")

def _get_search_credential():
    key = os.getenv("SEARCH_API_KEY")
    if key:
        dbg("search auth: using API key")
        return AzureKeyCredential(key)
    dbg("search auth: using AAD/Managed Identity")
    return get_azure_credential()

_search = SearchClient(
    endpoint=os.environ["SEARCH_ENDPOINT"],
    index_name=os.environ["SEARCH_INDEX_NAME"],
    credential=_get_search_credential(),
)

# --- Ensure index has 'url' field helper ---
def ensure_index_has_url_field():
    try:
        idx_client = SearchIndexClient(endpoint=os.environ["SEARCH_ENDPOINT"], credential=_get_search_credential())
        idx_name = os.environ["SEARCH_INDEX_NAME"]
        idx = idx_client.get_index(idx_name)
        if any(f.name == "url" for f in idx.fields):
            dbg("index already contains 'url' field")
            return
        dbg("adding 'url' field to index", idx_name)
        idx.fields.append(SearchField(name="url", type=SearchFieldDataType.String))
        idx_client.create_or_update_index(idx)
        dbg("'url' field added to index", idx_name)
    except Exception as e:
        dbg("ensure_index_has_url_field failed:", type(e).__name__, str(e))

# --- Helpers ---
ASSET_EXT = {".png",".jpg",".jpeg",".gif",".webp",".svg",".pdf",".zip",".gz",".mp4",".mov",".css",".js",".ico",".woff",".woff2",".ttf"}

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def norm_url(u: str) -> str:
    u = urllib.parse.urljoin(SITE, u)
    p = urllib.parse.urlparse(u)
    # strip query/fragment, normalize trailing slash (keep one canonical form)
    path = p.path
    # keep /docs/ canonical
    if path == "/docs":
        path = "/docs/"
    # avoid trailing slash duplicates except for /docs/
    if path.endswith("/") and path != "/docs/":
        path = path.rstrip("/")
    return urllib.parse.urlunparse((p.scheme, p.netloc, path, "", "", ""))

def is_allowed_url(u: str) -> bool:
    pu = urllib.parse.urlparse(u)
    ps = urllib.parse.urlparse(SITE)
    if pu.netloc != ps.netloc or pu.scheme != ps.scheme:
        return False
    # stay under /docs or /docs/
    if not (pu.path == "/docs" or pu.path.startswith("/docs/")):
        return False
    ext = os.path.splitext(pu.path)[1].lower()
    return ext not in ASSET_EXT

def load_state() -> Dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    return {"pages": {}}

def save_state(state: Dict):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

# --- Optional: JS rendering via Playwright for SPA sites (e.g., Document360) ---
class _Browser:
    def __init__(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._ctx = self._browser.new_context(user_agent=UA)
    def page(self):
        return self._ctx.new_page()
    def close(self):
        with contextlib.suppress(Exception):
            self._ctx.close()
        with contextlib.suppress(Exception):
            self._browser.close()
        with contextlib.suppress(Exception):
            self._pw.stop()

def render_html(url: str, wait_selector: str = "body", timeout_ms: int = 15000) -> Optional[str]:
    """Render URL with headless Chromium and return full HTML (including JS content)."""
    try:
        br = _Browser()
        p = br.page()
        p.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        try:
            p.wait_for_selector(wait_selector, timeout=timeout_ms)
        except Exception:
            pass
        html = p.content()
        br.close()
        dbg("rendered", url, "html-len:", len(html))
        return html
    except Exception as e:
        dbg("render-fail", url, type(e).__name__, str(e))
        return None

def discover_urls(max_pages: int = 1000, sleep_sec: float = 0.15) -> List[str]:
    """Discover URLs using the site's sitemap first, then fall back to a constrained crawl.
    Only URLs under /docs are retained.
    """
    seen: Set[str] = set()

    # 1) Try sitemap first
    try:
        r = requests.get(SITEMAP_URL, headers={"User-Agent": UA}, timeout=20)
        dbg("sitemap fetch:", r.status_code, r.headers.get("content-type"))
        if r.ok and ("xml" in r.headers.get("content-type", "") or r.text.lstrip().startswith("<")):
            root = ET.fromstring(r.text)
            # Handle namespaces (present or absent)
            if root.tag.startswith("{"):
                ns_uri = root.tag.split("}")[0].strip("{")
                ns = {"sm": ns_uri}
                locs = root.findall(".//sm:loc", ns)
            else:
                locs = root.findall(".//loc")
            for loc in locs:
                href = (loc.text or "").strip()
                if not href:
                    continue
                nu = norm_url(href)
                if is_allowed_url(nu):
                    seen.add(nu)
            dbg("sitemap urls:", len(seen))
    except Exception as e:
        dbg("sitemap error:", type(e).__name__, str(e))

    if seen:
        # Limit to max_pages and return
        urls = sorted(seen)
        return urls[:max_pages]

    # 2) Fallback: BFS crawl from /docs/
    q: List[str] = [START_URL]
    while q and len(seen) < max_pages:
        url = q.pop(0)
        if url in seen or not is_allowed_url(url):
            continue
        seen.add(url)
        try:
            if USE_BROWSER:
                html = render_html(url)
                if not html:
                    time.sleep(sleep_sec)
                    continue
                soup = BeautifulSoup(html, "html.parser")
            else:
                r = requests.get(url, headers={"User-Agent": UA}, timeout=20, allow_redirects=True)
                dbg("fetch:", url, r.status_code, r.headers.get("content-type"))
                if not r.ok or "text/html" not in r.headers.get("content-type", ""):
                    time.sleep(sleep_sec)
                    continue
                soup = BeautifulSoup(r.text, "html.parser")

            links = 0
            for a in soup.select("a[href]"):
                nu = norm_url(a.get("href", ""))
                if is_allowed_url(nu) and nu not in seen and nu not in q:
                    q.append(nu)
                    links += 1
            dbg("discovered-from", url, "→", links, "links; queue:", len(q))
        except Exception as e:
            dbg("discover-error:", url, type(e).__name__, str(e))
        time.sleep(sleep_sec)
    dbg("discovered count:", len(seen))
    return sorted(seen)

# --- Fetch with conditional GET ---
def fetch(url: str, etag: Optional[str], last_mod: Optional[str]) -> Tuple[int, Dict, Optional[str]]:
    headers = {"User-Agent": UA}
    if etag: headers["If-None-Match"] = etag
    if last_mod: headers["If-Modified-Since"] = last_mod
    if USE_BROWSER:
        html = render_html(url)
        # Simulate a 200 for rendered content
        meta = {"etag": None, "last_modified": None, "content_type": "text/html"}
        status = 200 if html else 500
        return status, meta, html
    r = requests.get(url, headers=headers, timeout=30)
    dbg("conditional fetch:", url, r.status_code, r.headers.get("content-type"))
    meta = {
        "etag": r.headers.get("ETag"),
        "last_modified": r.headers.get("Last-Modified"),
        "content_type": r.headers.get("content-type",""),
    }
    html = r.text if r.ok and "text/html" in meta["content_type"] else None
    return r.status_code, meta, html

# --- Parse & Clean (docs-friendly selectors) ---
RM_SELECTORS = [
    "header", "footer", "nav", ".navbar", ".menu", ".sidebar", ".breadcrumbs",
    ".cookie", ".cookie-banner", ".consent", ".skip-link", ".toc", ".search", ".ad"
]
MAIN_SELECTORS = [
    "main article", "article", "main .markdown", ".markdown", ".md-content", ".md-typeset",
    ".docs-content", ".docMainContainer", ".theme-doc-markdown", ".docItemContainer", "#doc-content",
    "main", "#content", ".content"
]

def extract_title_and_main(html: str) -> Tuple[str, BeautifulSoup]:
    soup = BeautifulSoup(html, "html.parser")
    for sel in RM_SELECTORS:
        for el in soup.select(sel):
            el.decompose()
    title = ""
    h1 = soup.select_one("h1")
    if h1: title = h1.get_text(" ", strip=True)
    elif soup.title: title = soup.title.get_text(" ", strip=True)

    main = None
    for sel in MAIN_SELECTORS:
        main = soup.select_one(sel)
        if main: break
    if not main:
        dbg("no main selector matched; using body")
        main = soup.body or soup
    # remove scripts/styles from main
    for t in main(["script","style","noscript"]):
        t.decompose()
    return title or "", main

# --- Sectioning by DOM (H2/H3) ---
def sections_from_dom(main: BeautifulSoup) -> List[Tuple[str, str]]:
    sections: List[Tuple[str,str]] = []
    buf: List[str] = []
    current = "Overview"
    def flush():
        nonlocal buf, current
        text = "\n".join([x.strip() for x in buf if x.strip()])
        if text:
            sections.append((current, text))
        buf = []

    for el in main.descendants:
        if getattr(el, "name", None) in ("h2","h3"):
            flush()
            current = el.get_text(" ", strip=True)
        elif getattr(el, "name", None) in ("p","li"):
            buf.append(el.get_text(" ", strip=True))
        elif getattr(el, "name", None) == "pre":
            buf.append(el.get_text("\n", strip=True))
    flush()
    if DEBUG and not sections:
        dbg("no sections parsed from DOM")
    return [(h,b) for h,b in sections if b]

def chunk_text(s: str, size: int = 2000, overlap: int = 200) -> List[str]:
    if not s: return []
    chunks, i = [], 0
    while i < len(s):
        chunks.append(s[i:i+size])
        i += max(1, size - overlap)
    return chunks

# --- Embeddings & Upload ---
def embed_batch(texts: List[str]) -> List[List[float]]:
    if not texts: return []
    resp = _openai.embeddings.create(model=_embed_deployment, input=texts)
    return [d.embedding for d in resp.data]

def upsert_docs(docs: List[Dict]):
    if not docs:
        return
    try:
        dbg("upload_documents count:", len(docs))
        _search.upload_documents(docs)
    except HttpResponseError as e:
        # Provide clearer guidance for Forbidden (403)
        if getattr(e, 'status_code', None) == 403:
            print("\n[ingest-error] 403 Forbidden from Azure AI Search while uploading documents.\n"
                  "Likely causes: (1) using a QUERY key instead of ADMIN key;\n"
                  "(2) using AAD without the 'Search Index Data Contributor' role;\n"
                  "(3) service firewall/public network access blocks your client IP.\n"
                  "Fix: set SEARCH_API_KEY to an admin key OR grant RBAC on the Search service and allow your IP.")
        raise

# --- Pipeline ---
def run(max_pages: int = 500, sleep_sec: float = 0.2):
    state = load_state()
    dbg("start crawl; browser=", USE_BROWSER, "reindex=", REINDEX_ALL)
    if str(os.getenv("INGEST_ENSURE_URL_FIELD", "1")).lower() not in ("0","false","no",""):
        ensure_index_has_url_field()
    pages = state["pages"]

    urls = discover_urls(max_pages=max_pages, sleep_sec=sleep_sec)
    total_chunks = total_upsert = 0

    for url in urls:
        meta = pages.get(url, {})
        status, new_meta, html = fetch(url, meta.get("etag"), meta.get("last_modified"))

        if status == 304:
            continue
        if status != 200 or not html:
            pages[url] = {"etag": new_meta.get("etag"), "last_modified": new_meta.get("last_modified"), "sections": meta.get("sections", {})}
            continue

        title, main = extract_title_and_main(html)
        secs = sections_from_dom(main)
        if not secs:
            pages[url] = {"etag": new_meta.get("etag"), "last_modified": new_meta.get("last_modified"), "sections": {}}
            continue

        parent_id_val = sha1(url)
        changed_docs, section_map = [], {}
        for heading, body in secs:
            section_key = heading
            body_hash = sha1(body)
            prev_hash = meta.get("sections", {}).get(section_key)
            if not REINDEX_ALL and prev_hash == body_hash:
                section_map[section_key] = body_hash
                continue

            # chunk + embed
            chunks = chunk_text(body, size=2000, overlap=200)
            vecs: List[List[float]] = []
            for i in range(0, len(chunks), 16):
                vecs.extend(embed_batch(chunks[i:i+16]))
            for idx, (c, v) in enumerate(zip(chunks, vecs)):
                doc_id = sha1(f"{url}#{section_key}-{idx}")
                changed_docs.append({
                    "chunk_id": doc_id,
                    "parent_id": parent_id_val,
                    "title": title or section_key,
                    "chunk": c,
                    "text_vector": v,
                    "url": url,
                    # "section": section_key,
                    # "source_type": "cyclopedia",
                })
            section_map[section_key] = body_hash
            total_chunks += len(chunks)

        if changed_docs:
            upsert_docs(changed_docs)
            total_upsert += len(changed_docs)

        pages[url] = {
            "etag": new_meta.get("etag"),
            "last_modified": new_meta.get("last_modified"),
            "sections": section_map,
        }
        save_state(state)
        time.sleep(sleep_sec)

    dbg("done")
    print(f"[cyclopedia] pages: {len(urls)}, new/updated chunks: {total_chunks}, upserts: {total_upsert}")

if __name__ == "__main__":
    run()