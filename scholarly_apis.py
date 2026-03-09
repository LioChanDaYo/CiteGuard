"""Multi-source scholarly API clients for CiteGuard.

Provides lookup and search functions for:
  Semantic Scholar, OpenAlex, PubMed (ESearch/ECitMatch/EFetch),
  INSPIRE-HEP, NASA ADS, DBLP, zbMATH, Unpaywall, Retraction Watch.

All functions return Optional[dict], Optional[List[dict]], or Optional[str]
and never raise on network/API errors — they return None or an empty list.

Environment variables
---------------------
CROSSREF_MAIL   : polite-pool e-mail for OpenAlex / Unpaywall (default: anonymous@example.com)
ADS_API_KEY     : Bearer token for NASA ADS (optional — functions degrade to None)
NCBI_API_KEY    : NCBI E-utilities key for 10 req/s instead of 3 (optional)
"""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from urllib.parse import quote, quote_plus

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_NAME = "CiteGuard"
APP_VERSION = "0.5"
DEFAULT_EMAIL = "anonymous@example.com"
EMAIL = os.getenv("CROSSREF_MAIL", DEFAULT_EMAIL)
USER_AGENT = f"{APP_NAME}/{APP_VERSION} (mailto:{EMAIL})"

DEFAULT_TIMEOUT = 15  # seconds
DEFAULT_RETRIES = 2
BACKOFF_FACTOR = 1  # seconds (exponential: 1, 2, 4…)


# ---------------------------------------------------------------------------
# Retry helper (mirrors crossref_api_client._request_with_retries but
# returns None instead of raising on failure)
# ---------------------------------------------------------------------------

def _request_with_retries(
    *,
    method: str = "GET",
    url: str,
    headers: Dict[str, str] | None = None,
    params: Dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> Optional[requests.Response]:
    """Exponential-backoff retry wrapper. Returns None on any failure."""
    for attempt in range(retries):
        try:
            resp = requests.request(
                method, url, headers=headers, params=params, timeout=timeout,
            )
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException:
            if attempt < retries - 1:
                time.sleep(BACKOFF_FACTOR * (2 ** attempt))
                continue
            return None
    return None


# ---------------------------------------------------------------------------
# 1. Semantic Scholar
# ---------------------------------------------------------------------------

def semantic_scholar_search(title: str, limit: int = 5) -> List[dict]:
    """Search Semantic Scholar by title. Returns a list of paper dicts."""
    resp = _request_with_retries(
        url="https://api.semanticscholar.org/graph/v1/paper/search",
        headers={"User-Agent": USER_AGENT},
        params={
            "query": title,
            "limit": limit,
            "fields": "title,authors,year,venue,externalIds",
        },
    )
    if resp is None:
        return []
    try:
        return resp.json().get("data", []) or []
    except (ValueError, KeyError):
        return []


def semantic_scholar_by_doi(doi: str) -> Optional[dict]:
    """Look up a single paper on Semantic Scholar by DOI."""
    resp = _request_with_retries(
        url=f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(doi, safe='')}",
        headers={"User-Agent": USER_AGENT},
        params={"fields": "title,authors,year,venue,externalIds"},
    )
    if resp is None:
        return None
    try:
        return resp.json()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 2. OpenAlex
# ---------------------------------------------------------------------------

def openalex_search(title: str, limit: int = 5) -> List[dict]:
    """Search OpenAlex works by title."""
    params: Dict[str, Any] = {
        "search": title,
        "per_page": limit,
    }
    if EMAIL != DEFAULT_EMAIL:
        params["mailto"] = EMAIL
    resp = _request_with_retries(
        url="https://api.openalex.org/works",
        headers={"User-Agent": USER_AGENT},
        params=params,
    )
    if resp is None:
        return []
    try:
        return resp.json().get("results", []) or []
    except (ValueError, KeyError):
        return []


# ---------------------------------------------------------------------------
# 3. PubMed ESearch
# ---------------------------------------------------------------------------

def pubmed_search(title: str, author: str = "") -> List[str]:
    """Search PubMed by title (and optionally author). Returns list of PMIDs."""
    term = f"{title}[Title]"
    if author:
        term += f" AND {author}[Author]"
    params: Dict[str, Any] = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": 5,
    }
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    resp = _request_with_retries(
        url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        headers={"User-Agent": USER_AGENT},
        params=params,
    )
    if resp is None:
        return []
    try:
        return resp.json().get("esearchresult", {}).get("idlist", []) or []
    except (ValueError, KeyError):
        return []


# ---------------------------------------------------------------------------
# 4. PubMed ECitMatch
# ---------------------------------------------------------------------------

def pubmed_ecitmatch(
    journal: str,
    year: str,
    volume: str,
    first_page: str,
    author: str = "",
) -> Optional[str]:
    """Match a citation to a PMID via PubMed ECitMatch. Returns PMID or None."""
    bdata = f"{journal}|{year}|{volume}|{first_page}|{author}|key\r"
    params: Dict[str, Any] = {
        "db": "pubmed",
        "rettype": "xml",
        "bdata": bdata,
    }
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    resp = _request_with_retries(
        url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/ecitmatch.cgi",
        headers={"User-Agent": USER_AGENT},
        params=params,
    )
    if resp is None:
        return None
    try:
        # Response is pipe-delimited text: journal|year|vol|page|author|key|PMID
        text = resp.text.strip()
        parts = text.split("|")
        pmid = parts[-1].strip() if parts else ""
        return pmid if pmid and pmid.lower() != "not found" else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 5. PubMed EFetch
# ---------------------------------------------------------------------------

def pubmed_fetch(pmid: str) -> Optional[dict]:
    """Fetch article metadata from PubMed by PMID. Returns parsed dict or None."""
    params: Dict[str, Any] = {
        "db": "pubmed",
        "id": pmid,
        "rettype": "xml",
    }
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    resp = _request_with_retries(
        url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        headers={"User-Agent": USER_AGENT},
        params=params,
    )
    if resp is None:
        return None
    try:
        root = ET.fromstring(resp.content)
        article = root.find(".//Article")
        if article is None:
            return None

        title_el = article.find("ArticleTitle")
        title = (title_el.text or "") if title_el is not None else ""

        authors: List[str] = []
        for au in article.findall(".//Author"):
            last = au.findtext("LastName", "")
            fore = au.findtext("ForeName", "")
            name = f"{last}, {fore}".strip(", ")
            if name:
                authors.append(name)

        journal_el = article.find("Journal")
        journal_title = ""
        if journal_el is not None:
            jt = journal_el.find("Title")
            if jt is None:
                jt = journal_el.find("ISOAbbreviation")
            journal_title = (jt.text or "") if jt is not None else ""

        return {
            "pmid": pmid,
            "title": title,
            "authors": authors,
            "journal": journal_title,
        }
    except ET.ParseError:
        return None


# ---------------------------------------------------------------------------
# 6. INSPIRE-HEP
# ---------------------------------------------------------------------------

def inspirehep_search_journal(
    journal: str, volume: str, page: str,
) -> Optional[dict]:
    """Search INSPIRE-HEP by journal reference (journal, volume, page)."""
    query = f'find j "{journal},{volume},{page}"'
    resp = _request_with_retries(
        url="https://inspirehep.net/api/literature",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        params={"q": query, "size": 1},
    )
    if resp is None:
        return None
    try:
        hits = resp.json().get("hits", {}).get("hits", [])
        if not hits:
            return None
        return _parse_inspire_hit(hits[0])
    except (ValueError, KeyError, IndexError):
        return None


def inspirehep_search_title(title: str) -> List[dict]:
    """Search INSPIRE-HEP by title string."""
    resp = _request_with_retries(
        url="https://inspirehep.net/api/literature",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        params={"q": title, "size": 5},
    )
    if resp is None:
        return []
    try:
        hits = resp.json().get("hits", {}).get("hits", [])
        return [_parse_inspire_hit(h) for h in hits if h]
    except (ValueError, KeyError):
        return []


def _parse_inspire_hit(hit: dict) -> dict:
    """Extract useful fields from a single INSPIRE-HEP hit."""
    meta = hit.get("metadata", {})
    authors_raw = meta.get("authors", [])
    authors = [a.get("full_name", "") for a in authors_raw[:20]]  # cap for sanity
    dois = [d.get("value", "") for d in meta.get("dois", [])]
    arxiv_ids = [e.get("value", "") for e in meta.get("arxiv_eprints", [])]
    return {
        "title": meta.get("titles", [{}])[0].get("title", "") if meta.get("titles") else "",
        "authors": authors,
        "doi": dois[0] if dois else None,
        "arxiv_id": arxiv_ids[0] if arxiv_ids else None,
    }


# ---------------------------------------------------------------------------
# 7. NASA ADS
# ---------------------------------------------------------------------------

def ads_search(title: str) -> List[dict]:
    """Search NASA ADS by title. Requires ADS_API_KEY env var."""
    api_key = os.getenv("ADS_API_KEY")
    if not api_key:
        return []
    resp = _request_with_retries(
        url="https://api.adsabs.harvard.edu/v1/search/query",
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": USER_AGENT,
        },
        params={
            "q": f'title:"{title}"',
            "fl": "title,author,doi,bibcode,year",
            "rows": 5,
        },
    )
    if resp is None:
        return []
    try:
        return resp.json().get("response", {}).get("docs", []) or []
    except (ValueError, KeyError):
        return []


# ---------------------------------------------------------------------------
# 8. DBLP
# ---------------------------------------------------------------------------

def dblp_search(title: str, limit: int = 5) -> List[dict]:
    """Search DBLP computer science bibliography by title."""
    resp = _request_with_retries(
        url="https://dblp.org/search/publ/api",
        headers={"User-Agent": USER_AGENT},
        params={"q": title, "format": "json", "h": limit},
    )
    if resp is None:
        return []
    try:
        hits = resp.json().get("result", {}).get("hits", {}).get("hit", [])
        results: List[dict] = []
        for h in hits:
            info = h.get("info", {})
            # authors can be a dict (single) or list
            authors_raw = info.get("authors", {}).get("author", [])
            if isinstance(authors_raw, dict):
                authors_raw = [authors_raw]
            authors = [
                a.get("text", a) if isinstance(a, dict) else str(a)
                for a in authors_raw
            ]
            results.append({
                "title": info.get("title", ""),
                "authors": authors,
                "year": info.get("year", ""),
                "venue": info.get("venue", ""),
                "doi": info.get("doi", ""),
                "url": info.get("url", ""),
            })
        return results
    except (ValueError, KeyError):
        return []


# ---------------------------------------------------------------------------
# 9. zbMATH
# ---------------------------------------------------------------------------

def zbmath_search(title: str, limit: int = 5) -> List[dict]:
    """Search zbMATH Open API by title."""
    resp = _request_with_retries(
        url="https://api.zbmath.org/v1/document/_structured_search",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        params={"title": title, "page_size": limit},
    )
    if resp is None:
        return []
    try:
        data = resp.json()
        results: List[dict] = []
        for item in data.get("result", []):
            authors = [a.get("name", "") for a in item.get("authors", [])]
            results.append({
                "title": item.get("title", ""),
                "authors": authors,
                "year": item.get("year", ""),
                "journal": item.get("source", {}).get("series", {}).get("title", ""),
                "msc_codes": [m.get("code", "") for m in item.get("msc", [])],
            })
        return results
    except (ValueError, KeyError):
        return []


# ---------------------------------------------------------------------------
# 10. Unpaywall
# ---------------------------------------------------------------------------

def unpaywall_lookup(doi: str) -> Optional[dict]:
    """Check Unpaywall for open-access availability of a DOI."""
    resp = _request_with_retries(
        url=f"https://api.unpaywall.org/v2/{quote(doi, safe='')}",
        headers={"User-Agent": USER_AGENT},
        params={"email": EMAIL},
    )
    if resp is None:
        return None
    try:
        data = resp.json()
        best_loc = data.get("best_oa_location") or {}
        return {
            "is_oa": data.get("is_oa", False),
            "oa_status": data.get("oa_status", ""),
            "best_oa_url": best_loc.get("url", ""),
            "best_oa_pdf": best_loc.get("url_for_pdf", ""),
        }
    except (ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# 11. Retraction Watch (via CrossRef update-to field)
# ---------------------------------------------------------------------------

def check_retraction(doi: str) -> Optional[bool]:
    """Check if a DOI has been retracted via CrossRef metadata.

    Returns True if retracted, False if not retracted, None if check failed.
    """
    resp = _request_with_retries(
        url=f"https://api.crossref.org/works/{quote(doi, safe='')}",
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    if resp is None:
        return None
    try:
        message = resp.json().get("message", {})
        updates = message.get("update-to", [])
        for update in updates:
            update_type = update.get("type", "").lower()
            if "retraction" in update_type:
                return True
        return False
    except (ValueError, KeyError):
        return None
