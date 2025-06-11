#!/usr/bin/env python3
"""Minimal bibliographic client: CrossRef + Open Library

Version 0.5 – 2025‑06‑11
-----------------------
* **CLI UX fix** : `--timeout` et `--retries` sont désormais reconnus *après* le
  sous‑commande (grâce à un « parent parser » partagé). Les exemples reflètent
  cette syntaxe naturelle ;
* Conserve : timeout par défaut 30 s, retry exponentiel, ISBN via Open Library.

Features
--------
* **Search CrossRef** records by free‑text query (articles, books, proceedings…).
* **Lookup CrossRef** metadata by DOI.
* **Lookup Open Library** metadata by ISBN.

Environment variables
---------------------
CROSSREF_MAIL : e‑mail address inserted in the User‑Agent header for CrossRef requests.
               Defaults to "anonymous@example.com" if unset.

Examples (Version 0.5)
----------------------
Search :
    python crossref_api_client.py search -q "histoire numérique" -n 3 --retries 3

DOI :
    python crossref_api_client.py doi --doi 10.1038/s41586-024-07031-2 --timeout 45

ISBN :
    python crossref_api_client.py isbn --isbn 9782070368228 --retries 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_NAME = "CiteGuard"
APP_VERSION = "0.5"
DEFAULT_EMAIL = "anonymous@example.com"
EMAIL = os.getenv("CROSSREF_MAIL", DEFAULT_EMAIL)
USER_AGENT = f"{APP_NAME}/{APP_VERSION} (mailto:{EMAIL})"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
}

CROSSREF_BASE_URL = "https://api.crossref.org"
OPENLIB_BASE_URL = "https://openlibrary.org/api/books"
DEFAULT_TIMEOUT = 30  # seconds
DEFAULT_RETRIES = 3
BACKOFF_FACTOR = 1  # seconds (exponential: 1,2,4…)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class HTTPRequestError(SystemExit):
    """Custom exit for HTTP/network issues."""


def _request_with_retries(
    *,
    method: str,
    url: str,
    headers: Dict[str, str],
    params: Dict[str, Any] | None = None,
    timeout: int,
    retries: int,
) -> requests.Response:
    """Simple exponential‑backoff retry wrapper around ``requests``."""
    for attempt in range(retries):
        try:
            resp = requests.request(method, url, headers=headers, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(BACKOFF_FACTOR * (2 ** attempt))
                continue
            raise HTTPRequestError(
                f"[Error] Request timed out after {timeout}s (attempt {attempt + 1}/{retries}) – giving up."
            )
        except requests.exceptions.HTTPError as exc:
            # 500/502/503 can be retried; 4xx usually not.
            if exc.response.status_code >= 500 and attempt < retries - 1:
                time.sleep(BACKOFF_FACTOR * (2 ** attempt))
                continue
            raise HTTPRequestError(
                f"[Error] HTTP {exc.response.status_code} {exc.response.reason} – {url}"
            )
        except requests.exceptions.RequestException as exc:
            if attempt < retries - 1:
                time.sleep(BACKOFF_FACTOR * (2 ** attempt))
                continue
            raise HTTPRequestError(f"[Error] Network error: {exc}")

    # Static analysers (e.g., Pylance) need an explicit path showing that the
    # function never returns ``None``.
    raise HTTPRequestError("[Bug] Reached end of _request_with_retries without returning.")


# ---------------------------------------------------------------------------
# CrossRef helpers
# ---------------------------------------------------------------------------

def crossref_search(
    query: str,
    rows: int = 20,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> List[Dict[str, Any]]:
    """Search CrossRef works via full‑text query."""
    params = {"query": query, "rows": rows}
    resp = _request_with_retries(
        method="GET",
        url=f"{CROSSREF_BASE_URL}/works",
        headers=HEADERS,
        params=params,
        timeout=timeout,
        retries=retries,
    )
    items = resp.json().get("message", {}).get("items", [])
    if not items:
        raise SystemExit("[Info] No results returned – check your query.")
    return items


def crossref_get(
    doi: str,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> Dict[str, Any]:
    """Retrieve one work record by DOI from CrossRef."""
    try:
        resp = _request_with_retries(
            method="GET",
            url=f"{CROSSREF_BASE_URL}/works/{doi}",
            headers=HEADERS,
            timeout=timeout,
            retries=retries,
        )
    except HTTPRequestError as exc:
        # Map 404 to specific message
        if "HTTP 404" in str(exc):
            raise SystemExit(f"[Error] DOI not found on CrossRef: {doi}")
        raise
    return resp.json()["message"]

# ---------------------------------------------------------------------------
# Open Library helpers (ISBN)
# ---------------------------------------------------------------------------

def openlib_get(
    isbn: str,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> Dict[str, Any]:
    """Retrieve book metadata by ISBN via Open Library."""
    isbn_clean = isbn.replace("-", "").strip()
    params = {"bibkeys": f"ISBN:{isbn_clean}", "format": "json", "jscmd": "data"}
    resp = _request_with_retries(
        method="GET",
        url=OPENLIB_BASE_URL,
        headers={"User-Agent": USER_AGENT},
        params=params,
        timeout=timeout,
        retries=retries,
    )
    data = resp.json()
    return data.get(f"ISBN:{isbn_clean}", {})

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def pretty_print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal CrossRef / Open Library client")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Parent parser with shared options so that they are recognised *after* the
    # sub‑command (argparse pattern).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds (default: 30)")
    common.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Number of retries on timeout/5xx (default: 3)")

    p_search = subparsers.add_parser(
        "search",
        parents=[common],
        help="Search CrossRef works (free‑text query)",
    )
    p_search.add_argument("--query", "-q", required=True, help="Search query string")
    p_search.add_argument("--rows", "-n", type=int, default=20, help="Number of results to return (max 1000)")

    p_doi = subparsers.add_parser(
        "doi",
        parents=[common],
        help="Get CrossRef work metadata by DOI",
    )
    p_doi.add_argument("--doi", required=True, help="DOI of the work")

    p_isbn = subparsers.add_parser(
        "isbn",
        parents=[common],
        help="Get book metadata by ISBN (Open Library)",
    )
    p_isbn.add_argument("--isbn", required=True, help="ISBN‑10 or ISBN‑13 of the book")

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "search":
        pretty_print(crossref_search(args.query, args.rows, args.timeout, args.retries))
    elif args.command == "doi":
        pretty_print(crossref_get(args.doi, args.timeout, args.retries))
    elif args.command == "isbn":
        pretty_print(openlib_get(args.isbn, args.timeout, args.retries))
    else:
        parser.error("Unknown command")

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main(sys.argv[1:])
