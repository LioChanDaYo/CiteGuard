#!/usr/bin/env python3
"""Minimal CrossRef REST API client.

Supports basic search by query and DOI lookup.

Environment variables
---------------------
CROSSREF_MAIL : e-mail address inserted in the User‑Agent header.
    Defaults to "anonymous@example.com" if unset.

Example
-------
Search for works::

    python crossref_api_client.py search --query "matérialité des archives numériques" --rows 5

Fetch metadata for a specific DOI::

    python crossref_api_client.py get --doi 10.1234/exampledoi
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import requests

BASE_URL = "https://api.crossref.org"

# -- configuration ----------------------------------------------------------
APP_NAME = "CiteGuard"
APP_VERSION = "0.1"
DEFAULT_EMAIL = "anonymous@example.com"
EMAIL = os.getenv("CROSSREF_MAIL", DEFAULT_EMAIL)
USER_AGENT = f"{APP_NAME}/{APP_VERSION} (mailto:{EMAIL})"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
}

def search_works(query: str, rows: int = 20) -> List[Dict[str, Any]]:
    """Search CrossRef works via full-text query."""
    params = {"query": query, "rows": rows}
    resp = requests.get(f"{BASE_URL}/works", params=params, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()["message"]["items"]

def get_work(doi: str) -> Dict[str, Any]:
    """Retrieve one work record by DOI."""
    resp = requests.get(f"{BASE_URL}/works/{doi}", headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()["message"]

def pretty_print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))

def main(argv=None):
    parser = argparse.ArgumentParser(description="Minimal CrossRef REST API client")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_search = subparsers.add_parser("search", help="Search works")
    p_search.add_argument("--query", "-q", required=True, help="Search query string")
    p_search.add_argument("--rows", "-n", type=int, default=20, help="Number of results to return")

    p_get = subparsers.add_parser("get", help="Get work metadata by DOI")
    p_get.add_argument("--doi", required=True, help="DOI of the work")

    args = parser.parse_args(argv)

    if args.command == "search":
        pretty_print(search_works(args.query, args.rows))
    elif args.command == "get":
        pretty_print(get_work(args.doi))

if __name__ == "__main__":
    main(sys.argv[1:])
