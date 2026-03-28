"""GROBID citation parsing client for CiteGuard.

Uses GROBID's processCitation endpoint to extract structured metadata
from raw citation strings. Falls back gracefully on any error.

Environment variables
---------------------
GROBID_URL : Base URL for GROBID API (default: public HF Space instance)
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import requests

from citation_extractor import Citation

GROBID_URL = os.getenv(
    "GROBID_URL",
    "https://kermitt2-grobid.hf.space",
)
GROBID_TIMEOUT = 15  # seconds per citation


def _parse_tei_citation(xml_text: str, index: int, raw_text: str) -> Optional[Citation]:
    """Parse GROBID TEI XML response into a Citation dataclass."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    # Handle both namespaced and non-namespaced responses
    def find(path: str) -> Optional[ET.Element]:
        el = root.find(path, ns)
        if el is None:
            # Try without namespace
            el = root.find(path.replace("tei:", ""))
        return el

    def findall(path: str) -> List[ET.Element]:
        els = root.findall(path, ns)
        if not els:
            els = root.findall(path.replace("tei:", ""))
        return els

    def text_or_none(path: str) -> Optional[str]:
        el = find(path)
        if el is not None and el.text:
            return el.text.strip()
        return None

    # Title: analytic/title or monogr/title
    title = text_or_none(".//tei:analytic/tei:title")
    if not title:
        title = text_or_none(".//tei:monogr/tei:title")

    # Authors
    authors = []
    for person in findall(".//tei:author/tei:persName"):
        forename_el = person.find("tei:forename", ns) or person.find("forename")
        surname_el = person.find("tei:surname", ns) or person.find("surname")
        forename = forename_el.text.strip() if forename_el is not None and forename_el.text else ""
        surname = surname_el.text.strip() if surname_el is not None and surname_el.text else ""
        if surname:
            authors.append(f"{surname}, {forename}".strip(", "))

    author_str = "; ".join(authors) if authors else None

    # Journal
    journal = text_or_none(".//tei:monogr/tei:title")

    # Volume, pages, year
    volume = None
    pages = None
    year = None
    doi = None

    for scope in findall(".//tei:biblScope"):
        unit = scope.get("unit", "")
        if unit == "volume" and scope.text:
            volume = scope.text.strip()
        elif unit == "page":
            from_page = scope.get("from", "")
            to_page = scope.get("to", "")
            if from_page and to_page:
                pages = f"{from_page}-{to_page}"
            elif from_page:
                pages = from_page
            elif scope.text:
                pages = scope.text.strip()

    date_el = find(".//tei:date[@type='published']")
    if date_el is None:
        date_el = find(".//tei:date")
    if date_el is not None:
        when = date_el.get("when", "")
        if when:
            year = when[:4]
        elif date_el.text:
            year = date_el.text.strip()[:4]

    # DOI (from idno element if present)
    for idno in findall(".//tei:idno"):
        if idno.get("type", "").upper() == "DOI" and idno.text:
            doi = idno.text.strip()

    return Citation(
        index=index,
        raw_text=raw_text,
        author=author_str,
        title=title if title else None,
        year=year,
        doi=doi,
        journal=journal,
        volume=volume,
        pages=pages,
    )


def parse_citation(raw_text: str, index: int = 1) -> Optional[Citation]:
    """Parse a single raw citation string via GROBID.

    Returns a Citation with structured fields, or None on any failure.
    """
    try:
        resp = requests.post(
            f"{GROBID_URL}/api/processCitation",
            data={"citations": raw_text, "consolidateCitations": "0"},
            headers={"Accept": "application/xml"},
            timeout=GROBID_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        return _parse_tei_citation(resp.text, index, raw_text)
    except (requests.RequestException, Exception):
        return None


def parse_citations_batch(
    raw_citations: List[str],
    start_index: int = 1,
    max_workers: int = 4,
) -> Dict[int, Optional[Citation]]:
    """Parse multiple citations via GROBID in parallel.

    Returns a dict mapping index -> Citation (or None for failures).
    """
    results: Dict[int, Optional[Citation]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(parse_citation, raw, start_index + i): start_index + i
            for i, raw in enumerate(raw_citations)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = None

    return results
