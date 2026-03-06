"""Verification engine: checks extracted citations against CrossRef and OpenLibrary.

Assigns each citation a verdict (verified / suspect / not_found) and confidence score.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import xml.etree.ElementTree as ET

import requests as _requests

from citation_extractor import Citation
from crossref_api_client import (
    crossref_get,
    crossref_search,
    openlib_get,
    HTTPRequestError,
)


@dataclass
class MatchDetails:
    source: str  # "crossref", "openlibrary", or "none"
    matched_title: Optional[str] = None
    matched_authors: Optional[List[str]] = None
    matched_doi: Optional[str] = None
    matched_isbn: Optional[str] = None


@dataclass
class VerificationResult:
    index: int
    raw_text: str
    author: Optional[str]
    title: Optional[str]
    year: Optional[str]
    doi: Optional[str]
    isbn: Optional[str]
    verdict: str  # "verified", "suspect", "not_found"
    confidence: int  # 0-100
    match_details: MatchDetails

    @classmethod
    def from_citation(
        cls,
        cit: Citation,
        verdict: str,
        confidence: int,
        match_details: MatchDetails,
    ) -> VerificationResult:
        return cls(
            index=cit.index,
            raw_text=cit.raw_text,
            author=cit.author,
            title=cit.title,
            year=cit.year,
            doi=cit.doi,
            isbn=cit.isbn,
            verdict=verdict,
            confidence=confidence,
            match_details=match_details,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_NOT_FOUND = MatchDetails(source="none")


def _title_similarity(a: str, b: str) -> float:
    """Normalized similarity between two title strings."""
    a_clean = re.sub(r'[^\w\s]', '', a.lower().strip())
    b_clean = re.sub(r'[^\w\s]', '', b.lower().strip())
    return SequenceMatcher(None, a_clean, b_clean).ratio()


def _verdict_from_confidence(confidence: int) -> str:
    return "verified" if confidence >= 70 else "suspect"


def _extract_authors_from_crossref(item: Dict[str, Any]) -> List[str]:
    """Pull author names from a CrossRef work record."""
    authors = []
    for a in item.get("author", []):
        given = a.get("given", "")
        family = a.get("family", "")
        if family:
            authors.append(f"{family}, {given}".strip(", "))
    return authors


def _extract_title_from_crossref(item: Dict[str, Any]) -> str:
    """Pull the first title from a CrossRef work record."""
    titles = item.get("title", [])
    return titles[0] if titles else ""


def _verify_by_doi(citation: Citation) -> Optional[VerificationResult]:
    """Verify a citation that has a DOI."""
    if not citation.doi:
        return None
    try:
        record = crossref_get(citation.doi, timeout=15, retries=2)
    except (HTTPRequestError, SystemExit):
        return VerificationResult.from_citation(citation, "not_found", 0, _NOT_FOUND)

    matched_title = _extract_title_from_crossref(record)
    matched_authors = _extract_authors_from_crossref(record)
    confidence = 95

    if citation.title and matched_title:
        sim = _title_similarity(citation.title, matched_title)
        if sim < 0.5:
            confidence = 60
        elif sim < 0.8:
            confidence = 80

    return VerificationResult.from_citation(
        citation,
        _verdict_from_confidence(confidence),
        confidence,
        MatchDetails(
            source="crossref",
            matched_title=matched_title,
            matched_authors=matched_authors,
            matched_doi=citation.doi,
        ),
    )


def _verify_by_isbn(citation: Citation) -> Optional[VerificationResult]:
    """Verify a citation that has an ISBN."""
    if not citation.isbn:
        return None
    try:
        record = openlib_get(citation.isbn, timeout=15, retries=2)
    except (HTTPRequestError, SystemExit):
        return None

    if not record:
        return None

    matched_title = record.get("title", "")
    matched_authors = [a.get("name", "") for a in record.get("authors", [])]
    confidence = 90

    if citation.title and matched_title:
        sim = _title_similarity(citation.title, matched_title)
        if sim < 0.5:
            confidence = 55
        elif sim < 0.8:
            confidence = 75

    return VerificationResult.from_citation(
        citation,
        _verdict_from_confidence(confidence),
        confidence,
        MatchDetails(
            source="openlibrary",
            matched_title=matched_title,
            matched_authors=matched_authors,
            matched_isbn=citation.isbn,
        ),
    )


def _verify_by_arxiv(citation: Citation) -> Optional[VerificationResult]:
    """Verify a citation that has an arXiv ID."""
    arxiv_id = getattr(citation, 'arxiv_id', None)
    if not arxiv_id:
        return None
    try:
        resp = _requests.get(
            f"http://export.arxiv.org/api/query?id_list={arxiv_id}",
            timeout=15,
        )
        resp.raise_for_status()
    except Exception:
        return None

    try:
        root = ET.fromstring(resp.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = root.find('atom:entry', ns)
        if entry is None:
            return None
        matched_title = (entry.findtext('atom:title', '', ns) or '').strip()
        matched_title = re.sub(r'\s+', ' ', matched_title)
        authors = [
            (a.findtext('atom:name', '', ns) or '').strip()
            for a in entry.findall('atom:author', ns)
        ]
        if not matched_title:
            return None
    except ET.ParseError:
        return None

    confidence = 92
    if citation.title and matched_title:
        sim = _title_similarity(citation.title, matched_title)
        if sim < 0.5:
            confidence = 55
        elif sim < 0.8:
            confidence = 78

    return VerificationResult.from_citation(
        citation,
        _verdict_from_confidence(confidence),
        confidence,
        MatchDetails(
            source="arxiv",
            matched_title=matched_title,
            matched_authors=authors,
        ),
    )


def _verify_by_search(citation: Citation) -> VerificationResult:
    """Verify a citation by searching CrossRef with title + author."""
    query_parts: List[str] = []
    if citation.title:
        query_parts.append(citation.title)
    if citation.author:
        query_parts.append(citation.author)

    if not query_parts:
        return VerificationResult.from_citation(citation, "not_found", 0, _NOT_FOUND)

    query = " ".join(query_parts)

    try:
        items = crossref_search(query, rows=5, timeout=15, retries=2)
    except (HTTPRequestError, SystemExit):
        return VerificationResult.from_citation(citation, "not_found", 0, _NOT_FOUND)

    best_sim = 0.0
    best_item: Optional[Dict[str, Any]] = None

    for item in items:
        item_title = _extract_title_from_crossref(item)
        if citation.title and item_title:
            sim = _title_similarity(citation.title, item_title)
            if sim > best_sim:
                best_sim = sim
                best_item = item

    if best_item is None or best_sim < 0.3:
        return VerificationResult.from_citation(citation, "not_found", 0, _NOT_FOUND)

    matched_title = _extract_title_from_crossref(best_item)
    matched_authors = _extract_authors_from_crossref(best_item)
    matched_doi = best_item.get("DOI")

    confidence = min(90, int(best_sim * 100)) if best_sim >= 0.9 else int(best_sim * 100)
    verdict = "verified" if best_sim >= 0.7 else "suspect"

    return VerificationResult.from_citation(
        citation,
        verdict,
        confidence,
        MatchDetails(
            source="crossref",
            matched_title=matched_title,
            matched_authors=matched_authors,
            matched_doi=matched_doi,
        ),
    )


def _verify_one(cit: Citation) -> VerificationResult:
    """Verify a single citation through the DOI → arXiv → ISBN → search cascade."""
    result = _verify_by_doi(cit)
    if result:
        return result
    result = _verify_by_arxiv(cit)
    if result:
        return result
    result = _verify_by_isbn(cit)
    if result:
        return result
    return _verify_by_search(cit)


def verify_citations(citations: List[Citation]) -> List[VerificationResult]:
    """Verify citations in parallel using a thread pool."""
    if not citations:
        return []

    results: Dict[int, VerificationResult] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(citations))) as pool:
        futures = {pool.submit(_verify_one, cit): cit.index for cit in citations}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return [results[i] for i in sorted(results)]


def compute_summary(results: List[VerificationResult]) -> Dict[str, Any]:
    """Compute summary statistics from verification results."""
    total = len(results)
    verified = sum(1 for r in results if r.verdict == "verified")
    suspect = sum(1 for r in results if r.verdict == "suspect")
    not_found = sum(1 for r in results if r.verdict == "not_found")

    if total == 0:
        risk_level = "unknown"
    elif not_found / total > 0.5:
        risk_level = "high"
    elif (not_found + suspect) / total > 0.3:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "total": total,
        "verified": verified,
        "suspect": suspect,
        "not_found": not_found,
        "risk_level": risk_level,
    }
