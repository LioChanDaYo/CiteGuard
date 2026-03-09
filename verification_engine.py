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


_LIGATURE_MAP = str.maketrans({
    '\ufb00': 'ff', '\ufb01': 'fi', '\ufb02': 'fl',
    '\ufb03': 'ffi', '\ufb04': 'ffl',
})


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: ligatures, punctuation, whitespace."""
    text = text.translate(_LIGATURE_MAP)
    text = re.sub(r'[^\w\s]', '', text.lower().strip())
    return re.sub(r'\s+', ' ', text)


def _title_similarity(a: str, b: str) -> float:
    """Normalized similarity between two title strings."""
    a_clean = _normalize_text(a)
    b_clean = _normalize_text(b)
    return SequenceMatcher(None, a_clean, b_clean).ratio()


def _author_similarity(extracted: str, api_authors: List[str]) -> float:
    """Check if extracted author string overlaps with API author list.

    Returns a score 0.0–1.0 based on how many API author last names appear
    in the extracted author string.
    """
    if not extracted or not api_authors:
        return 0.0
    extracted_lower = extracted.lower()
    # Also check against raw_text if provided (stored in extracted field)
    matched = 0
    for author in api_authors:
        # Extract last name — handle both "Last, First" and "First Last" formats
        parts = author.split(',')
        last_name = parts[0].strip().lower()
        # Also try last word (for "First Last" format from API)
        words = author.strip().split()
        last_word = words[-1].lower() if words else ""
        # Match if either form of last name appears
        if (len(last_name) >= 3 and last_name in extracted_lower) or \
           (len(last_word) >= 3 and last_word in extracted_lower):
            matched += 1
    return matched / len(api_authors)


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


def _build_search_query(citation: Citation) -> str:
    """Build a clean search query from citation fields.

    Prioritizes title, adds a short author snippet. Avoids sending
    journal/venue info or excessively long queries to CrossRef.
    """
    query_parts: List[str] = []

    if citation.title:
        # Use title as primary query (truncate if too long)
        title = citation.title[:150]
        query_parts.append(title)

    if citation.author:
        # Only use first author's last name to avoid noise
        author = citation.author
        # Take just the first author (before first comma that separates authors, or "and")
        first_author = re.split(r',\s*(?=[A-Z])|(?:\s+and\s+)', author)[0].strip()
        # Further clean: just last name (first word for "Last, F." or last word for "First Last")
        words = first_author.split()
        if words:
            # If it looks like "Last, F." take the first word
            last_name = words[0].rstrip(',').rstrip('.')
            if len(last_name) >= 2:
                query_parts.append(last_name)

    if not query_parts:
        # No title or author — use cleaned raw text as query
        # This helps with physics compact format: "Author, Journal Vol, Page (Year)"
        raw = citation.raw_text
        # Clean: remove URLs, DOIs, page numbers, excessive punctuation
        raw_clean = re.sub(r'https?://\S+', '', raw)
        raw_clean = re.sub(r'10\.\d{4,9}/\S+', '', raw_clean)
        raw_clean = re.sub(r'\s+', ' ', raw_clean).strip()
        if raw_clean:
            query_parts.append(raw_clean[:150])

    query = " ".join(query_parts)
    # Truncate to avoid overly long API queries
    return query[:200]


def _verify_by_search(citation: Citation) -> VerificationResult:
    """Verify a citation by searching CrossRef with title + author."""
    query = _build_search_query(citation)

    if not query or len(query.strip()) < 8:
        return VerificationResult.from_citation(citation, "not_found", 0, _NOT_FOUND)

    try:
        items = crossref_search(query, rows=5, timeout=15, retries=2)
    except (HTTPRequestError, SystemExit):
        return VerificationResult.from_citation(citation, "not_found", 0, _NOT_FOUND)

    best_score = 0.0
    best_sim = 0.0
    best_item: Optional[Dict[str, Any]] = None

    for item in items:
        item_title = _extract_title_from_crossref(item)
        if not item_title:
            continue

        # Title similarity is the primary signal
        if citation.title:
            sim = _title_similarity(citation.title, item_title)
        else:
            # No extracted title — check if API result title appears in raw text
            raw_clean = _normalize_text(citation.raw_text)
            api_title_clean = _normalize_text(item_title)
            if api_title_clean and api_title_clean in raw_clean:
                sim = 0.85  # Strong match: API title is substring of raw text
            else:
                sim = _title_similarity(citation.raw_text[:200], item_title)

        # Author similarity as secondary signal
        item_authors = _extract_authors_from_crossref(item)
        author_sim = _author_similarity(citation.author or "", item_authors)

        # Combined score: title is primary, author is a boost/penalty
        score = sim * 0.8 + author_sim * 0.2

        if score > best_score:
            best_score = score
            best_sim = sim
            best_item = item

    # Threshold: 0.35 balances false positives vs missed real matches
    # (0.3 was too low → false positives, 0.5 was too high → missed real matches)
    if best_item is None or best_sim < 0.35:
        return VerificationResult.from_citation(citation, "not_found", 0, _NOT_FOUND)

    matched_title = _extract_title_from_crossref(best_item)
    matched_authors = _extract_authors_from_crossref(best_item)
    matched_doi = best_item.get("DOI")

    # Author matching adjusts confidence (boosts only, no penalties for mismatch
    # since extracted author field is often incomplete/partial)
    # Check against both extracted author AND raw text for best match
    author_sim = _author_similarity(citation.author or "", matched_authors)
    raw_author_sim = _author_similarity(citation.raw_text[:300], matched_authors)
    author_sim = max(author_sim, raw_author_sim)

    if best_sim >= 0.9:
        confidence = 90
    elif best_sim >= 0.7:
        confidence = int(best_sim * 100)
        # Author match boosts confidence into verified range
        if author_sim >= 0.2:
            confidence = min(90, confidence + 8)
    elif best_sim >= 0.5:
        confidence = int(best_sim * 100)
        # In the 0.5-0.7 range, author matching helps a lot
        if author_sim >= 0.5:
            confidence = min(85, confidence + 20)
        elif author_sim >= 0.2:
            confidence = min(80, confidence + 12)
    else:
        confidence = int(best_sim * 100)
        # Below 0.5 title sim — only trust if strong author match
        if author_sim >= 0.5:
            confidence = min(75, confidence + 25)
        elif author_sim >= 0.2:
            confidence = min(65, confidence + 10)

    verdict = "verified" if confidence >= 70 else "suspect"

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


def _extract_arxiv_from_raw(raw_text: str) -> Optional[str]:
    """Try to find an arXiv ID in the raw citation text."""
    # Match arXiv URLs or explicit IDs
    m = re.search(r'arxiv[.:/\s]+(?:abs/|pdf/)?(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+/\d{7}(?:v\d+)?)', raw_text, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _verify_one(cit: Citation) -> VerificationResult:
    """Verify a single citation through the DOI → arXiv → ISBN → search cascade."""
    # Skip tiny/garbage citations
    if len(cit.raw_text.strip()) < 15:
        return VerificationResult.from_citation(cit, "not_found", 0, _NOT_FOUND)

    result = _verify_by_doi(cit)
    if result:
        return result
    result = _verify_by_arxiv(cit)
    if result:
        return result
    # Try extracting arXiv ID from raw text even if not in structured field
    if not cit.arxiv_id:
        raw_arxiv = _extract_arxiv_from_raw(cit.raw_text)
        if raw_arxiv:
            cit_with_arxiv = Citation(
                index=cit.index, raw_text=cit.raw_text,
                author=cit.author, title=cit.title, year=cit.year,
                doi=cit.doi, isbn=cit.isbn, arxiv_id=raw_arxiv,
            )
            result = _verify_by_arxiv(cit_with_arxiv)
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
