"""Verification engine: checks extracted citations against multiple scholarly databases.

Assigns each citation a verdict (verified / suspect / not_found) and confidence score.
Supports discipline-aware routing for physics, biomedical, math, CS, social science.
"""

from __future__ import annotations

import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import xml.etree.ElementTree as ET

import requests as _requests
from rapidfuzz import fuzz

from citation_extractor import Citation, detect_citation_style
from crossref_api_client import (
    crossref_get,
    crossref_search,
    openlib_get,
    HTTPRequestError,
)
from scholarly_apis import (
    semantic_scholar_search,
    semantic_scholar_by_doi,
    openalex_search,
    openlibrary_search,
    pubmed_search,
    pubmed_ecitmatch,
    pubmed_fetch,
    inspirehep_search_journal,
    inspirehep_search_title,
    ads_search,
    dblp_search,
    zbmath_search,
    hal_search,
    unpaywall_lookup,
    check_retraction,
)


@dataclass
class MatchDetails:
    source: str  # "crossref", "openlibrary", "semantic_scholar", "pubmed", etc.
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
    is_retracted: Optional[bool] = None
    oa_url: Optional[str] = None

    @classmethod
    def from_citation(
        cls,
        cit: Citation,
        verdict: str,
        confidence: int,
        match_details: MatchDetails,
        is_retracted: Optional[bool] = None,
        oa_url: Optional[str] = None,
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
            is_retracted=is_retracted,
            oa_url=oa_url,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_NOT_FOUND = MatchDetails(source="none")


# ---------------------------------------------------------------------------
# Text normalization (improved with NFKD, accent removal, article stripping)
# ---------------------------------------------------------------------------

_LIGATURE_MAP = str.maketrans({
    '\ufb00': 'ff', '\ufb01': 'fi', '\ufb02': 'fl',
    '\ufb03': 'ffi', '\ufb04': 'ffl',
})


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: NFKD, ligatures, accents, articles, punctuation."""
    # NFKD unicode normalization
    text = unicodedata.normalize("NFKD", text)
    # Expand ligatures
    text = text.translate(_LIGATURE_MAP)
    # Remove combining marks (accents)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Lowercase
    text = text.lower().strip()
    # Remove leading articles (English + French)
    # Note: 'l' and 'd' match French contractions (l'amour → "l amour", d'autres → "d autres")
    # because apostrophes become spaces after accent/combining-mark removal above.
    text = re.sub(r'^(the|a|an|le|la|les|l|un|une|des|du|de|d)\s+', '', text)
    # Strip punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse whitespace
    return re.sub(r'\s+', ' ', text).strip()


def _title_similarity(a: str, b: str) -> float:
    """Normalized similarity between two title strings using rapidfuzz."""
    a_clean = _normalize_text(a)
    b_clean = _normalize_text(b)
    if not a_clean or not b_clean:
        return 0.0
    # token_sort_ratio handles word reordering (common with subtitles)
    return fuzz.token_sort_ratio(a_clean, b_clean) / 100.0


def _author_similarity(extracted: str, api_authors: List[str]) -> float:
    """Check if extracted author string overlaps with API author list.

    Returns a score 0.0-1.0 based on how many API author last names appear
    in the extracted author string.
    """
    if not extracted or not api_authors:
        return 0.0
    extracted_lower = extracted.lower()
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


# ---------------------------------------------------------------------------
# Discipline-calibrated scoring weights
# ---------------------------------------------------------------------------

_DISCIPLINE_WEIGHTS = {
    "physics": {"title": 0.10, "author": 0.15, "year": 0.05, "journal": 0.25, "volume": 0.25, "page": 0.20},
    "biomedical": {"title": 0.40, "author": 0.20, "year": 0.10, "journal": 0.15, "pmid": 0.15},
    "math": {"title": 0.30, "author": 0.25, "year": 0.10, "journal": 0.20, "volume": 0.15},
    "cs": {"title": 0.50, "author": 0.25, "year": 0.10, "venue": 0.15},
    "social_science": {"title": 0.45, "author": 0.25, "year": 0.15, "venue": 0.15},
    "humanities": {"title": 0.40, "author": 0.20, "year": 0.10, "publisher": 0.15, "isbn": 0.15},
    "unknown": {"title": 0.50, "author": 0.25, "year": 0.10, "venue": 0.15},
}

# Map citation styles to discipline categories
_STYLE_TO_DISCIPLINE = {
    "ieee": "cs",
    "apa": "social_science",
    "vancouver": "biomedical",
    "physics": "physics",
    "acl": "cs",
    "chicago": "humanities",
    "unknown": "unknown",
}


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


# ---------------------------------------------------------------------------
# Direct-identifier verification (DOI, arXiv, ISBN)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Discipline-specific verification methods
# ---------------------------------------------------------------------------

def _verify_by_inspire(citation: Citation) -> Optional[VerificationResult]:
    """Verify via INSPIRE-HEP using journal+volume+page (physics)."""
    # Try journal+volume+page first (highest confidence for physics)
    if citation.journal and citation.volume and citation.pages:
        record = inspirehep_search_journal(citation.journal, citation.volume, citation.pages)
        if record and record.get("title"):
            return VerificationResult.from_citation(
                citation,
                "verified",
                95,
                MatchDetails(
                    source="inspirehep",
                    matched_title=record["title"],
                    matched_authors=record.get("authors", []),
                    matched_doi=record.get("doi"),
                ),
            )
    # Fallback: search by title
    if citation.title:
        results = inspirehep_search_title(citation.title)
        return _pick_best_match(citation, results, "inspirehep")
    return None


def _verify_by_ads(citation: Citation) -> Optional[VerificationResult]:
    """Verify via NASA ADS (astrophysics)."""
    if not citation.title:
        return None
    results = ads_search(citation.title)
    if not results:
        return None
    # ADS returns different format — normalize to our expected shape
    normalized = []
    for doc in results:
        normalized.append({
            "title": doc.get("title", [""])[0] if isinstance(doc.get("title"), list) else doc.get("title", ""),
            "authors": doc.get("author", []),
            "doi": doc.get("doi", [""])[0] if isinstance(doc.get("doi"), list) else doc.get("doi"),
        })
    return _pick_best_match(citation, normalized, "ads")


def _verify_by_pubmed(citation: Citation) -> Optional[VerificationResult]:
    """Verify via PubMed (biomedical)."""
    # Try ECitMatch first (journal+volume+page → PMID)
    if citation.journal and citation.volume and citation.pages:
        first_page = citation.pages.split("-")[0].strip()
        first_author = ""
        if citation.author:
            first_author = citation.author.split(",")[0].strip()
        pmid = pubmed_ecitmatch(citation.journal, citation.year or "", citation.volume, first_page, first_author)
        if pmid:
            record = pubmed_fetch(pmid)
            if record:
                return VerificationResult.from_citation(
                    citation,
                    "verified",
                    95,
                    MatchDetails(
                        source="pubmed",
                        matched_title=record.get("title", ""),
                        matched_authors=record.get("authors", []),
                    ),
                )

    # Fallback: search by title
    if citation.title:
        first_author = ""
        if citation.author:
            first_author = citation.author.split(",")[0].strip()
        pmids = pubmed_search(citation.title, first_author)
        if pmids:
            record = pubmed_fetch(pmids[0])
            if record and record.get("title"):
                sim = _title_similarity(citation.title, record["title"])
                if sim >= 0.6:
                    confidence = 90 if sim >= 0.85 else int(sim * 100) + 10
                    return VerificationResult.from_citation(
                        citation,
                        _verdict_from_confidence(confidence),
                        confidence,
                        MatchDetails(
                            source="pubmed",
                            matched_title=record["title"],
                            matched_authors=record.get("authors", []),
                        ),
                    )
    return None


def _verify_by_dblp(citation: Citation) -> Optional[VerificationResult]:
    """Verify via DBLP (computer science)."""
    if not citation.title:
        return None
    results = dblp_search(citation.title, limit=5)
    return _pick_best_match(citation, results, "dblp")


def _verify_by_zbmath(citation: Citation) -> Optional[VerificationResult]:
    """Verify via zbMATH (mathematics)."""
    if not citation.title:
        return None
    results = zbmath_search(citation.title, limit=5)
    return _pick_best_match(citation, results, "zbmath")


def _verify_by_hal(citation: Citation) -> Optional[VerificationResult]:
    """Verify via HAL (French national open archive, 4.5M+ documents)."""
    query = citation.title or _build_search_query(citation)
    if not query or len(query.strip()) < 8:
        return None
    author = citation.author or ""
    results = hal_search(query, author=author, limit=5)
    return _pick_best_match(citation, results, "hal")


def _verify_by_openlibrary_search(citation: Citation) -> Optional[VerificationResult]:
    """Verify books via Open Library title+author search (no ISBN needed)."""
    query = citation.title or _build_search_query(citation)
    if not query or len(query.strip()) < 8:
        return None
    author = citation.author or ""
    results = openlibrary_search(query, author=author, limit=3)
    return _pick_best_match(citation, results, "openlibrary")


def _verify_by_semantic_scholar(citation: Citation) -> Optional[VerificationResult]:
    """Verify via Semantic Scholar (cross-discipline)."""
    # Try DOI first
    if citation.doi:
        record = semantic_scholar_by_doi(citation.doi)
        if record and record.get("title"):
            authors = [a.get("name", "") for a in record.get("authors", [])]
            return VerificationResult.from_citation(
                citation,
                "verified",
                93,
                MatchDetails(
                    source="semantic_scholar",
                    matched_title=record["title"],
                    matched_authors=authors,
                    matched_doi=citation.doi,
                ),
            )
    # Fallback: search by title or raw text
    query = citation.title or _build_search_query(citation)
    if not query or len(query.strip()) < 8:
        return None
    results = semantic_scholar_search(query, limit=5)
    if not results:
        return None
    # Normalize Semantic Scholar format
    normalized = []
    for paper in results:
        authors = [a.get("name", "") for a in paper.get("authors", [])]
        ext_ids = paper.get("externalIds", {}) or {}
        normalized.append({
            "title": paper.get("title", ""),
            "authors": authors,
            "doi": ext_ids.get("DOI"),
        })
    return _pick_best_match(citation, normalized, "semantic_scholar")


def _verify_by_openalex(citation: Citation) -> Optional[VerificationResult]:
    """Verify via OpenAlex (broad cross-discipline)."""
    query = citation.title or _build_search_query(citation)
    if not query or len(query.strip()) < 8:
        return None
    results = openalex_search(query, limit=5)
    if not results:
        return None
    # Normalize OpenAlex format
    normalized = []
    for work in results:
        authors = []
        for authorship in work.get("authorships", []):
            author_obj = authorship.get("author", {})
            name = author_obj.get("display_name", "")
            if name:
                authors.append(name)
        doi_url = work.get("doi", "")
        doi = doi_url.replace("https://doi.org/", "") if doi_url else None
        normalized.append({
            "title": work.get("title", ""),
            "authors": authors,
            "doi": doi,
        })
    return _pick_best_match(citation, normalized, "openalex")


# ---------------------------------------------------------------------------
# Generic best-match picker (scores top results, flags ambiguity)
# ---------------------------------------------------------------------------

def _pick_best_match(
    citation: Citation,
    results: List[dict],
    source: str,
) -> Optional[VerificationResult]:
    """Score all results, pick the best match, flag ambiguous if top-2 gap < 0.10."""
    if not results:
        return None

    scored: List[tuple[float, float, dict]] = []  # (score, title_sim, result)

    for item in results:
        item_title = item.get("title", "")
        if not item_title:
            continue

        # Title similarity
        if citation.title:
            sim = _title_similarity(citation.title, item_title)
        else:
            raw_clean = _normalize_text(citation.raw_text[:200])
            api_title_clean = _normalize_text(item_title)
            if api_title_clean and api_title_clean in raw_clean:
                sim = 0.85
            else:
                sim = _title_similarity(citation.raw_text[:200], item_title)

        # Author similarity
        item_authors = item.get("authors", [])
        author_sim = _author_similarity(citation.author or "", item_authors)
        raw_author_sim = _author_similarity(citation.raw_text[:300], item_authors)
        author_sim = max(author_sim, raw_author_sim)

        score = sim * 0.8 + author_sim * 0.2
        scored.append((score, sim, item))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_sim, best_item = scored[0]

    # Threshold: 0.65 for title-based matching, 0.45 for raw-text fallback
    # Raised from 0.50/0.35 to reduce false positives on fabricated citations
    threshold = 0.65 if citation.title else 0.45
    if best_sim < threshold:
        return None

    matched_title = best_item.get("title", "")
    matched_authors = best_item.get("authors", [])
    matched_doi = best_item.get("doi")

    # Recompute author similarity for best match specifically
    best_authors = best_item.get("authors", [])
    best_author_sim = _author_similarity(citation.author or "", best_authors)
    best_raw_author_sim = _author_similarity(citation.raw_text[:300], best_authors)
    best_author_sim = max(best_author_sim, best_raw_author_sim)

    # Compute confidence from similarity
    if best_sim >= 0.9:
        confidence = 90
    elif best_sim >= 0.7:
        confidence = int(best_sim * 100)
        if best_author_sim >= 0.2:
            confidence = min(90, confidence + 8)
    elif best_sim >= 0.65:
        confidence = int(best_sim * 100)
        if best_author_sim >= 0.5:
            confidence = min(80, confidence + 15)
        elif best_author_sim >= 0.2:
            confidence = min(75, confidence + 8)
    else:
        confidence = int(best_sim * 100)
        if best_author_sim >= 0.5:
            confidence = min(70, confidence + 20)
        elif best_author_sim >= 0.2:
            confidence = min(60, confidence + 10)

    # Penalty: no author overlap at all → cap confidence (likely wrong match)
    if citation.author and best_authors and best_author_sim == 0.0:
        confidence = min(confidence, 55)

    # Penalty: year mismatch → reduce confidence
    if citation.year and best_item.get("year"):
        try:
            year_diff = abs(int(citation.year) - int(str(best_item["year"])[:4]))
            if year_diff >= 5:
                confidence = min(confidence, 50)
            elif year_diff >= 2:
                confidence = max(0, confidence - 10)
        except (ValueError, TypeError):
            pass

    verdict = _verdict_from_confidence(confidence)

    return VerificationResult.from_citation(
        citation,
        verdict,
        confidence,
        MatchDetails(
            source=source,
            matched_title=matched_title,
            matched_authors=matched_authors,
            matched_doi=matched_doi,
        ),
    )


# ---------------------------------------------------------------------------
# CrossRef free-text search (fallback)
# ---------------------------------------------------------------------------

def _build_search_query(citation: Citation) -> str:
    """Build a clean search query from citation fields."""
    query_parts: List[str] = []

    if citation.title:
        title = citation.title[:150]
        query_parts.append(title)

    if citation.author:
        author = citation.author
        first_author = re.split(r',\s*(?=[A-Z])|(?:\s+and\s+)', author)[0].strip()
        words = first_author.split()
        if words:
            last_name = words[0].rstrip(',').rstrip('.')
            if len(last_name) >= 2:
                query_parts.append(last_name)

    if not query_parts:
        raw = citation.raw_text
        raw_clean = re.sub(r'https?://\S+', '', raw)
        raw_clean = re.sub(r'10\.\d{4,9}/\S+', '', raw_clean)
        raw_clean = re.sub(r'\s+', ' ', raw_clean).strip()
        if raw_clean:
            query_parts.append(raw_clean[:150])

    query = " ".join(query_parts)
    return query[:200]


def _verify_by_crossref_search(citation: Citation) -> VerificationResult:
    """Verify a citation by searching CrossRef with title + author."""
    query = _build_search_query(citation)

    if not query or len(query.strip()) < 8:
        return VerificationResult.from_citation(citation, "not_found", 0, _NOT_FOUND)

    try:
        items = crossref_search(query, rows=5, timeout=15, retries=2)
    except (HTTPRequestError, SystemExit):
        return VerificationResult.from_citation(citation, "not_found", 0, _NOT_FOUND)

    # Normalize CrossRef results to common format
    normalized = []
    for item in items:
        normalized.append({
            "title": _extract_title_from_crossref(item),
            "authors": _extract_authors_from_crossref(item),
            "doi": item.get("DOI"),
        })

    result = _pick_best_match(citation, normalized, "crossref")
    if result:
        return result
    return VerificationResult.from_citation(citation, "not_found", 0, _NOT_FOUND)


def _extract_arxiv_from_raw(raw_text: str) -> Optional[str]:
    """Try to find an arXiv ID in the raw citation text."""
    m = re.search(r'arxiv[.:/\s]+(?:abs/|pdf/)?(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+/\d{7}(?:v\d+)?)', raw_text, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Enrichment (retraction check + OA link)
# ---------------------------------------------------------------------------

def _enrich_result(result: VerificationResult) -> VerificationResult:
    """Add retraction status and OA URL to a verified result."""
    doi = result.match_details.matched_doi or result.doi
    if not doi:
        return result

    # Retraction check
    retracted = check_retraction(doi)
    if retracted is not None:
        result.is_retracted = retracted

    # Unpaywall OA link
    oa = unpaywall_lookup(doi)
    if oa and oa.get("is_oa"):
        result.oa_url = oa.get("best_oa_url") or oa.get("best_oa_pdf") or None

    return result


# ---------------------------------------------------------------------------
# Discipline-routed verification cascade
# ---------------------------------------------------------------------------

def _verify_one(cit: Citation, discipline: str = "unknown") -> VerificationResult:
    """Verify a single citation through the discipline-appropriate cascade."""
    # Skip tiny/garbage citations
    if len(cit.raw_text.strip()) < 15:
        return VerificationResult.from_citation(cit, "not_found", 0, _NOT_FOUND)

    # --- Direct identifier checks first (same for all disciplines) ---
    result = _verify_by_doi(cit)
    if result:
        return _enrich_result(result)

    result = _verify_by_arxiv(cit)
    if result:
        return _enrich_result(result)

    # Try extracting arXiv ID from raw text
    if not cit.arxiv_id:
        raw_arxiv = _extract_arxiv_from_raw(cit.raw_text)
        if raw_arxiv:
            cit_with_arxiv = Citation(
                index=cit.index, raw_text=cit.raw_text,
                author=cit.author, title=cit.title, year=cit.year,
                doi=cit.doi, isbn=cit.isbn, arxiv_id=raw_arxiv,
                journal=cit.journal, volume=cit.volume, pages=cit.pages,
            )
            result = _verify_by_arxiv(cit_with_arxiv)
            if result:
                return _enrich_result(result)

    result = _verify_by_isbn(cit)
    if result:
        return _enrich_result(result)

    # --- Discipline-specific cascades ---
    if discipline == "physics":
        cascade = [_verify_by_inspire, _verify_by_ads, _verify_by_crossref_search, _verify_by_semantic_scholar]
    elif discipline == "biomedical":
        cascade = [_verify_by_pubmed, _verify_by_crossref_search, _verify_by_semantic_scholar]
    elif discipline == "math":
        cascade = [_verify_by_zbmath, _verify_by_hal, _verify_by_crossref_search, _verify_by_semantic_scholar]
    elif discipline == "cs":
        cascade = [_verify_by_dblp, _verify_by_crossref_search, _verify_by_semantic_scholar]
    elif discipline in ("social_science", "humanities"):
        cascade = [_verify_by_hal, _verify_by_crossref_search, _verify_by_semantic_scholar, _verify_by_openalex, _verify_by_openlibrary_search]
    else:
        # Unknown: broad coverage (includes HAL for French papers, OpenLibrary for books)
        cascade = [_verify_by_crossref_search, _verify_by_hal, _verify_by_semantic_scholar, _verify_by_openalex, _verify_by_openlibrary_search]

    # First pass: try each source, collect best result
    best_result: Optional[VerificationResult] = None
    for verify_fn in cascade:
        try:
            result = verify_fn(cit)
            if result is None:
                continue
            # Return immediately on high-confidence verified
            if result.verdict == "verified" and result.confidence >= 80:
                return _enrich_result(result)
            # Track best result so far
            if best_result is None or result.confidence > best_result.confidence:
                best_result = result
        except Exception:
            continue

    if best_result and best_result.confidence > 0:
        return _enrich_result(best_result)

    # All cascades failed
    return VerificationResult.from_citation(cit, "not_found", 0, _NOT_FOUND)


def verify_citations(
    citations: List[Citation],
    ref_section: str = "",
) -> List[VerificationResult]:
    """Verify citations in parallel using a thread pool.

    If ref_section is provided, detects citation style to route verification.
    """
    if not citations:
        return []

    # Detect discipline from citation style
    style = detect_citation_style(ref_section) if ref_section else "unknown"
    discipline = _STYLE_TO_DISCIPLINE.get(style, "unknown")

    results: Dict[int, VerificationResult] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(citations))) as pool:
        futures = {pool.submit(_verify_one, cit, discipline): cit.index for cit in citations}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = VerificationResult.from_citation(
                    citations[idx - 1], "not_found", 0, _NOT_FOUND,
                )

    return [results[i] for i in sorted(results)]


def compute_summary(results: List[VerificationResult]) -> Dict[str, Any]:
    """Compute summary statistics from verification results."""
    total = len(results)
    verified = sum(1 for r in results if r.verdict == "verified")
    suspect = sum(1 for r in results if r.verdict == "suspect")
    not_found = sum(1 for r in results if r.verdict == "not_found")
    retracted = sum(1 for r in results if r.is_retracted)

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
        "retracted": retracted,
        "risk_level": risk_level,
    }
