"""Citation extraction from plain text using regex/heuristic parsing.

Supports:
- Numbered reference lists: [1], 1., 1)
- APA-style: Author, A. B. (Year). Title. Journal...
- DOI patterns: 10.XXXX/...
- ISBN patterns: ISBN-10 and ISBN-13
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Citation:
    index: int
    raw_text: str
    author: Optional[str] = None
    title: Optional[str] = None
    year: Optional[str] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None
    arxiv_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# DOI: 10.XXXX/anything-until-whitespace-or-common-punctuation
DOI_RE = re.compile(r'\b(10\.\d{4,9}/[^\s,;}\]]+)')

# ISBN-13 (with or without hyphens) or ISBN-10
ISBN_RE = re.compile(
    r'\bISBN[-:\s]*'
    r'((?:97[89][-\s]?\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?\d)'  # ISBN-13
    r'|(?:\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?[\dXx]))',         # ISBN-10
    re.IGNORECASE,
)

# Bare ISBN-13 without prefix (978/979 start)
BARE_ISBN13_RE = re.compile(
    r'\b(97[89][-\s]?\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?\d)\b'
)

# arXiv ID: old format (hep-ph/9901234) or new format (1234.56789v2)
ARXIV_RE = re.compile(
    r'arXiv[:\s]*(\d{4}\.\d{4,5}(?:v\d+)?'
    r'|[a-z-]+/\d{7}(?:v\d+)?)',
    re.IGNORECASE,
)

# Year in parentheses: (2023) or (2023, January)
YEAR_PAREN_RE = re.compile(r'\((\d{4})\b[^)]*\)')

# Standalone 4-digit year (fallback)
YEAR_BARE_RE = re.compile(r'\b((?:19|20)\d{2})\b')

# Numbered reference: starts with [1], 1., or 1)
NUMBERED_REF_RE = re.compile(
    r'^\s*(?:\[(\d+)\]|(\d+)[.\)])\s+',
    re.MULTILINE,
)

# APA-ish author block: "LastName, F. I." or "LastName, FirstName"
# Captures up to the year parenthetical
APA_AUTHOR_RE = re.compile(
    r'^([A-Z][a-zA-Z\u00C0-\u024F\'-]+(?:,\s*(?:[A-Z]\.?\s*)+)'
    r'(?:(?:,?\s*(?:&|and)\s*)?[A-Z][a-zA-Z\u00C0-\u024F\'-]+(?:,\s*(?:[A-Z]\.?\s*)+))*'
    r'(?:,?\s*et\s+al\.?)?)'
)

# Title after year parenthetical in APA: (Year). Title.
APA_TITLE_RE = re.compile(
    r'\(\d{4}[^)]*\)\.\s*(.+?)(?:\.\s|$)',
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Reference block detection
# ---------------------------------------------------------------------------

def _find_reference_section(text: str) -> str:
    """Try to locate a 'References' / 'Bibliography' section; fall back to full text."""
    patterns = [
        r'(?:^|\n)\s*(?:References|Bibliography|Works Cited|Literature|Bibliographie|Références)\s*\n',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return text[m.start():]
    return text


def _split_numbered_references(text: str) -> List[str]:
    """Split text into individual references using numbered markers."""
    parts = NUMBERED_REF_RE.split(text)
    refs: List[str] = []
    i = 1
    while i < len(parts):
        # parts[i] or parts[i+1] is the number, parts[i+2] is the text
        # NUMBERED_REF_RE has 2 groups, so stride is 3
        ref_text = parts[i + 2].strip() if i + 2 < len(parts) else ""
        if ref_text:
            # Take only until next blank line or next numbered ref
            ref_text = re.split(r'\n\s*\n', ref_text)[0].strip()
            refs.append(ref_text)
        i += 3
    return refs


def _split_apa_references(text: str) -> List[str]:
    """Split text into references by detecting APA author-year patterns on new lines."""
    lines = text.split('\n')
    refs: List[str] = []
    current: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                refs.append(' '.join(current))
                current = []
            continue
        # New reference starts with an author-like pattern
        if APA_AUTHOR_RE.match(stripped) and current:
            refs.append(' '.join(current))
            current = [stripped]
        else:
            current.append(stripped)

    if current:
        refs.append(' '.join(current))

    return [r for r in refs if len(r) > 20]


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def _extract_doi(text: str) -> Optional[str]:
    m = DOI_RE.search(text)
    if m:
        doi = m.group(1).rstrip('.')
        return doi
    return None


def _extract_isbn(text: str) -> Optional[str]:
    m = ISBN_RE.search(text)
    if m:
        return re.sub(r'[-\s]', '', m.group(1))
    m = BARE_ISBN13_RE.search(text)
    if m:
        return re.sub(r'[-\s]', '', m.group(1))
    return None


def _extract_arxiv_id(text: str) -> Optional[str]:
    m = ARXIV_RE.search(text)
    if m:
        return m.group(1)
    return None


def _extract_year(text: str) -> Optional[str]:
    m = YEAR_PAREN_RE.search(text)
    if m:
        return m.group(1)
    m = YEAR_BARE_RE.search(text)
    if m:
        return m.group(1)
    return None


def _normalize(text: str) -> str:
    """Collapse newlines, fix hyphenated line breaks, normalize whitespace."""
    # Fix hyphenated line breaks: "Convolu-\ntional" or "Convolu- tional" → "Convolutional"
    text = re.sub(r'-\s+(?=[a-z])', '', text)
    return re.sub(r'\s+', ' ', text).strip()


# Matches a sentence-ending period: word of 2+ chars followed by ". " then uppercase.
# Skips single-letter initials like "V." or "E."
_SENT_BOUNDARY_RE = re.compile(r'(?<=\w{2})\.\s+(?=[A-Z])')


def _split_author_rest(text: str) -> tuple[Optional[str], str]:
    """Split CS/ML citation into (author_block, rest) at the first sentence boundary."""
    parts = _SENT_BOUNDARY_RE.split(text, maxsplit=1)
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip()
    return None, text


def _extract_author(text: str) -> Optional[str]:
    norm = _normalize(text)
    m = APA_AUTHOR_RE.match(norm)
    if m:
        return m.group(1).strip().rstrip(',')
    # Fallback: take text before first parenthetical year — APA style
    m = re.match(r'^(.+?)\s*\(\d{4}', norm)
    if m:
        author = m.group(1).strip().rstrip(',').rstrip('.')
        if len(author) < 200:
            return author
    # Fallback: CS/ML style — "First Last, First Last, and First Last. Title."
    candidate, rest = _split_author_rest(norm)
    if candidate and len(candidate) < 200:
        # Multi-author: must have commas or "and"
        if ',' in candidate or ' and ' in candidate:
            return candidate
        # Single author: 2-4 capitalized words (e.g. "Francois Chollet")
        words = candidate.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words):
            return candidate
    return None


def _extract_title(text: str) -> Optional[str]:
    norm = _normalize(text)
    m = APA_TITLE_RE.search(norm)
    if m:
        title = m.group(1).strip()
        if len(title) > 10:
            return title
    # Remove DOI and ISBN portions for cleaner matching
    clean = DOI_RE.sub('', norm)
    clean = ISBN_RE.sub('', clean)
    # Try to find a quoted or italicized title
    m = re.search(r'["\u201c](.+?)["\u201d]', clean)
    if m and len(m.group(1)) > 10:
        return m.group(1)
    # APA: text between (Year). and next period
    m = re.search(r'\(\d{4}[^)]*\)\.\s*(.+?)\.', clean)
    if m and len(m.group(1)) > 10:
        return m.group(1).strip()
    # CS/ML style: "Authors. Title. Venue, Year."
    # After splitting off author, take text up to the next sentence boundary
    _, rest = _split_author_rest(clean)
    if rest:
        # Take up to the first period (that's not after an initial)
        m = re.match(r'(.+?)(?:\.\s|\.?$)', rest)
        if m:
            title = m.group(1).strip()
            if len(title) > 10:
                return title
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_citations(text: str) -> List[Citation]:
    """Extract citations from document text. Returns list of Citation objects."""
    ref_section = _find_reference_section(text)

    # Try numbered references first
    raw_refs = _split_numbered_references(ref_section)

    # Fall back to APA-style splitting
    if len(raw_refs) < 2:
        raw_refs = _split_apa_references(ref_section)

    # If still nothing, try the full text
    if len(raw_refs) < 2:
        raw_refs = _split_apa_references(text)

    citations: List[Citation] = []
    for i, raw in enumerate(raw_refs, start=1):
        cit = Citation(
            index=i,
            raw_text=raw,
            doi=_extract_doi(raw),
            isbn=_extract_isbn(raw),
            year=_extract_year(raw),
            author=_extract_author(raw),
            title=_extract_title(raw),
            arxiv_id=_extract_arxiv_id(raw),
        )
        citations.append(cit)

    return citations
