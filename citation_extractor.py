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

# Bracket-code reference: [BR06], [A], [HBP17], [HMSC95], [C-Chu 1], [H 1], etc.
# These codes sit on their own line or are followed by the author on the same line
BRACKET_CODE_RE = re.compile(
    r'^\s*\[([A-Za-z][A-Za-z0-9+\- ]{0,12}(?:\d{0,4})?)\]\s*\n?',
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
        # Exact section headers (case-insensitive)
        r'(?:^|\n)\s*(?:References|Bibliography|Works Cited|Literature|Bibliographie|Références)\s*\n',
        # All-caps variants
        r'(?:^|\n)\s*(?:REFERENCES|BIBLIOGRAPHY|WORKS CITED)\s*\n',
    ]
    best_match = None
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            # Prefer the latest match (closer to end of document = more likely the real refs)
            if best_match is None or m.start() > best_match.start():
                best_match = m
    if best_match:
        return text[best_match.start():]
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


def _split_bracket_code_references(text: str) -> List[str]:
    """Split text into individual references using bracket-code markers like [BR06], [A]."""
    parts = BRACKET_CODE_RE.split(text)
    refs: List[str] = []
    # parts[0] = text before first code, then alternating: code, text, code, text...
    i = 1
    while i < len(parts):
        # parts[i] = code, parts[i+1] = text after code
        ref_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if ref_text:
            # Take until next blank line (paragraph boundary)
            ref_text = re.split(r'\n\s*\n', ref_text)[0].strip()
            # Collapse internal newlines
            ref_text = re.sub(r'\s+', ' ', ref_text)
            if len(ref_text) > 15:
                refs.append(ref_text)
        i += 2
    return refs


# ACL/NLP style: "FirstName LastName, FirstName LastName, and FirstName LastName."
# Matches lines starting with a name pattern (handles initials like "R." and "D.")
_ACL_AUTHOR_START_RE = re.compile(
    r'^(?:[A-Z][a-zA-Z\u00C0-\u024F\'-]*\.?\s+){1,4}'  # First/Middle names or initials
    r'[A-Z][a-zA-Z\u00C0-\u024F\'-]+'                   # Last name (2+ chars)
    r'(?:\s*,|\s*\.)',                                    # Followed by comma or period
)


def _split_acl_references(text: str) -> List[str]:
    """Split text into references using ACL/NLP author-year format.

    Pattern: "First Last, First Last, and First Last. Year. Title. In Venue."
    No blank lines between references.
    """
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

        # Check if this line starts a new reference
        is_new_ref = False
        if _ACL_AUTHOR_START_RE.match(stripped):
            # Only consider it a new reference if:
            # 1. Not a continuation word
            # 2. The current buffer already contains a year (previous ref is complete)
            if not re.match(r'^(?:In|The|And|Or|A|An|On|For|With|From|Using)\s', stripped):
                current_text = ' '.join(current)
                # Check if current buffer has a year pattern (reference is complete)
                has_year = bool(re.search(r'\b(?:19|20)\d{2}[a-z]?\b', current_text))
                if has_year or not current:
                    is_new_ref = True

        if is_new_ref and current:
            refs.append(' '.join(current))
            current = [stripped]
        else:
            current.append(stripped)

    if current:
        refs.append(' '.join(current))

    return [r for r in refs if len(r) > 20]


def _split_blank_line_references(text: str) -> List[str]:
    """Split references by blank lines — each paragraph is one reference."""
    paragraphs = re.split(r'\n\s*\n', text)
    refs: List[str] = []
    for para in paragraphs:
        stripped = para.strip()
        # Skip very short paragraphs and section headers
        if len(stripped) < 25:
            continue
        # Skip lines that look like section headers
        if re.match(r'^(?:References|Bibliography|REFERENCES)\s*$', stripped, re.IGNORECASE):
            continue
        # Collapse whitespace
        normalized = re.sub(r'\s+', ' ', stripped)
        refs.append(normalized)
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


_LIGATURE_MAP = str.maketrans({
    '\ufb00': 'ff', '\ufb01': 'fi', '\ufb02': 'fl',
    '\ufb03': 'ffi', '\ufb04': 'ffl',
})


def _normalize(text: str) -> str:
    """Collapse newlines, fix hyphenated line breaks, normalize ligatures and whitespace."""
    # Replace Unicode ligatures (ﬁ→fi, ﬂ→fl, ﬀ→ff, etc.)
    text = text.translate(_LIGATURE_MAP)
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


def _is_journal_like(text: str) -> bool:
    """Check if a string looks like a journal name/venue rather than an author."""
    journal_markers = [
        'phys. rev', 'astrophys.', 'nature', 'science', 'j. ', 'ann.', 'rev.',
        'proc.', 'lett.', 'sov.', 'appl.', 'classical', 'quantum', 'gravit',
        'nips', 'icml', 'iclr', 'cvpr', 'iccv', 'eccv', 'emnlp', 'acl',
        'in proceedings', 'in advances', 'conference', 'journal', 'workshop',
        'transactions', 'arxiv', 'preprint',
    ]
    lower = text.lower()
    return any(marker in lower for marker in journal_markers)


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
            # For physics compact format, the "author" often includes journal info
            # e.g. "A. Einstein, Sitzungsber. K. Preuss. Akad. Wiss. 1, 688"
            # Try to extract just the author part (before journal info)
            parts = author.split(',')
            if len(parts) >= 2:
                # Check if later parts look like journal/venue info
                author_parts = []
                for j, part in enumerate(parts):
                    if _is_journal_like(part):
                        break
                    author_parts.append(part)
                if author_parts:
                    cleaned = ','.join(author_parts).strip().rstrip(',')
                    if cleaned:
                        return cleaned
            return author
    # ACL/NLP style: "First Last, First Last, and First Last. Year. Title."
    m = re.match(r'^(.+?)\.\s+(?:19|20)\d{2}[a-z]?\.', norm)
    if m:
        author = m.group(1).strip().rstrip(',')
        if len(author) < 200 and not _is_journal_like(author):
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
    # Bracket-code style: "Author, Title, Journal Vol (Year), Pages."
    # After author (text before first comma-separated title), extract the title part
    m = re.match(r'^[A-Z][^,]+(?:,\s*[A-Z][^,]*)*,\s+(.+?)(?:,\s+(?:[A-Z][\w.\s]+\d|\d{4})|$)', clean)
    if m and len(m.group(1)) > 10:
        candidate = m.group(1).strip().rstrip(',').rstrip('.')
        # Don't return page numbers or bare numbers as titles
        if not re.match(r'^[\d\s,\-LlSs()]+$', candidate):
            return candidate
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
    # ACL/NLP style: "Author1, Author2, and Author3. Year. Title. In Venue."
    # Look for "Year. Title." pattern
    m = re.search(r'(?:19|20)\d{2}[a-z]?\.\s+(.+?)(?:\.\s|\.?$)', clean)
    if m and len(m.group(1)) > 10:
        return m.group(1).strip()
    return None


def _clean_title(title: str) -> str:
    """Clean up extracted title — remove author/venue fragments."""
    # Strip "and AuthorName. Year." prefix that sometimes leaks in
    # e.g. "and Geoffrey E Hinton. Layer normalization. arXiv..." → "Layer normalization"
    cleaned = re.sub(
        r'^(?:and\s+)?(?:[A-Z][a-zA-Z\u00C0-\u024F\'-]*\.?\s+){1,5}'
        r'(?:19|20)\d{2}[a-z]?\.\s+',
        '', title,
    )
    if cleaned and len(cleaned) > 10:
        title = cleaned

    # Strip venue/proceedings suffix
    # e.g. "Title. In Proceedings of..." → "Title"
    m = re.match(r'(.{10,}?)\.\s+(?:In\s+|arXiv\s|Proceedings|CoRR)', title)
    if m:
        title = m.group(1)

    # Strip "and AuthorName." prefix without year
    m = re.match(r'^and\s+[A-Z][a-zA-Z\s.]+\.\s+(.+)', title)
    if m and len(m.group(1)) > 10:
        title = m.group(1)

    return title.strip().rstrip('.')


def _validate_title(title: Optional[str]) -> Optional[str]:
    """Reject garbage titles (page numbers, bare numbers, very short)."""
    if not title:
        return None
    # Clean up first
    title = _clean_title(title)
    # Reject titles that are just page numbers, volume info, or too short
    stripped = title.strip()
    if len(stripped) < 8:
        return None
    # Reject if it's mostly numbers/punctuation (e.g. "688 (1916)", "L105 (1971)")
    alpha_chars = sum(1 for c in stripped if c.isalpha())
    if alpha_chars < 5:
        return None
    # Reject if it looks like "vol, page (year)" pattern
    if re.match(r'^[\d\s,\-LlA-Z().:]+$', stripped) and alpha_chars < 10:
        return None
    # Reject arXiv IDs as titles (e.g. "arXiv:1302.4389")
    if re.match(r'^arXiv[:\s]*\d{4}\.\d{4,5}', stripped, re.IGNORECASE):
        return None
    # Reject journal names as titles
    journal_names = [
        'neural computation', 'machine learning', 'nature', 'science',
        'proceedings of', 'journal of', 'transactions on',
    ]
    if stripped.lower().strip('.') in journal_names:
        return None
    return stripped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_citations(text: str) -> List[Citation]:
    """Extract citations from document text. Returns list of Citation objects."""
    ref_section = _find_reference_section(text)

    def _is_good_split(refs: List[str]) -> bool:
        """Check if a split produced reasonable citations (not too long, not too few)."""
        if len(refs) < 3:
            return False
        avg_len = sum(len(r) for r in refs) / len(refs)
        # If average citation is > 400 chars, the split likely failed
        return avg_len < 400

    # Try all splitting strategies, pick the best one
    candidates: List[List[str]] = []

    # Strategy 1: Numbered references (e.g. [1], 1., 1))
    numbered = _split_numbered_references(ref_section)
    if _is_good_split(numbered):
        candidates.append(numbered)

    # Strategy 2: Bracket-code references (e.g. [BR06], [A], [HBP17])
    bracket = _split_bracket_code_references(ref_section)
    if _is_good_split(bracket):
        candidates.append(bracket)

    # Strategy 3: APA-style (LastName, F. I. (Year). Title.)
    apa = _split_apa_references(ref_section)
    if _is_good_split(apa):
        candidates.append(apa)

    # Strategy 4: ACL/NLP style (FirstName LastName. Year. Title.)
    acl = _split_acl_references(ref_section)
    if _is_good_split(acl):
        candidates.append(acl)

    # Strategy 5: Blank-line splitting (each paragraph = one reference)
    blank = _split_blank_line_references(ref_section)
    if _is_good_split(blank):
        candidates.append(blank)

    # Pick the strategy that found the most citations
    if candidates:
        raw_refs = max(candidates, key=len)
    else:
        # Fallback: try any strategy that got > 0 results
        all_attempts = [numbered, bracket, apa, acl, blank]
        non_empty = [r for r in all_attempts if r]
        if non_empty:
            raw_refs = max(non_empty, key=len)
        else:
            raw_refs = []

    # Last resort: try APA on full text
    if len(raw_refs) < 3:
        apa_full = _split_apa_references(text)
        if len(apa_full) > len(raw_refs):
            raw_refs = apa_full

    # Filter out tiny/empty citations
    raw_refs = [r for r in raw_refs if len(r.strip()) >= 15]

    citations: List[Citation] = []
    for i, raw in enumerate(raw_refs, start=1):
        cit = Citation(
            index=i,
            raw_text=raw,
            doi=_extract_doi(raw),
            isbn=_extract_isbn(raw),
            year=_extract_year(raw),
            author=_extract_author(raw),
            title=_validate_title(_extract_title(raw)),
            arxiv_id=_extract_arxiv_id(raw),
        )
        citations.append(cit)

    return citations
