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
    journal: Optional[str] = None
    volume: Optional[str] = None
    pages: Optional[str] = None


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


def _split_french_references(text: str) -> List[str]:
    """Split French-style references: Author, Year, « Title », Journal.

    French academic citations typically start with a capitalized surname
    followed by an initial and a 4-digit year separated by commas, without
    parentheses around the year.
    Pattern: LastName, I., Year, ...
    """
    # Match lines starting with a capitalized word/phrase followed by a year:
    # "Amossé, T., 2024, ..." or "Insee, 2025, ..." or "Di Paola V., 2023, ..."
    # or "Observatoire des inégalités, 2025, ..."
    _FR_REF_START = re.compile(
        r'^[A-Z\u00C0-\u00DC][a-zA-Z\u00C0-\u024F\'\-.\xa0 ]*'  # Name (spaces, dots, nbsp for "Di Paola V.")
        r'(?:,\s*[A-Z\u00C0-\u024F][\w.\-\xa0 ]*)*'              # Optional initials/co-authors
        r',\s*(?:19|20)\d{2}\b'                                    # Comma then year
    )
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

        if _FR_REF_START.match(stripped) and current:
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


# ---------------------------------------------------------------------------
# Physics journal abbreviations for detection
# ---------------------------------------------------------------------------

_PHYSICS_JOURNALS = {
    'phys. rev. lett.': 'Phys.Rev.Lett.',
    'phys. rev. d': 'Phys.Rev.D',
    'phys. rev.': 'Phys.Rev.',
    'j. high energy phys.': 'JHEP',
    'jhep': 'JHEP',
    'nucl. phys.': 'Nucl.Phys.',
    'eur. phys. j.': 'Eur.Phys.J.',
    'class. quantum grav.': 'Class.Quant.Grav.',
    'j. cosmol. astropart. phys.': 'JCAP',
    'jcap': 'JCAP',
    'astrophys. j.': 'Astrophys.J.',
    'mon. not. r. astron. soc.': 'Mon.Not.Roy.Astron.Soc.',
    'ann. phys.': 'Annals Phys.',
    'rev. mod. phys.': 'Rev.Mod.Phys.',
    'living rev. relativ.': 'Living Rev.Rel.',
    'gen. relativ. gravit.': 'Gen.Rel.Grav.',
    'new j. phys.': 'New J.Phys.',
    'prog. theor. phys.': 'Prog.Theor.Phys.',
    'sov. phys. jetp': 'Sov.Phys.JETP',
}

# Sort by length descending so longer matches take priority (e.g. "phys. rev. lett." before "phys. rev.")
_PHYSICS_JOURNALS_SORTED = sorted(_PHYSICS_JOURNALS.keys(), key=len, reverse=True)

# Medical journal volume;issue:pages pattern: "2020;382(8):727-733"
_MEDICAL_VOL_RE = re.compile(r'(\d{4})\s*;\s*(\d+)\s*(?:\((\d+)\))?\s*:\s*([\d\-]+)')

# Physics compact: "Journal Name Vol, Pages (Year)" or "Journal Name Vol (Year) Pages"
_PHYSICS_VOL_PAGES_RE = re.compile(
    r'(\d+)\s*[,]\s*(\d[\d\-]+)'
)


def _extract_journal(text: str) -> Optional[str]:
    """Extract journal name from citation text.

    Checks against known physics journal abbreviations first, then falls back
    to medical journal patterns.
    """
    lower = text.lower()
    for abbrev in _PHYSICS_JOURNALS_SORTED:
        if abbrev in lower:
            return _PHYSICS_JOURNALS[abbrev]

    # Medical journal pattern: "JournalName. Year;Vol(Issue):Pages"
    # Try to extract journal name before the year;vol pattern
    m = _MEDICAL_VOL_RE.search(text)
    if m:
        # Find the journal name: text before the year;vol pattern
        idx = text.find(m.group(0))
        if idx > 0:
            prefix = text[:idx].strip().rstrip('.')
            # Walk backwards past the year to find journal name
            # Look for "JournalName. Year" pattern
            jm = re.search(r'([A-Z][A-Za-z\s]+(?:J|Med|Lancet|BMJ|JAMA)[A-Za-z\s]*)\.\s*$', prefix)
            if jm:
                return jm.group(1).strip()

    return None


def _extract_volume(text: str) -> Optional[str]:
    """Extract volume number from citation text.

    Handles physics compact format (bare number after journal) and
    medical format (number after semicolon).
    """
    # Medical: "Year;Vol(Issue):Pages"
    m = _MEDICAL_VOL_RE.search(text)
    if m:
        return m.group(2)

    # IEEE/standard: "vol. 116" or "Vol. 116"
    m = re.search(r'\bvol\.?\s*(\d+)', text, re.IGNORECASE)
    if m:
        return m.group(1)

    # Physics compact: find journal, then the number after it is the volume
    lower = text.lower()
    for abbrev in _PHYSICS_JOURNALS_SORTED:
        idx = lower.find(abbrev)
        if idx >= 0:
            after = text[idx + len(abbrev):].strip().lstrip('.,')
            vm = re.match(r'\s*(\d+)', after)
            if vm:
                return vm.group(1)

    return None


def _extract_pages(text: str) -> Optional[str]:
    """Extract page numbers from citation text.

    Handles physics compact (bare number after volume comma),
    medical format (after colon), and standard pp. format.
    """
    # Medical: "Year;Vol(Issue):Pages"
    m = _MEDICAL_VOL_RE.search(text)
    if m:
        return m.group(4)

    # Standard: "pp. 123-456" or "pages 123--456"
    m = re.search(r'\bpp\.?\s*([\d]+[\-\u2013]+[\d]+|\d+)', text, re.IGNORECASE)
    if m:
        return m.group(1).replace('\u2013', '-')
    m = re.search(r'\bpages?\s*([\d]+[\-\u2013]+[\d]+|\d+)', text, re.IGNORECASE)
    if m:
        return m.group(1).replace('\u2013', '-')

    # Physics compact: journal Vol, Pages (Year)
    lower = text.lower()
    for abbrev in _PHYSICS_JOURNALS_SORTED:
        idx = lower.find(abbrev)
        if idx >= 0:
            after = text[idx + len(abbrev):].strip().lstrip('.,')
            # Match: Vol, Pages or Vol, Pages (Year)
            pm = re.match(r'\s*\d+\s*[,]\s*([\d\-]+)', after)
            if pm:
                return pm.group(1)

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
    # Try to find a quoted or italicized title (including French guillemets «»)
    m = re.search(r'[\u00ab\u2039]\s*(.+?)\s*[\u00bb\u203a]', clean)
    if m and len(m.group(1)) > 10:
        return m.group(1)
    m = re.search(r'["\u201c](.+?)["\u201d]', clean)
    if m and len(m.group(1)) > 10:
        return m.group(1)
    # French style without guillemets (books): Author, Year, Title, Publisher.
    # Extract text between "Year, " and the next comma or period that looks like a publisher
    m = re.search(r'(?:19|20)\d{2},\s+(.+?)(?:,\s+(?:\u00c9ditions|Editions|Ed\.|Presses|PUF|La\s+D\u00e9couverte|Minuit|Gallimard|Seuil|De\s+Gruyter|Springer|Cambridge|Oxford|Routledge)\b|\.?\s*$)', clean)
    if m and len(m.group(1)) > 5:
        candidate = m.group(1).strip().rstrip(',').rstrip('.')
        if len(candidate) > 5 and not re.match(r'^[\d\s,\-]+$', candidate):
            return candidate
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
    # Bracket-code style: "Author, Title, Journal Vol (Year), Pages."
    # After author (text before first comma-separated title), extract the title part
    m = re.match(r'^[A-Z][^,]+(?:,\s*[A-Z][^,]*)*,\s+(.+?)(?:,\s+(?:[A-Z][\w.\s]+\d|\d{4})|$)', clean)
    if m and len(m.group(1)) > 10:
        candidate = m.group(1).strip().rstrip(',').rstrip('.')
        # Don't return page numbers or bare numbers as titles
        if not re.match(r'^[\d\s,\-LlSs()]+$', candidate):
            return candidate
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
# Citation style detection
# ---------------------------------------------------------------------------

def detect_citation_style(ref_section: str) -> str:
    """Detect citation style from reference section.

    Analyzes ALL references in the section (not individual ones) and
    takes the majority signal.

    Returns one of: 'ieee', 'apa', 'vancouver', 'physics', 'acl', 'chicago', 'unknown'
    """
    # Split into individual lines/references for scoring
    lines = [line.strip() for line in ref_section.split('\n') if line.strip() and len(line.strip()) > 20]

    if not lines:
        return 'unknown'

    scores = {
        'ieee': 0,
        'apa': 0,
        'vancouver': 0,
        'physics': 0,
        'acl': 0,
        'chicago': 0,
    }

    for line in lines:
        lower = line.lower()

        # IEEE: [N] numbering + vol./no./pp. abbreviations
        if re.match(r'^\s*\[\d+\]', line):
            scores['ieee'] += 1
        if re.search(r'\bvol\.\s*\d+', lower) or re.search(r'\bno\.\s*\d+', lower):
            scores['ieee'] += 1
        if re.search(r'\bpp\.\s*\d+', lower):
            scores['ieee'] += 0.5

        # APA: (Year). pattern after author block
        if re.search(r'\(\d{4}[a-z]?\)\.\s', line):
            scores['apa'] += 2

        # Vancouver: Author initials without periods (Smith AB) +
        # semicolons after year + colon in volume:pages
        if re.match(r'^[A-Z][a-z]+\s+[A-Z]{1,3}[,;]', line):
            scores['vancouver'] += 1.5
        if re.search(r'\d{4}\s*;\s*\d+', line):
            scores['vancouver'] += 1
        if re.search(r'\d+\s*:\s*\d+[\-\u2013]\d+', line):
            scores['vancouver'] += 0.5

        # Physics compact: known physics journal abbreviations + bare volume numbers
        for abbrev in _PHYSICS_JOURNALS_SORTED:
            if abbrev in lower:
                scores['physics'] += 2
                break
        # Also check for physics-style "Vol, Pages (Year)" pattern
        if re.search(r'[A-Za-z.]\s+\d+\s*,\s*\d+\s*\(\d{4}\)', line):
            scores['physics'] += 1

        # ACL: "In Proceedings of" + "pages X--Y" (double-dash)
        if 'in proceedings of' in lower:
            scores['acl'] += 1.5
        if re.search(r'pages?\s+\d+\s*--\s*\d+', lower):
            scores['acl'] += 1.5

        # Chicago: Full first names + publisher city pattern
        # Check for full first names (not just initials)
        if re.match(r'^[A-Z][a-z]+,\s+[A-Z][a-z]{2,}', line):
            scores['chicago'] += 1
        # Publisher city: "City: Publisher" pattern
        if re.search(r'[A-Z][a-z]+:\s+[A-Z][a-z]+\s+(?:University|Press|Publishing|Books)', line):
            scores['chicago'] += 1.5
        # No abbreviated field labels (vol., no., pp.)
        if not re.search(r'\b(?:vol\.|no\.|pp\.)', lower):
            scores['chicago'] += 0.2

    # Return the style with the highest score
    if not any(scores.values()):
        return 'unknown'

    best_style = max(scores, key=lambda k: scores[k])
    if scores[best_style] < 1:
        return 'unknown'

    return best_style


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

    # Strategy 5: French style (Author, Year, « Title », Journal)
    french = _split_french_references(ref_section)
    if _is_good_split(french):
        candidates.append(french)

    # Strategy 6: Blank-line splitting (each paragraph = one reference)
    blank = _split_blank_line_references(ref_section)
    if _is_good_split(blank):
        candidates.append(blank)

    # Pick the strategy that found the most citations
    if candidates:
        raw_refs = max(candidates, key=len)
    else:
        # Fallback: try any strategy that got > 0 results
        all_attempts = [numbered, bracket, apa, acl, french, blank]
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

    # Try GROBID for structured extraction (enhances compact citation formats)
    grobid_results: dict = {}
    try:
        from grobid_client import parse_citations_batch
        grobid_results = parse_citations_batch(raw_refs, start_index=1)
    except Exception:
        pass  # GROBID unavailable — fall back to regex only

    citations: List[Citation] = []
    for i, raw in enumerate(raw_refs, start=1):
        # Regex extraction (always runs)
        regex_cit = Citation(
            index=i,
            raw_text=raw,
            doi=_extract_doi(raw),
            isbn=_extract_isbn(raw),
            year=_extract_year(raw),
            author=_extract_author(raw),
            title=_validate_title(_extract_title(raw)),
            arxiv_id=_extract_arxiv_id(raw),
            journal=_extract_journal(raw),
            volume=_extract_volume(raw),
            pages=_extract_pages(raw),
        )

        # Merge GROBID fields (GROBID wins when it has data, regex fills gaps)
        grobid_cit = grobid_results.get(i)
        if grobid_cit is not None:
            cit = Citation(
                index=i,
                raw_text=raw,
                author=grobid_cit.author or regex_cit.author,
                title=grobid_cit.title or regex_cit.title,
                year=grobid_cit.year or regex_cit.year,
                doi=regex_cit.doi or grobid_cit.doi,  # regex DOI more reliable (direct pattern match)
                isbn=regex_cit.isbn,  # GROBID doesn't extract ISBN
                arxiv_id=regex_cit.arxiv_id,  # regex arXiv more reliable (direct pattern match)
                journal=grobid_cit.journal or regex_cit.journal,
                volume=grobid_cit.volume or regex_cit.volume,
                pages=grobid_cit.pages or regex_cit.pages,
            )
        else:
            cit = regex_cit

        citations.append(cit)

    return citations
