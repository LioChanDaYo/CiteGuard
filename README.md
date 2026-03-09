# CiteGuard

Citation verification for academic and LLM-generated documents. Upload a PDF, DOCX, or TXT file and get a report showing which citations are real, suspect, or not found.

## No AI, no LLMs — just deterministic verification

CiteGuard does **not** use large language models, AI classifiers, or any form of machine learning to verify citations. There is no guessing, no probabilistic generation, no hallucination risk in the verification itself.

Instead, CiteGuard works by **cross-referencing extracted citation metadata against authoritative bibliographic databases**:

- [CrossRef](https://www.crossref.org/) — the canonical DOI registry, covering 150M+ scholarly works
- [arXiv](https://arxiv.org/) — preprint server for physics, math, CS, and more
- [OpenLibrary](https://openlibrary.org/) — open catalog for books (ISBN lookup)

If a citation has a DOI, CiteGuard resolves it directly against CrossRef. If it has an arXiv ID, it queries the arXiv API. If it has an ISBN, it checks OpenLibrary. When none of those identifiers are present, it falls back to a free-text search on CrossRef using the extracted title and author, then compares the API response against the citation using string similarity.

Every verdict is traceable to an API response. Nothing is inferred or generated.

## How it works

1. **Extract** — Parses the reference section using regex and heuristics. Supports APA style, CS/ML style, ACL/NLP style, numbered references (`[1]`, `1.`, `1)`), and bracket-code references (`[BR06]`, `[HBP17]`).
2. **Identify** — Extracts structured metadata from each citation: DOI, ISBN, arXiv ID, author names, title, and year.
3. **Verify** — Checks each citation against CrossRef, arXiv, and OpenLibrary APIs in a cascade (DOI → arXiv → ISBN → free-text search). Runs in parallel (up to 8 workers).
4. **Report** — Assigns a verdict with a confidence score:
   - **Verified** (confidence ≥ 70) — citation matches a real publication
   - **Suspect** (confidence < 70) — a match was found but with low confidence (possible wrong match, partial title match, etc.)
   - **Not Found** — no match found in any database

## Tested against real papers

CiteGuard has been tested against well-known papers across multiple disciplines:

| Paper | Discipline | Citations | Verified |
|-------|-----------|-----------|----------|
| Attention Is All You Need | CS / ML | 40 | 78% |
| BERT | NLP | 52 | 65% |
| ResNet | Computer Vision | 52 | 65% |
| GANs | CS / ML | 31 | 58% |
| Deep Hedging | Quantitative Finance | 32 | 53% |
| AlphaFold | Bioinformatics | 69 | 48% |
| Ricci Flow (Perelman) | Mathematics | 27 | 7% |
| LIGO Gravitational Waves | Physics | 118 | 7% |

Physics and math papers score low because they use compact citation formats (author, journal, volume, page, year) with no explicit title — making free-text search unreliable. CS/NLP/finance papers with standard citation formats verify well.

## Quick start

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000

### Docker

```bash
docker compose up --build
```

## CLI

The CrossRef/OpenLibrary client can also be used standalone:

```bash
python crossref_api_client.py search -q "machine learning" -n 5
python crossref_api_client.py doi --doi 10.1038/s41586-024-07031-2
python crossref_api_client.py isbn --isbn 9782070368228
```

## Supported citation formats

- **Numbered references**: `[1] Author. Title. Venue, Year.`
- **Bracket-code references**: `[BR06] Author, Title, Journal (Year).`
- **APA style**: `Author, A. B. (Year). Title. Journal.`
- **CS/ML style**: `Author, Author, and Author. Title. Venue, Year.`
- **ACL/NLP style**: `Author1, Author2, and Author3. Year. Title. In Venue.`
- **Physics compact**: `Author, Journal Volume, Page (Year).` (limited verification — no title)

## Known limitations

- **Regex-based parsing** — works well with standard academic formats but may miss unusual citation styles (footnote-style, legal citations, non-Latin scripts)
- **CrossRef coverage** — some legitimate citations are not in CrossRef (conference workshops, theses, technical reports, very old publications). These will show as "not found"
- **Free-text search noise** — when no DOI/ISBN/arXiv ID is available, title+author search sometimes matches the wrong paper. The confidence score reflects match quality
- **PDF text extraction** — multi-column layouts may produce merged text. PyMuPDF handles most cases but scanned PDFs without OCR are not supported
- **In-memory job store** — jobs are stored in RAM with a 1-hour TTL, not suitable for production without persistent storage

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CROSSREF_MAIL` | Email for CrossRef polite pool (faster rate limits) | `anonymous@example.com` |

## License

GPL-3.0
