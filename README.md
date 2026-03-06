# CiteGuard

Citation verification for LLM-generated documents. Upload a PDF, DOCX, or TXT file and get a report showing which citations are real, suspect, or hallucinated.

## How it works

1. **Extract** — Parses the reference section using regex/heuristics (APA style, CS/ML style, numbered references)
2. **Identify** — Extracts DOIs, ISBNs, arXiv IDs, author names, titles, and years from each citation
3. **Verify** — Checks each citation against CrossRef, arXiv, and OpenLibrary APIs (in parallel)
4. **Report** — Assigns a verdict (verified / suspect / not found) with a confidence score

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

The original CrossRef/OpenLibrary client is still available:

```bash
python crossref_api_client.py search -q "machine learning" -n 5
python crossref_api_client.py doi --doi 10.1038/s41586-024-07031-2
python crossref_api_client.py isbn --isbn 9782070368228
```

## Known limitations

- **Citation parsing is regex-based** — works well with APA and CS/ML formats, but may miss or misparse citations in unusual formats (e.g., footnote-style, legal citations, non-Latin scripts)
- **Accuracy is not 100%** — some legitimate citations may show as "not found" if they are not indexed in CrossRef or arXiv (conference workshops, theses, technical reports, very recent papers)
- **CrossRef free-text search can return false positives** — when no DOI, ISBN, or arXiv ID is available, verification falls back to title+author search, which sometimes matches the wrong paper. The confidence score reflects match quality
- **PDF text extraction varies** — multi-column layouts, scanned PDFs, and unusual encodings may produce garbled text that breaks citation parsing
- **Hyphenated line breaks** — most are handled, but edge cases in complex PDF layouts may persist
- **In-memory job store** — jobs are stored in RAM with a 1-hour TTL. Not suitable for production deployment without adding persistent storage
- **Single-author short names** — very short author names (e.g., "Li, X.") may not be recognized by the heuristic parser

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CROSSREF_MAIL` | Email for CrossRef polite pool | `anonymous@example.com` |

## License

GPL-3.0
