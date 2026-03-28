---
title: CiteGuard
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# CiteGuard

Citation verification for academic and LLM-generated documents. Upload a PDF, DOCX, or TXT file and get a report showing which citations are real, suspect, or not found.

## How it works

1. **Extract** — Parses the reference section using [GROBID](https://github.com/kermitt2/grobid) (ML-based) with regex fallback. Handles all citation formats including compact physics/math styles.
2. **Identify** — Extracts structured metadata: DOI, ISBN, arXiv ID, author names, title, journal, volume, pages, year.
3. **Verify** — Checks each citation against 12 scholarly databases in discipline-aware cascades (up to 8 workers in parallel):
   - [CrossRef](https://www.crossref.org/) — 150M+ scholarly works
   - [Semantic Scholar](https://www.semanticscholar.org/) — 225M+ papers
   - [OpenAlex](https://openalex.org/) — 250M+ works
   - [PubMed](https://pubmed.ncbi.nlm.nih.gov/) — 37M+ biomedical records
   - [INSPIRE-HEP](https://inspirehep.net/) — 1.7M+ physics papers
   - [NASA ADS](https://ui.adsabs.harvard.edu/) — 28M+ astronomy records
   - [DBLP](https://dblp.org/) — 7M+ CS publications
   - [zbMATH](https://zbmath.org/) — 4.8M+ math records
   - [HAL](https://hal.science/) — 4.5M+ French research papers
   - [arXiv](https://arxiv.org/), [OpenLibrary](https://openlibrary.org/), [Unpaywall](https://unpaywall.org/)
4. **Report** — Color-coded verdicts with confidence scores. Flags retracted papers. Links to open access versions.

## Verification is deterministic

CiteGuard does not use LLMs to verify citations. Every verdict is traceable to an API response from an authoritative database. Nothing is inferred or generated.

GROBID is used only for citation *extraction* (parsing raw reference strings into structured fields). The verification itself is pure database lookup + fuzzy matching.

## Tested against real papers

| Paper | Discipline | Citations | Verified |
|-------|-----------|-----------|----------|
| GANs | CS / ML | 31 | 97% |
| Attention Is All You Need | CS / ML | 40 | 88% |
| BERT | NLP | 52 | 83% |
| ResNet | Computer Vision | 52 | 83% |
| AlphaFold | Bioinformatics | 69 | 65% |
| Deep Hedging | Finance | 32 | 56% |
| Ricci Flow (Perelman) | Mathematics | 27 | 11% |
| LIGO Gravitational Waves | Physics | 118 | 7% |

Physics and math papers score low because they use compact citation formats (journal abbreviation + volume + page, no title). GROBID integration improves extraction of structured fields from these formats.

## Quick start

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

Then open http://localhost:7860

### Docker

```bash
docker compose up --build
```

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CROSSREF_MAIL` | Email for CrossRef/OpenAlex polite pool | `anonymous@example.com` |
| `GROBID_URL` | GROBID API base URL | `https://kermitt2-grobid.hf.space` |
| `ADS_API_KEY` | NASA ADS API key (optional) | — |
| `NCBI_API_KEY` | PubMed API key for higher rate limits (optional) | — |

## License

GPL-3.0
