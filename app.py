"""CiteGuard Web App — FastAPI backend for citation verification."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict

import fitz  # PyMuPDF
from docx import Document as DocxDocument
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from citation_extractor import extract_citations
from verification_engine import compute_summary, verify_citations

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
JOB_TTL_SECONDS = 3600  # 1 hour

app = FastAPI(title="CiteGuard", version="1.0")

# In-memory job store: job_id -> {status, created_at, ...}
jobs: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(data: bytes) -> str:
    doc = fitz.open(stream=data, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def extract_text_from_docx(data: bytes) -> str:
    import io
    doc = DocxDocument(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(data)
    elif ext == ".docx":
        return extract_text_from_docx(data)
    elif ext == ".txt":
        return data.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# Job cleanup
# ---------------------------------------------------------------------------

def _evict_stale_jobs() -> None:
    """Remove jobs older than JOB_TTL_SECONDS."""
    now = time.time()
    stale = [jid for jid, j in jobs.items() if now - j.get("created_at", 0) > JOB_TTL_SECONDS]
    for jid in stale:
        del jobs[jid]


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------

def process_document(job_id: str, filename: str, data: bytes) -> None:
    """Run citation extraction and verification in the background."""
    try:
        text = extract_text(filename, data)
        citations = extract_citations(text)
        # Build ref_section from extracted citation raw texts for style detection
        ref_section = "\n".join(c.raw_text for c in citations)
        results = verify_citations(citations, ref_section=ref_section)
        summary = compute_summary(results)

        jobs[job_id].update({
            "status": "complete",
            "citations": [r.to_dict() for r in results],
            "summary": summary,
        })
    except Exception as exc:
        jobs[job_id].update({
            "status": "error",
            "error": f"Processing failed: {type(exc).__name__}",
        })


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_file(file: UploadFile, background_tasks: BackgroundTasks) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read in chunks to enforce size limit without buffering oversized files
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(64 * 1024):
        total += len(chunk)
        if total > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File exceeds 10 MB limit")
        chunks.append(chunk)
    data = b"".join(chunks)

    _evict_stale_jobs()

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "processing", "created_at": time.time()}

    background_tasks.add_task(process_document, job_id, file.filename, data)

    return JSONResponse({"job_id": job_id})


@app.get("/api/report/{job_id}")
async def get_report(job_id: str) -> JSONResponse:
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(jobs[job_id])


# ---------------------------------------------------------------------------
# Static files (frontend)
# ---------------------------------------------------------------------------

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
