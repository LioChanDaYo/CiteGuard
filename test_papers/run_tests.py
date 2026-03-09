#!/usr/bin/env python3
"""Test CiteGuard citation extraction and verification against real papers."""

import json
import os
import sys
import time

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz
from citation_extractor import extract_citations, Citation
from verification_engine import verify_citations, compute_summary, VerificationResult


def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def test_paper(path: str) -> dict:
    """Test CiteGuard on a single paper. Returns detailed results."""
    name = os.path.basename(path)
    print(f"\n{'='*80}")
    print(f"TESTING: {name}")
    print(f"{'='*80}")

    # Extract text
    text = extract_text_from_pdf(path)
    print(f"  Text length: {len(text)} chars")

    # Extract citations
    t0 = time.time()
    citations = extract_citations(text)
    extract_time = time.time() - t0
    print(f"  Citations extracted: {len(citations)} (in {extract_time:.2f}s)")

    if not citations:
        print("  ⚠️  NO CITATIONS EXTRACTED!")
        return {
            "paper": name,
            "text_length": len(text),
            "citations_extracted": 0,
            "extraction_time": extract_time,
            "results": [],
            "summary": {"total": 0, "verified": 0, "suspect": 0, "not_found": 0},
        }

    # Show extraction details
    print(f"\n  --- Extraction Details ---")
    fields_missing = {"author": 0, "title": 0, "year": 0, "doi": 0}
    for cit in citations:
        if not cit.author:
            fields_missing["author"] += 1
        if not cit.title:
            fields_missing["title"] += 1
        if not cit.year:
            fields_missing["year"] += 1
        if not cit.doi:
            fields_missing["doi"] += 1

    for field, count in fields_missing.items():
        pct = count / len(citations) * 100
        status = "✅" if pct < 30 else "⚠️" if pct < 60 else "❌"
        print(f"  {status} {field} missing: {count}/{len(citations)} ({pct:.0f}%)")

    # Show first 3 citations for inspection
    print(f"\n  --- Sample Citations (first 3) ---")
    for cit in citations[:3]:
        print(f"  [{cit.index}] author={cit.author!r}")
        print(f"       title={cit.title!r}")
        print(f"       year={cit.year!r} doi={cit.doi!r} arxiv={cit.arxiv_id!r}")
        print(f"       raw={cit.raw_text[:120]!r}...")
        print()

    # Verify citations
    print(f"  --- Verification (this may take a while) ---")
    t0 = time.time()
    results = verify_citations(citations)
    verify_time = time.time() - t0
    summary = compute_summary(results)
    print(f"  Verification completed in {verify_time:.1f}s")
    print(f"  Total: {summary['total']} | Verified: {summary['verified']} | "
          f"Suspect: {summary['suspect']} | Not Found: {summary['not_found']} | "
          f"Risk: {summary['risk_level']}")

    # Show failures for analysis
    not_found = [r for r in results if r.verdict == "not_found"]
    suspect = [r for r in results if r.verdict == "suspect"]

    if not_found:
        print(f"\n  --- Not Found ({len(not_found)}) ---")
        for r in not_found[:5]:
            print(f"  [{r.index}] title={r.title!r}")
            print(f"       author={r.author!r} doi={r.doi!r}")
            print(f"       raw={r.raw_text[:120]!r}")
            print()

    if suspect:
        print(f"\n  --- Suspect ({len(suspect)}) ---")
        for r in suspect[:5]:
            print(f"  [{r.index}] title={r.title!r} confidence={r.confidence}")
            print(f"       matched_title={r.match_details.matched_title!r}")
            print()

    return {
        "paper": name,
        "text_length": len(text),
        "citations_extracted": len(citations),
        "extraction_time": extract_time,
        "verify_time": verify_time,
        "fields_missing": fields_missing,
        "summary": summary,
        "not_found_details": [
            {
                "index": r.index,
                "title": r.title,
                "author": r.author,
                "doi": r.doi,
                "raw": r.raw_text[:200],
            }
            for r in not_found
        ],
        "suspect_details": [
            {
                "index": r.index,
                "title": r.title,
                "confidence": r.confidence,
                "matched_title": r.match_details.matched_title,
            }
            for r in suspect
        ],
    }


def main():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    pdfs = sorted([
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.endswith(".pdf")
    ])

    if not pdfs:
        print("No PDF files found in test_papers/")
        return

    print(f"Found {len(pdfs)} papers to test")

    all_results = []
    for pdf in pdfs:
        result = test_paper(pdf)
        all_results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    total_cit = sum(r["citations_extracted"] for r in all_results)
    total_verified = sum(r["summary"]["verified"] for r in all_results)
    total_suspect = sum(r["summary"]["suspect"] for r in all_results)
    total_not_found = sum(r["summary"]["not_found"] for r in all_results)

    print(f"Papers tested: {len(all_results)}")
    print(f"Total citations: {total_cit}")
    print(f"Verified: {total_verified} ({total_verified/max(total_cit,1)*100:.0f}%)")
    print(f"Suspect: {total_suspect} ({total_suspect/max(total_cit,1)*100:.0f}%)")
    print(f"Not Found: {total_not_found} ({total_not_found/max(total_cit,1)*100:.0f}%)")

    # Per-paper table
    print(f"\n{'Paper':<45} {'Cit':>4} {'V':>4} {'S':>4} {'NF':>4} {'V%':>5}")
    print("-" * 72)
    for r in all_results:
        s = r["summary"]
        total = max(s["total"], 1)
        v_pct = s["verified"] / total * 100
        print(f"{r['paper']:<45} {s['total']:>4} {s['verified']:>4} {s['suspect']:>4} {s['not_found']:>4} {v_pct:>5.0f}%")

    # Save detailed results
    output_path = os.path.join(test_dir, "test_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
