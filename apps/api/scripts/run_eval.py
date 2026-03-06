from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small local retrieval and citation benchmark."
    )
    parser.add_argument(
        "--benchmark",
        default="benchmarks/sample_eval.json",
        help="Path to a benchmark JSON file relative to apps/api or absolute.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k retrieval depth to use for each question.",
    )
    return parser.parse_args()


def create_pdf(page_texts: list[str]) -> bytes:
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    payload = document.tobytes()
    document.close()
    return payload


def resolve_benchmark_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parent.parent / candidate


def main() -> int:
    args = parse_args()
    benchmark_path = resolve_benchmark_path(args.benchmark)
    benchmark = json.loads(benchmark_path.read_text())

    with tempfile.TemporaryDirectory(prefix="ebook-rag-eval-") as temp_dir:
        temp_root = Path(temp_dir)
        os.environ["DATABASE_URL"] = f"sqlite:///{temp_root / 'eval.db'}"
        os.environ["UPLOADS_DIR"] = str(temp_root / "uploads")
        os.environ["MAX_UPLOAD_SIZE_MB"] = "5"

        from ebook_rag_api.core.config import get_settings
        from ebook_rag_api.db import Base, get_engine, get_session_factory
        from ebook_rag_api.main import create_app
        from ebook_rag_api.services.embeddings import get_embedding_provider
        from ebook_rag_api.services.qa import get_answer_provider
        from fastapi.testclient import TestClient

        get_settings.cache_clear()
        get_engine.cache_clear()
        get_session_factory.cache_clear()
        get_embedding_provider.cache_clear()
        get_answer_provider.cache_clear()

        app = create_app()
        Base.metadata.create_all(bind=get_engine())

        total_questions = 0
        retrieval_hits = 0
        citation_hits = 0
        support_hits = 0
        answer_hits = 0
        latencies_ms: list[float] = []
        results: list[dict[str, Any]] = []

        with TestClient(app) as client:
            uploaded_documents: dict[str, str] = {}
            for document_case in benchmark["documents"]:
                response = client.post(
                    "/api/documents/upload",
                    files={
                        "file": (
                            document_case["filename"],
                            create_pdf(document_case["pages"]),
                            "application/pdf",
                        )
                    },
                )
                response.raise_for_status()
                payload = response.json()
                uploaded_documents[document_case["filename"]] = payload["document"]["id"]

            for document_case in benchmark["documents"]:
                document_id = uploaded_documents[document_case["filename"]]
                for question_case in document_case["questions"]:
                    total_questions += 1
                    response = client.post(
                        "/api/qa/ask",
                        json={
                            "question": question_case["question"],
                            "document_id": document_id,
                            "top_k": args.top_k,
                            "include_trace": True,
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    trace = payload["trace"]
                    expected_pages = set(question_case.get("expected_citation_pages", []))
                    retrieved_pages = {
                        chunk["page_start"] for chunk in trace["retrieved_chunks"]
                    }
                    cited_pages = {
                        citation["page_start"] for citation in payload["citations"]
                    }
                    retrieval_hit = (
                        not expected_pages
                        or bool(retrieved_pages & expected_pages)
                    )
                    citation_hit = (
                        not expected_pages
                        or bool(cited_pages & expected_pages)
                    )
                    support_hit = (
                        payload["supported"] == question_case["expected_supported"]
                    )
                    answer_terms = [
                        term.lower()
                        for term in question_case.get("expected_answer_contains", [])
                    ]
                    answer_hit = (
                        not answer_terms
                        or any(term in payload["answer"].lower() for term in answer_terms)
                    )
                    retrieval_hits += int(retrieval_hit)
                    citation_hits += int(citation_hit)
                    support_hits += int(support_hit)
                    answer_hits += int(answer_hit)
                    latencies_ms.append(trace["timings"]["total_ms"])
                    results.append(
                        {
                            "document": document_case["filename"],
                            "question": question_case["question"],
                            "supported": payload["supported"],
                            "expected_supported": question_case["expected_supported"],
                            "retrieval_hit": retrieval_hit,
                            "citation_hit": citation_hit,
                            "answer_hit": answer_hit,
                            "latency_ms": trace["timings"]["total_ms"],
                        }
                    )

        summary = {
            "benchmark": benchmark["name"],
            "questions": total_questions,
            "retrieval_hit_rate": retrieval_hits / total_questions if total_questions else 0.0,
            "citation_hit_rate": citation_hits / total_questions if total_questions else 0.0,
            "support_accuracy": support_hits / total_questions if total_questions else 0.0,
            "answer_match_rate": answer_hits / total_questions if total_questions else 0.0,
            "average_latency_ms": mean(latencies_ms) if latencies_ms else 0.0,
            "results": results,
        }
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
