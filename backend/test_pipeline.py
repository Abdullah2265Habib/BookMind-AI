import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add workspace to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.pipeline import BookPipeline
from backend.vector_store import BookVectorStore
from backend.rag_engine import RAGEngine
import asyncio

def progress_logger(stage: str, percent: int, data: dict):
    print(f"[{percent:3d}%] {stage}")
    if data and "total_pages" in data:
        print(f"       Pages: {data.get('total_pages')}, Type: {data.get('document_type')}, Language: {data.get('language')}")

async def main():
    test_pdf = BASE_DIR / "PDF" / "next generation computer networks.pdf"
    print("=" * 60)
    print("Testing Generic Book-to-AI Pipeline")
    print(f"Target PDF: {test_pdf}")
    print(f"File exists: {test_pdf.exists()} ({test_pdf.stat().st_size / 1024 / 1024:.2f} MB)")
    print("=" * 60)

    pipeline = BookPipeline(num_workers=4)
    result = pipeline.process_book(
        pdf_path=str(test_pdf),
        book_id="book_6g_networks",
        progress_callback=progress_logger
    )

    print("\n--- Pipeline Completed Successfully! ---")
    print("Book ID:", result["book_id"])
    print("Duration:", result["duration_seconds"], "seconds")
    print("Chunks Generated:", result["chunks_count"])
    meta = result["metadata"]
    print("Title:", meta["title"])
    print("Pages:", meta["total_pages"])
    print("Images:", meta["total_images"])

    print("\n" + "=" * 60)
    print("Testing Hybrid Search & RAG Grounding")
    print("=" * 60)

    store = BookVectorStore("book_6g_networks")
    test_query = "What are the primary enabling technologies for 6G wireless communication?"
    print(f"Query: '{test_query}'\n")

    rag = RAGEngine()
    response = await rag.answer(book_store=store, query=test_query, provider="local")

    print("\n--- AI Answer with Grounded Citations ---")
    print(response["answer"])

    print("\n--- Retrieved Citations & Passages ---")
    for i, cite in enumerate(response["citations"]):
        print(f"[{i+1}] {cite['header_context']} (RRF Score: {cite['rrf_score']})")
        print(f"    Snippet: {cite['snippet'][:150]}...\n")

if __name__ == "__main__":
    asyncio.run(main())
