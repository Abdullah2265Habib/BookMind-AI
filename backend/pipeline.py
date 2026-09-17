import os
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Callable, Optional

from backend.config import UPLOADS_DIR, BOOKS_DIR, NUM_WORKERS
from backend.analyzer import DocumentAnalyzer
from backend.parser import DocumentParser
from backend.chunker import SemanticChunker
from backend.vector_store import BookVectorStore

class BookPipeline:
    """
    Automated Generic Book-to-AI Pipeline:
    PDF Upload -> 4-Core Analysis/OCR -> Structure Parsing -> Semantic Chunking -> Vector/BM25 Index
    """

    def __init__(self, num_workers: int = NUM_WORKERS):
        self.num_workers = num_workers
        self.analyzer = DocumentAnalyzer(max_workers=num_workers)
        self.parser = DocumentParser()
        self.chunker = SemanticChunker()

    def process_book(
        self,
        pdf_path: str,
        book_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes the entire 4-stage pipeline for a PDF document.
        """
        start_time = time.time()
        file_path = Path(pdf_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not book_id:
            import uuid
            # Clean safe filename for id
            safe_stem = "".join(c if c.isalnum() else "_" for c in file_path.stem).lower()
            book_id = f"{safe_stem}_{uuid.uuid4().hex[:6]}"

        def notify(stage: str, percent: int, data: Optional[Dict[str, Any]] = None):
            if progress_callback:
                progress_callback(stage, percent, data or {})

        # --- Stage 1: Document Analysis & OCR (4 CPU Cores) ---
        notify("Analyzing document layout, scans, and typography on 4 CPU cores...", 15)
        analysis = self.analyzer.analyze(str(file_path))
        doc_title = analysis.get("title") or file_path.stem

        notify("Document analysis & OCR complete.", 40, {
            "title": doc_title,
            "total_pages": analysis["total_pages"],
            "document_type": analysis["document_type"],
            "scanned_pages": analysis["scanned_pages_count"],
            "total_images": analysis["total_images"],
            "language": analysis["primary_language"]
        })

        # --- Stage 2: Structural Parsing & Cleaning ---
        notify("Parsing document structure: chapters, sections, tables, page numbers...", 55)
        structured_elements = self.parser.parse(analysis)

        # --- Stage 3: Semantic Chunking ---
        notify("Generating semantic chunks with hierarchical metadata...", 70)
        chunks = self.chunker.create_chunks(book_id, doc_title, structured_elements)

        # --- Stage 4: Vector Embeddings & Knowledge Base ---
        notify("Building FastEmbed ONNX vectors and BM25 index on CPU...", 85)
        vector_store = BookVectorStore(book_id)

        # Copy original PDF into book directory for reference / viewing
        saved_pdf_copy = vector_store.book_dir / "document.pdf"
        if not saved_pdf_copy.exists():
            shutil.copy2(file_path, saved_pdf_copy)

        metadata = {
            "book_id": book_id,
            "title": doc_title,
            "author": analysis.get("author", "Unknown"),
            "subject": analysis.get("subject", ""),
            "total_pages": analysis["total_pages"],
            "document_type": analysis["document_type"],
            "scanned_pages_count": analysis["scanned_pages_count"],
            "ocr_applied_count": analysis["ocr_applied_count"],
            "total_images": analysis["total_images"],
            "primary_language": analysis["primary_language"],
            "total_chunks": len(chunks),
            "table_of_contents": analysis.get("table_of_contents", []),
            "processing_time_seconds": round(time.time() - start_time, 2),
            "cpu_cores_used": self.num_workers
        }

        vector_store.build_knowledge_base(metadata=metadata, chunks=chunks)

        notify("Pipeline completed successfully! Knowledge base ready.", 100, metadata)

        return {
            "status": "success",
            "book_id": book_id,
            "metadata": metadata,
            "chunks_count": len(chunks),
            "duration_seconds": metadata["processing_time_seconds"]
        }
