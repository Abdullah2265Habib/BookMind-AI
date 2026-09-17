import os
import sys
import re
import fitz  # PyMuPDF
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any, Tuple
from pathlib import Path

def _ocr_page_worker(page_img_bytes: bytes) -> str:
    """Worker function for running OCR on a single rendered page pixmap using RapidOCR."""
    try:
        from rapidocr_onnxruntime import RapidOCR
        import numpy as np
        import cv2

        nparr = np.frombuffer(page_img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return ""
        
        engine = RapidOCR()
        result, _ = engine(img)
        if result:
            lines = [line[1] for line in result if line and len(line) > 1]
            return "\n".join(lines)
        return ""
    except Exception as e:
        return f"[OCR Error: {str(e)}]"

def _analyze_single_page(args: Tuple[str, int]) -> Dict[str, Any]:
    """Analyzes a single page of a PDF. Runs inside process pool."""
    pdf_path, page_num = args
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # 1. Extract digital text
    text = page.get_text("text") or ""
    clean_text = text.strip()
    char_count = len(clean_text)

    # 2. Count images
    images = page.get_images()
    image_count = len(images)

    # 3. Detect tables/drawings
    drawings = page.get_drawings()
    has_drawings = len(drawings) > 5

    # 4. Classify text-based vs scanned
    # If text is extremely short or nonexistent, but images exist, consider scanned
    is_scanned = False
    ocr_applied = False
    ocr_text = ""

    # Scanned detection heuristic
    if char_count < 50 and image_count > 0:
        is_scanned = True
        # Render page pixmap to bytes for OCR
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        ocr_text = _ocr_page_worker(img_bytes)
        if ocr_text:
            text = ocr_text
            clean_text = text.strip()
            char_count = len(clean_text)
            ocr_applied = True

    # Check for basic language features
    sample_text = clean_text[:500].lower()
    lang = "en"
    if re.search(r'[\u0600-\u06FF]', sample_text):
        lang = "ar"
    elif re.search(r'[\u4e00-\u9fff]', sample_text):
        lang = "zh"
    elif re.search(r'[\u0400-\u04FF]', sample_text):
        lang = "ru"
    elif re.search(r'[\u0900-\u097F]', sample_text):
        lang = "hi"

    # Extract block structure for layout preservation
    blocks = page.get_text("blocks")
    # block tuple: (x0, y0, x1, y1, text, block_no, block_type)
    extracted_blocks = []
    for b in blocks:
        if b[6] == 0:  # text block
            b_text = b[4].strip()
            if b_text:
                extracted_blocks.append({
                    "bbox": [round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2)],
                    "text": b_text,
                    "block_id": b[5]
                })

    doc.close()

    return {
        "page_number": page_num + 1,
        "is_scanned": is_scanned,
        "ocr_applied": ocr_applied,
        "char_count": char_count,
        "image_count": image_count,
        "has_drawings": has_drawings,
        "language": lang,
        "text": clean_text,
        "blocks": extracted_blocks
    }

class DocumentAnalyzer:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def analyze(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyzes the uploaded PDF document using 4 parallel CPU processes.
        Determines:
        - Text-based vs scanned status per page and overall
        - Total page count, images, tables/drawings
        - Preserves page blocks and detected language
        """
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        metadata = doc.metadata or {}
        toc = doc.get_toc() # [[lvl, title, page], ...]
        doc.close()

        tasks = [(pdf_path, p) for p in range(total_pages)]
        
        # Parallel execution across 4 CPU cores
        pages_analysis = []
        if total_pages == 1:
            pages_analysis = [_analyze_single_page(tasks[0])]
        else:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                pages_analysis = list(executor.map(_analyze_single_page, tasks))

        # Sort pages by page number to guarantee proper order
        pages_analysis.sort(key=lambda x: x["page_number"])

        scanned_pages = sum(1 for p in pages_analysis if p["is_scanned"])
        ocr_applied_pages = sum(1 for p in pages_analysis if p["ocr_applied"])
        total_chars = sum(p["char_count"] for p in pages_analysis)
        total_images = sum(p["image_count"] for p in pages_analysis)

        doc_type = "scanned" if scanned_pages > (total_pages / 2) else "text-based"
        if 0 < scanned_pages <= (total_pages / 2):
            doc_type = "hybrid (mixed text & scanned)"

        languages = [p["language"] for p in pages_analysis if p["language"]]
        primary_lang = max(set(languages), key=languages.count) if languages else "en"

        return {
            "title": metadata.get("title") or Path(pdf_path).stem,
            "author": metadata.get("author") or "Unknown Author",
            "subject": metadata.get("subject") or "",
            "keywords": metadata.get("keywords") or "",
            "total_pages": total_pages,
            "document_type": doc_type,
            "scanned_pages_count": scanned_pages,
            "ocr_applied_count": ocr_applied_pages,
            "total_images": total_images,
            "total_characters": total_chars,
            "primary_language": primary_lang,
            "table_of_contents": [{"level": item[0], "title": item[1], "page": item[2]} for item in toc],
            "pages": pages_analysis
        }
