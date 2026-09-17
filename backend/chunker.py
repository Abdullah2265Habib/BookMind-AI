import re
from typing import List, Dict, Any

class SemanticChunker:
    """
    Divides structured document elements into semantic chunks
    enriched with hierarchical metadata (book, chapter, section, page).
    """

    def __init__(self, target_size: int = 1200, overlap: int = 200):
        self.target_size = target_size
        self.overlap = overlap

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits paragraph into sentences cleanly."""
        # Splits on period/exclamation/question followed by space and uppercase
        sentences = re.split(r'(?<=[.?!])\s+(?=[A-Z0-9])', text)
        return [s.strip() for s in sentences if s.strip()]

    def create_chunks(self, book_id: str, book_title: str, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        chunk_idx = 0

        current_chapter = ""
        current_section = ""
        current_subsection = ""
        current_pages = set()
        current_text_buffer = []
        current_buffer_length = 0

        def flush_buffer():
            nonlocal chunk_idx, current_text_buffer, current_buffer_length, current_pages
            if not current_text_buffer:
                return

            full_text = "\n\n".join(current_text_buffer).strip()
            if not full_text:
                return

            pages_list = sorted(list(current_pages))
            page_str = str(pages_list[0]) if len(pages_list) == 1 else f"{pages_list[0]}-{pages_list[-1]}"

            # Construct contextual breadcrumb
            breadcrumb_parts = []
            if current_chapter:
                breadcrumb_parts.append(f"Chapter: {current_chapter}")
            if current_section:
                breadcrumb_parts.append(f"Section: {current_section}")
            if current_subsection:
                breadcrumb_parts.append(f"Subsection: {current_subsection}")
            breadcrumb_parts.append(f"Page: {page_str}")
            header_context = " | ".join(breadcrumb_parts)

            # Enriched text includes breadcrumb for higher semantic search accuracy
            enriched_content = f"[{header_context}]\n\n{full_text}"

            chunks.append({
                "chunk_id": f"{book_id}_chunk_{chunk_idx}",
                "index": chunk_idx,
                "book_id": book_id,
                "book_title": book_title,
                "chapter": current_chapter or "General",
                "section": current_section or "Overview",
                "subsection": current_subsection,
                "page_number": pages_list[0] if pages_list else 1,
                "page_range": page_str,
                "header_context": header_context,
                "content": full_text,
                "enriched_content": enriched_content,
                "char_count": len(full_text),
                "estimated_tokens": max(1, len(full_text) // 4)
            })
            chunk_idx += 1

            # Semantic sliding overlap
            if self.overlap > 0 and len(full_text) > self.target_size:
                # Keep the last segment as overlap in next buffer
                last_piece = full_text[-self.overlap:]
                current_text_buffer = [last_piece]
                current_buffer_length = len(last_piece)
                # Keep last page in current_pages
                if pages_list:
                    current_pages = {pages_list[-1]}
            else:
                current_text_buffer = []
                current_buffer_length = 0
                current_pages = set()

        for el in elements:
            el_type = el.get("type", "paragraph")
            el_chap = el.get("chapter", "")
            el_sec = el.get("section", "")
            el_subsec = el.get("subsection", "")
            el_page = el.get("page_number", 1)
            el_text = el.get("content", "").strip()

            if not el_text:
                continue

            # If moving to a major new chapter or section and we have significant buffer, flush
            is_major_boundary = (
                (el_chap and el_chap != current_chapter) or
                (el_sec and el_sec != current_section and current_buffer_length > (self.target_size // 2))
            )

            if is_major_boundary and current_text_buffer:
                flush_buffer()

            current_chapter = el_chap or current_chapter
            current_section = el_sec or current_section
            current_subsection = el_subsec or current_subsection

            # If element is exceptionally large (larger than target_size), break into sentence blocks
            if len(el_text) > self.target_size:
                sentences = self._split_into_sentences(el_text)
                for sentence in sentences:
                    if current_buffer_length + len(sentence) > self.target_size and current_text_buffer:
                        flush_buffer()
                    current_text_buffer.append(sentence)
                    current_buffer_length += len(sentence)
                    current_pages.add(el_page)
            else:
                if current_buffer_length + len(el_text) > self.target_size and current_text_buffer:
                    flush_buffer()
                current_text_buffer.append(el_text)
                current_buffer_length += len(el_text)
                current_pages.add(el_page)

        # Flush any remaining items
        flush_buffer()

        return chunks
