import re
from typing import List, Dict, Any, Optional

class DocumentParser:
    """
    Parses document pages into structured semantic elements
    preserving chapters, sections, subsections, paragraphs, tables,
    figure captions, equations, and page numbers.
    Cleans running headers, footers, hyphenations, and artifacts.
    """

    def __init__(self):
        # Heading Regexes
        self.chapter_pattern = re.compile(r'^(?:chapter|ch\.)\s*(\d+|[ivxlcdm]+)[:\.\s]*(.*)$', re.IGNORECASE)
        self.section_pattern = re.compile(r'^(\d+)\.\s+([A-Z0-9][^\n\.\:\;]{2,80})$')
        self.subsection_pattern = re.compile(r'^(\d+\.\d+)\.?\s+([A-Z0-9][^\n\.\:\;]{2,80})$')
        self.subsubsection_pattern = re.compile(r'^(\d+\.\d+\.\d+)\.?\s+([A-Z0-9][^\n\.\:\;]{2,80})$')
        self.table_pattern = re.compile(r'^(?:table|tbl\.)\s*(\d+)[:\.\s]*(.*)$', re.IGNORECASE)
        self.figure_pattern = re.compile(r'^(?:figure|fig\.)\s*(\d+)[:\.\s]*(.*)$', re.IGNORECASE)
        self.equation_pattern = re.compile(r'^(?:equation|eq\.)\s*(\d+)[:\.\s]*(.*)$|^\s*\(\d+\)\s*$', re.IGNORECASE)

    def clean_text(self, text: str) -> str:
        """Removes formatting artifacts, hyphenated line breaks, and OCR noise."""
        if not text:
            return ""

        # Normalize linebreaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Re-join hyphenated words split across lines (e.g., 'trans-\nmission' -> 'transmission')
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

        # Remove strange unicode replacement chars
        text = text.replace('\ufffd', ' ').replace('\u0000', '')

        # Standardize quotation marks and dashes
        text = re.sub(r'[\u2018\u2019]', "'", text)
        text = re.sub(r'[\u201C\u201D]', '"', text)
        text = re.sub(r'[\u2013\u2014]', '-', text)

        # Consolidate multiple consecutive whitespaces (while preserving single linebreaks)
        lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
        
        # Remove empty lines excess
        cleaned_lines = []
        for line in lines:
            if line:
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")

        return "\n".join(cleaned_lines)

    def detect_repeated_headers_footers(self, pages: List[Dict[str, Any]]) -> set:
        """Detects recurring header/footer strings that appear at the top or bottom of multiple pages."""
        candidate_lines = {}
        for p in pages:
            lines = p.get("text", "").split("\n")
            if len(lines) >= 1:
                # First 2 lines (header candidate)
                for line in lines[:2]:
                    s = line.strip()
                    if 3 < len(s) < 120 and not re.match(r'^\d+$', s):
                        candidate_lines[s] = candidate_lines.get(s, 0) + 1
                # Last 2 lines (footer candidate)
                for line in lines[-2:]:
                    s = line.strip()
                    if 3 < len(s) < 120 and not re.match(r'^\d+$', s):
                        candidate_lines[s] = candidate_lines.get(s, 0) + 1

        # If a line appears on at least 3 pages or > 20% of pages, mark as header/footer
        threshold = max(2, int(len(pages) * 0.2))
        return {line for line, count in candidate_lines.items() if count >= threshold}

    def parse(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extracts structured semantic nodes from analyzed pages.
        Each node includes:
        - type: 'heading', 'subheading', 'paragraph', 'table', 'figure', 'equation'
        - chapter: current chapter title/number
        - section: current section title/number
        - subsection: current subsection title/number
        - page_number: integer page number
        - content: normalized text
        """
        pages = analysis_result.get("pages", [])
        boilerplate = self.detect_repeated_headers_footers(pages)
        toc = analysis_result.get("table_of_contents", [])

        current_chapter = "General"
        current_section = "Overview"
        current_subsection = ""

        # Pre-populate chapter if document has title
        doc_title = analysis_result.get("title", "")
        if doc_title:
            current_chapter = doc_title

        structured_elements = []

        for p in pages:
            page_num = p["page_number"]
            page_blocks = p.get("blocks", [])

            raw_paragraphs = []
            if page_blocks:
                for b in page_blocks:
                    b_text = b.get("text", "").strip()
                    if b_text:
                        raw_paragraphs.append(b_text)
            else:
                raw_text = p.get("text", "")
                raw_paragraphs = [para.strip() for para in raw_text.split("\n\n") if para.strip()]

            for raw_para in raw_paragraphs:
                # Clean linebreaks within paragraph: replace single \n with space, un-hyphenate
                para = self.clean_text(raw_para)
                if not para:
                    continue

                # Filter repeated header/footer boilerplate
                if para in boilerplate or any(bp in para for bp in boilerplate if len(bp) > 25):
                    continue

                # Strip lone page numbers
                if re.match(r'^\d{1,4}$', para):
                    continue

                lines = [l.strip() for l in para.split("\n") if l.strip()]
                if not lines:
                    continue
                first_line = lines[0]

                # 1. Chapter Match
                chap_match = self.chapter_pattern.match(first_line)
                if chap_match:
                    current_chapter = f"Chapter {chap_match.group(1)}: {chap_match.group(2).strip()}".strip(" :")
                    current_section = "Introduction"
                    current_subsection = ""
                    structured_elements.append({
                        "type": "chapter",
                        "chapter": current_chapter,
                        "section": current_section,
                        "subsection": current_subsection,
                        "page_number": page_num,
                        "content": para
                    })
                    continue

                # 2. Section Match (e.g., '1. Introduction', '3. Methodology')
                sec_match = self.section_pattern.match(first_line)
                if sec_match:
                    current_section = f"{sec_match.group(1)}. {sec_match.group(2).strip()}"
                    current_subsection = ""
                    structured_elements.append({
                        "type": "section",
                        "chapter": current_chapter,
                        "section": current_section,
                        "subsection": current_subsection,
                        "page_number": page_num,
                        "content": para
                    })
                    continue

                # 3. Subsection Match (e.g., '2.1. Terahertz Communication')
                subsec_match = self.subsection_pattern.match(first_line)
                if subsec_match:
                    current_subsection = f"{subsec_match.group(1)} {subsec_match.group(2).strip()}"
                    structured_elements.append({
                        "type": "subsection",
                        "chapter": current_chapter,
                        "section": current_section,
                        "subsection": current_subsection,
                        "page_number": page_num,
                        "content": para
                    })
                    continue

                # 4. Sub-subsection Match (e.g., '2.1.1. Architecture')
                sub3_match = self.subsubsection_pattern.match(first_line)
                if sub3_match:
                    current_subsection = f"{sub3_match.group(1)} {sub3_match.group(2).strip()}"
                    structured_elements.append({
                        "type": "subsection",
                        "chapter": current_chapter,
                        "section": current_section,
                        "subsection": current_subsection,
                        "page_number": page_num,
                        "content": para
                    })
                    continue

                # 5. Table Match
                if self.table_pattern.match(first_line):
                    structured_elements.append({
                        "type": "table",
                        "chapter": current_chapter,
                        "section": current_section,
                        "subsection": current_subsection,
                        "page_number": page_num,
                        "content": para
                    })
                    continue

                # 6. Figure Match
                if self.figure_pattern.match(first_line):
                    structured_elements.append({
                        "type": "figure",
                        "chapter": current_chapter,
                        "section": current_section,
                        "subsection": current_subsection,
                        "page_number": page_num,
                        "content": para
                    })
                    continue

                # 7. Equation Match
                if self.equation_pattern.match(first_line):
                    structured_elements.append({
                        "type": "equation",
                        "chapter": current_chapter,
                        "section": current_section,
                        "subsection": current_subsection,
                        "page_number": page_num,
                        "content": para
                    })
                    continue

                # 8. All-caps short heading (e.g., 'ABSTRACT', 'CONCLUSION', 'REFERENCES')
                if len(first_line) < 40 and first_line.isupper() and len(first_line.split()) < 5:
                    current_section = first_line.title()
                    current_subsection = ""
                    structured_elements.append({
                        "type": "section",
                        "chapter": current_chapter,
                        "section": current_section,
                        "subsection": current_subsection,
                        "page_number": page_num,
                        "content": para
                    })
                    continue

                structured_elements.append({
                    "type": "paragraph",
                    "chapter": current_chapter,
                    "section": current_section,
                    "subsection": current_subsection,
                    "page_number": page_num,
                    "content": para
                })

        return structured_elements
