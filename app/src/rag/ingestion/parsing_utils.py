# utils/parsing.py
import pdfplumber
import fitz # pymupdf
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter

def parse_pdf_with_sections(file_path):
    """
    Extracts sections based off PDF metadata if available, 
    otherwise orchestrates the extraction of section titles using three independent methods:
    1. Font Size (Visual Hierarchy)
    2. Bold Emphasis (Text Styling)
    3. Pattern Matching (Keywords & Case)
    """
    all_headings = []
    seen_headings = set()

    headings = _extract_headings_from_toc(file_path)
    all_headings.extend(headings)
    if not all_headings:
        print("🔍 No TOC found in metadata. Falling back to visual analysis...")
        try:
            with pdfplumber.open(file_path) as pdf:
                # Method 1: Font Size
                font_headings = _extract_by_font_size(pdf, seen_headings)
                all_headings.extend(font_headings)

                # Method 2: Bold Detection
                bold_headings = _extract_by_bold(pdf, seen_headings)
                all_headings.extend(bold_headings)

                # Method 3: Patterns & Case. Only use as a fallback if the first two methods did not return results
                if not all_headings:
                    pattern_headings = _extract_by_patterns(pdf, seen_headings)
                    all_headings.extend(pattern_headings)

        except Exception as e:
            print(f"⚠️ Error opening PDF: {e}")

    # Sort by page number to maintain document flow
    all_headings.sort(key=lambda x: x['page'])

    # DEBUG OUTPUT
    print_headings(all_headings)
        
    return all_headings

def parse_markdown_with_sections(markdown_text: str):
    """
    Splits markdown into logical blocks while preserving header hierarchy in metadata.
    """
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False # Keep headers in text for vector similarity
    )
    
    return header_splitter.split_text(markdown_text)


def get_breadcrumb_section(metadata: dict) -> str:
    """
    Collapses hierarchical headers into a single breadcrumb string.
    Example: 'Animal Welfare > Fish > Welfare Range'
    """
    levels = ["h1", "h2", "h3"]
    path = [metadata.get(lv) for lv in levels if metadata.get(lv)]
    
    return " > ".join(path) if path else "Main Content"

def clean_forum_content(text):
    """
        TODO: Specific logic for cleaning forum artifacts,
        like signatures, 'Read More' buttons, or vote counts
    """
    return text




# ============================================================================
# PDF SECTION EXTRACTION 
# ============================================================================
def get_safe_pdf_info(file_path):
    """Extracts document-level metadata safely."""
    with pdfplumber.open(file_path) as pdf:
        info = pdf.metadata
        
        return {
            "doc_title_internal": str(info.get("Title", ""))[:100],
            "doc_author": str(info.get("Author", ""))[:100],
        }

def detect_pdf_publication_date(file_path: str) -> str:
    """
    Best-effort publication date detection for PDFs.
    Returns ISO-like YYYY-MM-DD when possible; otherwise empty string.
    Priority:
      1) PDF metadata dates (CreationDate/ModDate)
      2) First-page text date patterns
      3) Filename date hints
    """
    # 1) Metadata date candidates
    try:
        with pdfplumber.open(file_path) as pdf:
            meta = pdf.metadata or {}
            for k in ("CreationDate", "ModDate", "Date", "Creationdate", "Moddate"):
                v = meta.get(k)
                parsed = _parse_pdf_date_value(str(v) if v else "")
                if parsed:
                    return parsed
            # 2) First-page text fallback
            if pdf.pages:
                text = pdf.pages[0].extract_text() or ""
                parsed = _parse_date_from_text(text)
                if parsed:
                    return parsed
    except Exception:
        pass

    # 3) Filename hints
    parsed = _parse_date_from_text(file_path)
    return parsed or ""

def _parse_pdf_date_value(raw: str) -> str:
    """
    Parse PDF metadata date formats like:
    D:20240501123000Z or 2024-05-01
    """
    if not raw:
        return ""
    s = raw.strip()
    m = re.search(r"D:(\d{4})(\d{2})(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return _parse_date_from_text(s)

def _parse_date_from_text(text: str) -> str:
    if not text:
        return ""
    # ISO-ish dates first
    m = re.search(r"\b(19|20)\d{2}[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b", text)
    if m:
        d = m.group(0).replace("/", "-")
        return d
    # Month name + year, normalize to first day of month
    month_map = {
        "january":"01","february":"02","march":"03","april":"04","may":"05","june":"06",
        "july":"07","august":"08","september":"09","october":"10","november":"11","december":"12"
    }
    m = re.search(r"\b(" + "|".join(month_map.keys()) + r")\s+((?:19|20)\d{2})\b", text.lower())
    if m:
        return f"{m.group(2)}-{month_map[m.group(1)]}-01"
    # Year only
    m = re.search(r"\b((?:19|20)\d{2})\b", text)
    if m:
        return f"{m.group(1)}-01-01"
    return ""
        
def _extract_headings_from_toc(file_path):
    """Extracts structural bookmarks from the PDF metadata."""
    headings = []
    try:
        doc = fitz.open(file_path)
        toc = doc.get_toc()  # Returns: [[level, title, page_number], ...]
        for level, title, page in toc:
            headings.append({
                'text': title.strip(),
                'page': page,  # Note: fitz uses 1-based indexing for TOC. Subtract to make 0-based 
                'method': 'metadata_toc'
            })
    except Exception as e:
        print(f"⚠️ Metadata TOC extraction failed: {e}")
        
    return headings

def _extract_by_font_size(pdf, seen_headings):
    """Identifies headings based on visual hierarchy (font size)."""
    headings = []
    font_sizes = {}
    
    # 1. Profile font sizes
    for page in pdf.pages:
        for char in page.chars:
            size = round(char['size'], 1)
            font_sizes[size] = font_sizes.get(size, 0) + 1
            
    if not font_sizes:
        return []

    # 2. Determine heading thresholds
    sorted_sizes = sorted(font_sizes.items(), key=lambda x: x[1], reverse=True)
    median_size = sorted(font_sizes.keys())[len(font_sizes) // 2]
    heading_sizes = {size for size, count in sorted_sizes if size > median_size and count >= 2}
    
    if not heading_sizes:
        heading_sizes = {size for size, _ in sorted_sizes[:3]}

    # 3. Extract text
    for page_num, page in enumerate(pdf.pages, 1):
        current_line, current_size = "", None
        for char in page.chars:
            char_size = round(char['size'], 1)
            if char_size in heading_sizes:
                if char['text'] == '\n' or (current_size and current_size != char_size):
                    _add_if_valid(headings, current_line, page_num, 'font_size', seen_headings)
                    current_line = ""
                if char['text'] != '\n':
                    current_line += char['text']
                current_size = char_size
            else:
                _add_if_valid(headings, current_line, page_num, 'font_size', seen_headings)
                current_line, current_size = "", None
    return headings

def _extract_by_bold(pdf, seen_headings):
    """Captures lines where every non-whitespace character is bold."""
    headings = []
    for page_num, page in enumerate(pdf.pages, 1):
        current_line = ""
        line_is_all_bold, non_space_count = True, 0
        
        for char in page.chars:
            is_bold = 'bold' in char.get('fontname', '').lower()
            if char['text'] == '\n':
                if line_is_all_bold and non_space_count > 0:
                    _add_if_valid(headings, current_line, page_num, 'bold', seen_headings)
                current_line, line_is_all_bold, non_space_count = "", True, 0
            else:
                current_line += char['text']
                if char['text'].strip():
                    non_space_count += 1
                    if not is_bold: line_is_all_bold = False
    return headings

def _extract_by_patterns(pdf, seen_headings):
    """Matches common heading keywords and ALL-CAPS styling."""
    headings = []
    patterns = ['§', 'Chapter', 'Section', 'Article', 'Subchapter', 'Part ', 
                'Summary', 'Introduction', 'Findings', 'Conclusion']
    
    for page_num, page in enumerate(pdf.pages, 1):
        text = page.extract_text()
        if not text: continue
        
        for line in text.split('\n'):
            stripped = line.strip()
            # Check keywords or ALL-CAPS
            is_pattern = any(p in stripped for p in patterns)
            is_caps = stripped.isupper() and 5 < len(stripped) < 100 and len(stripped.split()) < 15
            
            if is_pattern or is_caps:
                _add_if_valid(headings, stripped, page_num, 'pattern', seen_headings)
    return headings

def _add_if_valid(collection, text, page, method, seen_set):
    """Helper to validate and deduplicate findings."""
    clean_text = text.strip()
    if 3 < len(clean_text) < 200 and clean_text not in seen_set:
        collection.append({'text': clean_text, 'page': page, 'method': method})
        seen_set.add(clean_text)

def print_headings(headings):
    """
    Pretty-print all detected section titles.
    
    Shows: number, detection method, page, and text (truncated if long).
    This helps verify that section detection is working correctly.
    
    OUTPUT FORMAT:
    📋 Section titles:
        1. [font_size] p.  1 - Subchapter 3: Rental Horse Licensing...
        2. [bold     ] p.  1 - § 17-326 Definitions.
        3. [pattern  ] p.  2 - § 17-327 License required.
    
    INTERPRETATION:
    - "font_size" = Detected by large text size (high confidence)
    - "bold" = Detected by entire line being bold (medium confidence)
    - "pattern" = Detected by § symbol or legal keywords (pattern-based)
    """
    if headings:
        print(f"\n📋 Section titles:")
        for i, h in enumerate(headings, 1):
            method = h.get('method', 'unknown')
            page = h.get('page', '?')
            text = h['text']
            # Truncate long titles for display
            display_text = text[:70] + "..." if len(text) > 70 else text
            print(f"   {i:2d}. [{method:8s}] p.{page:3d} - {display_text}")
    else:
        print(f"\n⚠️  No section titles found")
