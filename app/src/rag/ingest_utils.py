# ingest_utils.py
#
# Utilities for PDF ingestion pipeline (ingest.py).
# Provides: text splitting, metadata building, section extraction, hashing.
#
# PDF SECTION EXTRACTION STRATEGY (3-method combined approach):
# ────────────────────────────────────────────────────────────────────────
# 1. Font size detection: Identifies large text (visual hierarchy)
#    Example: "Subchapter 3: Rental Horse Licensing and Protection Law"
#
# 2. Bold detection: Captures lines where ENTIRE text is bold
#    Example: "§ 17-330 Regulations." (when all characters are bold)
#    Skips: "ASPCA." (only one word, mixed formatting)
#
# 3. Pattern matching: Matches known heading patterns
#    Examples: § symbols, "Chapter", "Article", ALL-CAPS text
#
# All three methods run independently. Results are deduplicated by text
# to avoid capturing the same heading twice with different methods.

import os
import hashlib
import pdfplumber
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================================
# CONFIG - Text Chunking Settings
# ============================================================================
# These settings are tuned for dense policy language where concepts must
# stay together to preserve meaning.
# 
# chunk_size=750: Large chunks preserve context for policy documents
# chunk_overlap=150: Ensures concepts split across boundaries aren't lost
# separators: Prefer paragraph/line breaks over arbitrary character splits

splitter = RecursiveCharacterTextSplitter(
    chunk_size=750,  # Characters per chunk
    chunk_overlap=150,  # Overlapping characters between chunks
    separators=["\n\n", "\n", ". ", " ", ""]  # Hierarchy: paragraph → line → sentence → word
)

# ============================================================================
# UTILITY FUNCTIONS - Path, Hash, Metadata
# ============================================================================

def get_full_path(relative_path):
    """
    Convert relative path to absolute path.
    
    Assumes paths are relative to this script's directory.
    Allows data_sources.json entries like "sources/file.pdf" to work
    regardless of where the script is run from.
    
    Args:
        relative_path: e.g., "sources/legislation.pdf"
    
    Returns:
        Absolute Path object
    """
    base_dir = Path(__file__).resolve().parent
    return (base_dir / relative_path).resolve()

def get_source_hash(input_string):
    """
    Generate unique MD5 hash for a source file path.
    
    Purpose: Deduplication at document level.
    - Same file path always produces same hash
    - Hash + chunk_index = globally unique chunk ID
    - Enables re-ingestion without duplicates (Pinecone upserts overwrite)
    
    Args:
        input_string: File path (e.g., "sources/legislation.pdf")
    
    Returns:
        32-char hex string (e.g., "abc123def456...")
    """
    return hashlib.md5(input_string.encode()).hexdigest()

def build_chunk_metadata(file_path, source_hash, chunk_index, chunk, meta, ingestion_date):
    """
    Construct rich metadata for a document chunk.
    
    This metadata enables powerful Pinecone filtering:
    
    FILTER EXAMPLES:
    - By organization: {"source_organization": "Sierra Club"}
    - By doc type: {"doc_type": "legislation"}
    - By section: {"section": "§ 17-330"}
    - By tags: {"tags": {"$in": ["California", "city-level"]}}
    - By date: {"publication_date": {"$gte": "2020-01-01"}}
    
    Args:
        file_path: Relative path from data_sources.json
        source_hash: MD5 hash of file_path (for deduplication)
        chunk_index: Position within document (0-indexed)
        chunk: LangChain Document object (contains page metadata)
        meta: Source metadata dict from data_sources.json
        ingestion_date: ISO date string (YYYY-MM-DD)
    
    Returns:
        Dictionary of metadata to attach to chunk in Pinecone
    """
    return {
        # SOURCE IDENTIFICATION (for attribution & filtering)
        "source_name": meta.get('name') or Path(file_path).stem,
        "source_url": meta.get("url"),
        "source_organization": meta.get("organization"),
        "source_hash": source_hash,
        
        # CHUNK POSITIONING (for context & reconstruction)
        "chunk_index": chunk_index,
        "page_number": chunk.metadata.get("page", 0) + 1,
        
        # DOCUMENT CLASSIFICATION (for semantic filtering)
        "doc_type": meta.get("doc_type"),
        "primary_focus": meta.get("primary_focus"),
        "tags": meta.get("tags", []),
        
        # TEMPORAL METADATA (for date-based filtering)
        "publication_date": meta.get("pub_date"),
        "ingestion_date": ingestion_date,
        
        # CHUNK IDENTIFICATION (for Pinecone consistency)
        "chunk_id": f"{source_hash}_{chunk_index}",
    }

# ============================================================================
# PDF SECTION EXTRACTION - THREE METHODS COMBINED
# ============================================================================

def extract_section_titles_by_font_size(file_path):
    """
    Extract section titles using ALL THREE methods combined:
    
    1. Font size detection: Large text (visual hierarchy)
    2. Bold detection: Entire line is bold (text emphasis)
    3. Pattern matching: § symbols, Chapter, ALL-CAPS
    
    All three methods run independently, results deduplicated by text.
    """
    
    if not Path(file_path).is_absolute():
        file_path = str(get_full_path(file_path))
    
    all_headings = []
    seen_headings = set()  # Deduplication set
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # METHOD 1: FONT SIZE DETECTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Identifies large text that indicates visual hierarchy.
    #
    # APPROACH:
    # 1. Scan entire PDF and collect all font sizes with frequencies
    # 2. Calculate median font size (body text is usually most common)
    # 3. Identify "heading sizes" = sizes larger than median + appear 2+ times
    # 4. Extract all text rendered at heading sizes
    # 5. Filter results (too short/long, duplicates)
    #
    # WHY IT WORKS:
    # - PDFs with clear visual hierarchy have distinct heading sizes
    # - Example: body=11pt, headings=16pt or 18pt
    # - Median-based approach avoids false positives from rare large characters
    
    try:
        with pdfplumber.open(file_path) as pdf:
            # Step 1: Collect all font sizes in document
            font_sizes = {}
            for page in pdf.pages:
                for char in page.chars:
                    size = round(char['size'], 1)  # Round to 0.1pt precision
                    font_sizes[size] = font_sizes.get(size, 0) + 1
            
            if font_sizes:
                # Step 2: Identify likely heading sizes
                sorted_sizes = sorted(font_sizes.items(), key=lambda x: x[1], reverse=True)
                median_size = sorted(font_sizes.keys())[len(font_sizes) // 2]
                
                # Step 3: Collect sizes that are larger than median AND appear 2+ times
                heading_sizes = set()
                for size, count in sorted_sizes:
                    if size > median_size and count >= 2:
                        heading_sizes.add(size)
                
                # Fallback: if no heading sizes identified, use top 3 largest sizes
                if not heading_sizes:
                    heading_sizes = set(size for size, _ in sorted_sizes[:3])
                
                # Step 4: Extract text at heading sizes (page by page)
                for page_num, page in enumerate(pdf.pages, 1):
                    current_line = ""
                    current_size = None
                    
                    # Process characters in reading order
                    for char in page.chars:
                        char_size = round(char['size'], 1)
                        char_text = char['text']
                        
                        # When we encounter a heading-size character
                        if char_size in heading_sizes:
                            # End of heading line (newline or size change)
                            if char_text == '\n' or current_size != char_size:
                                if current_line.strip():
                                    heading_text = current_line.strip()
                                    # Step 5: Filter results
                                    if (len(heading_text) > 3 and  # Not too short
                                        len(heading_text) < 200 and  # Not too long
                                        heading_text not in seen_headings):  # No duplicates
                                        all_headings.append({
                                            'text': heading_text,
                                            'page': page_num,
                                            'method': 'font_size'
                                        })
                                        seen_headings.add(heading_text)
                                current_line = ""
                                current_size = char_size
                            # Accumulate heading characters
                            if char_text != '\n':
                                current_line += char_text
                        else:
                            # Non-heading character encountered
                            # Wrap up any heading we were accumulating
                            if current_line.strip() and current_size in heading_sizes:
                                heading_text = current_line.strip()
                                if (len(heading_text) > 3 and 
                                    len(heading_text) < 200 and
                                    heading_text not in seen_headings):
                                    all_headings.append({
                                        'text': heading_text,
                                        'page': page_num,
                                        'method': 'font_size'
                                    })
                                    seen_headings.add(heading_text)
                            current_line = ""
                            current_size = None
    except Exception as e:
        pass
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # METHOD 2: BOLD DETECTION (Entire line must be bold)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Captures lines where EVERY non-whitespace character is bold.
    # This avoids picking up individual bold words while catching full headers.
    #
    # APPROACH:
    # 1. Process PDF line by line
    # 2. Track if each non-space character is bold
    # 3. When newline found, check if ENTIRE line was bold
    # 4. If yes, capture as heading
    #
    # EXAMPLE:
    # - "§ 17-330 Regulations." (all bold) → CAPTURED
    # - "ASPCA." (only one word, mixed with non-bold text) → SKIPPED
    # - "The ASPCA uses bold here" (mixed formatting) → SKIPPED
    #
    # WHY THIS WORKS:
    # - Section headers are typically emphasized with bold
    # - Individual bold words (labels) are not headers
    # - This distinction filters out false positives
    
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                current_line = ""
                line_is_all_bold = True  # Assume bold until proven otherwise
                non_space_count = 0
                
                for char in page.chars:
                    # Check if character is bold by examining fontname
                    # Examples: "Arial-Bold", "HelveticaBold", "THAISansBold"
                    is_bold = 'bold' in char.get('fontname', '').lower()
                    char_text = char['text']
                    
                    # End of line
                    if char_text == '\n':
                        # Check if ENTIRE line was bold
                        if current_line.strip() and line_is_all_bold and non_space_count > 0:
                            heading_text = current_line.strip()
                            if (len(heading_text) > 3 and
                                len(heading_text) < 200 and
                                heading_text not in seen_headings):
                                all_headings.append({
                                    'text': heading_text,
                                    'page': page_num,
                                    'method': 'bold'
                                })
                                seen_headings.add(heading_text)
                        
                        # Reset for next line
                        current_line = ""
                        line_is_all_bold = True
                        non_space_count = 0
                    else:
                        # Accumulate characters
                        current_line += char_text
                        
                        # Track if ANY non-space character is not bold
                        if char_text.strip():  # Non-whitespace
                            non_space_count += 1
                            if not is_bold:
                                line_is_all_bold = False  # Mark line as mixed
    except Exception as e:
        pass
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # METHOD 3: PATTERN MATCHING (Traditional headings + ALL-CAPS)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Matches common heading patterns in legal and research documents.
    # Works on any PDF with extractable text (including scanned with OCR).
    #
    # PATTERNS MATCHED:
    # - Legal sections: § symbol, "Chapter", "Article", "Subchapter", etc.
    # - Report sections: "Summary", "Introduction", "Findings", "Conclusion"
    # - ALL-CAPS titles: Text that is all uppercase (4-15 words, 5-100 chars)
    #
    # WHY IT WORKS:
    # - Legal documents consistently use § for sections
    # - Reports use specific keywords for section headers
    # - ALL-CAPS is common emphasis technique for titles
    # - Works even on scanned PDFs (OCR-extracted text)
    
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if not text:
                    continue
                
                lines = text.split('\n')
                
                for line in lines:
                    stripped = line.strip()
                    
                    # PATTERN 1: Traditional section headers
                    # Matches if line contains any of these keywords
                    if any(pattern in stripped for pattern in [
                        '§',  # Legal section symbol
                        'Chapter',  # Legal: "Chapter 5"
                        'Section',  # Legal: "Section 17-330"
                        'Article',  # Legal: "Article IV"
                        'Subchapter',  # Legal: "Subchapter 3"
                        'Part ',  # Legal: "Part A"
                        'Summary',  # Report: "Summary"
                        'Introduction',  # Report: "Introduction"
                        'Findings',  # Report: "Findings"
                        'Conclusion'  # Report: "Conclusion"
                    ]):
                        if len(stripped) < 150 and stripped not in seen_headings:
                            all_headings.append({
                                'text': stripped,
                                'page': page_num,
                                'method': 'pattern'
                            })
                            seen_headings.add(stripped)
                    
                    # PATTERN 2: ALL-CAPS titles
                    # Matches if text is all uppercase AND
                    # - 5-100 characters long (reasonable title length)
                    # - Less than 15 words (avoid full sentences)
                    elif (stripped.isupper() and 
                          len(stripped) > 5 and 
                          len(stripped) < 100 and
                          len(stripped.split()) < 15 and
                          stripped not in seen_headings):
                        all_headings.append({
                            'text': stripped,
                            'page': page_num,
                            'method': 'pattern'
                        })
                        seen_headings.add(stripped)
    except Exception as e:
        pass
    
    # Sort by page number (maintains reading order)
    all_headings.sort(key=lambda x: x['page'])
    
    # Print results for debugging
    print_headings(all_headings)

    return all_headings
        

def print_headings(headings):
    """
    Pretty-print all detected section titles.
    
    Shows: number, detection method, page, and text (truncated if long).
    This helps verify that section detection is working correctly.
    
    OUTPUT FORMAT:
    📋 Section titles:
        1. [font_size] p.  1 - Subchapter 3: Rental Horse Licensing...
        2. [bold    ] p.  1 - § 17-326 Definitions.
        3. [pattern ] p.  2 - § 17-327 License required.
    
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