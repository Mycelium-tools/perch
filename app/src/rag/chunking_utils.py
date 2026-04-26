# chunking_utils.py
#
# Utilities for chunking documents to be ingested into Pinecone.
# Provides: text splitting, metadata building, section extraction
#
# PDF SECTION EXTRACTION STRATEGY (3-method combined approach):
# ────────────────────────────────────────────────────────────────────────
# 1. Font size detection: Identifies large text (visual hierarchy)
#
# 2. Bold detection: Captures lines where ENTIRE text is bold. 
#    Skips lines with mixed formatting
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
from taxonomies import ChunkMetadata

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
# UTILITY FUNCTIONS - Build Chunk Metadata
# ============================================================================
def build_chunk_metadata_validated(file_path_or_url, chunk_index, chunk, meta, section_title = ""):
    """
    Creates a validated metadata object using the Pydantic ChunkMetadata schema.
    """
    # 1. Generate the hash for deduplication
    source_content = chunk.page_content.encode('utf-8')
    source_hash = hashlib.sha256(source_content).hexdigest()
    
    # 2. Map data_sources.json 'meta' to our Schema
    # This assumes your data_sources.json entries match your Enum names
    validated_meta = ChunkMetadata(
        source_name=meta.get("name", "Unknown"),
        source_organization=meta.get("organization", "N/A"),
        primary_focus=meta.get("primary_focus"), # Validates against Enum
        doc_type=meta.get("doc_type"),           # Validates against Enum
        source_url=file_path_or_url,
        section=section_title,
        source_hash=source_hash,
        chunk_index=chunk_index,
        chunk_id=f"{source_hash}_{chunk_index}",
        raw_date=meta.get("publication_date", "1970-01-01"),
        tags=meta.get("tags", []) # Triggers auto-normalization to snake_case
    )
    
    # Return as a Pinecone-compatible dictionary
    return validated_meta.dict()


def build_chunk_metadata(file_path_or_url, chunk_index, chunk, meta, ingestion_date):
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
        file_path_or_url: Relative path from data_sources.json or URL for web/scrape ingestion
        chunk_index: Position within document (0-indexed)
        chunk: LangChain Document object (contains page metadata)
        meta: Source metadata dict from data_sources.json
        ingestion_date: ISO date string (YYYY-MM-DD)
    
    Returns:
        Dictionary of metadata to attach to chunk in Pinecone
    """
    source_hash = get_source_hash(file_path_or_url)
    return {
        # SOURCE IDENTIFICATION (for attribution & filtering)
        "source_name": meta.get('name') or Path(file_path_or_url).stem,
        "source_url": meta.get("url"),
        "source_organization": meta.get("organization", ""),
        # Used for deduplication in Pinecone
        "source_hash": source_hash, 
        
        # CHUNK POSITIONING (for context & reconstruction)
        "chunk_index": chunk_index,
        "page_number": chunk.metadata.get("page", 0) + 1,
        
        # DOCUMENT CLASSIFICATION (for semantic filtering)
        "doc_type": meta.get("doc_type", ""),
        "primary_focus": meta.get("primary_focus", ""),
        "tags": meta.get("tags", []),
        
        # TEMPORAL METADATA (for date-based filtering)
        "publication_date": meta.get("pub_date", ""),
        "ingestion_date": ingestion_date,
        
        # CHUNK IDENTIFICATION (for Pinecone consistency)
        "chunk_id": f"{source_hash}_{chunk_index}",
    }




# ============================================================================
# UTILITY FUNCTIONS - File Path, File Hash
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