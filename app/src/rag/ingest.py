# ingest.py
#
# Document ingestion script for Perch's RAG pipeline.
#
# Supports PDF files and URLs:
# - PDFs: Extracted, chunked, and embedded locally
# - URLs: Fetched, converted to markdown, chunked, and embedded
#
# Reads document metadata from data_sources.json and ingests into Pinecone
# with rich metadata (organization, doc_type, tags, sections, etc.).
#
# Usage:
#   python ingest.py

import os
import json
import hashlib
import time
import sys
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor

# Local utility imports
from chunking_utils import splitter, build_chunk_metadata_validated
from parsing_utils import parse_pdf_with_sections
from scraper import WebScraper

# Load environment variables
load_dotenv()

# ============================================================================
# PINECONE CONFIGURATION
# ============================================================================

# Default namespace to ingest to
DEFAULT_NAMESPACE = 'animal_policies'

# Embedding model
model_name = 'multilingual-e5-large'

# Rate limiting: batch size and delay (in seconds)
BATCH_SIZE = 50  # Chunks per batch
BATCH_DELAY = 2  # Seconds between batches

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Serverless spec
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

# Target Pinecone index
index_name = "perch"

# Initialize embeddings
embeddings = PineconeEmbeddings(
    model=model_name,
    pinecone_api_key=os.environ.get('PINECONE_API_KEY')
)

# Ensure index exists in Pinecone
if index_name not in [idx.name for idx in pc.list_indexes()]:
    print(f"Creating index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=1024,  # Dimension for multilingual-e5-large
        metric="cosine",
        spec=spec
    )
    print(f"✅ Index created: {index_name}")
else:
    print(f"✅ Using existing index: {index_name}")
idx = pc.Index(index_name)

# ============================================================================
# HELPER FUNCTIONS - Rate-Limited Upsertion
# ============================================================================
def fetch_metadata_batched(idx, candidate_ids, namespace, batch_size=30):
    """
    Fetches metadata in small batches to avoid 414 Request-URI Too Large errors.
    """
    all_fetched = {}
    
    # Break the long list of IDs into small batches (e.g., 30 at a time)
    for i in range(0, len(candidate_ids), batch_size):
        batch = candidate_ids[i : i + batch_size]
        try:
            fetch_response = idx.fetch(ids=batch, namespace=namespace)
            all_fetched.update(fetch_response.vectors)
        except Exception as e:
            if "414" in str(e):
                print(f"⚠️ Batch size {batch_size} still too large, retrying smaller...")
                # Recursive fallback if IDs are exceptionally long
                return fetch_metadata_safely(idx, candidate_ids, namespace, batch_size=10)
            raise e
            
    return all_fetched

def upsert_chunks_batched(chunks, index_name, embedding, namespace):
    """
    Upsert chunks to Pinecone in batches to respect rate limits.
    NOTE: This skips upserting chunks that already exist in Pinecone

    Avoids 429 "Too Many Requests" errors by:
    - Splitting chunks into BATCH_SIZE groups
    - Upserting each batch separately
    - Adding BATCH_DELAY seconds between batches
    """
    total_chunks = len(chunks)
    if total_chunks == 0:
        return

    # 1. Collect all candidate IDs in this batch
    candidate_ids = [c.id for c in chunks]

    # 2. Check which IDs already exist in Pinecone
    fetch_response = fetch_metadata_batched(idx, candidate_ids,namespace)

    existing_ids = set(fetch_response.keys())
    
    # 3. Filter to only "new" chunks
    new_chunks = [c for c in chunks if c.id not in existing_ids]
    
    # Metadata for logging
    first_chunk_meta = chunks[0].metadata
    title = first_chunk_meta.get('title', 'N/A')
    source_val = first_chunk_meta.get('source', 'NA')

    snippet = chunks[0].page_content[:100]
    print(f"   📄 Snippet: {snippet}...")
    if not new_chunks:
        print(f"   ⏩ [SKIPPING INGESTION]: All {len(chunks)} chunks in this batch already exist for '{title}'.")    
        return

    # 4. Proceed with embedding and upserting only the new_chunks
    total_new = len(new_chunks)
    batches = [new_chunks[i:i + BATCH_SIZE] for i in range(0, total_new, BATCH_SIZE)]
        
    print(f"   📦 Upserting {total_new} NEW chunks from '{title}' in {len(batches)} batches...")
    
    for batch_num, batch in enumerate(batches, 1):
        try:
            PineconeVectorStore.from_documents(
                batch,
                index_name=index_name,
                embedding=embedding,
                namespace=namespace
            )
            print(f"   ✅ Batch {batch_num}/{len(batches)} upserted")
            
            # Delay before next batch
            if batch_num < len(batches):
                time.sleep(BATCH_DELAY)
        except Exception as e:
            print(f"   ❌ Batch {batch_num} failed: {e}")
            raise

     # Print stats
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    ns_stats = stats.get('namespaces', {}).get(namespace, {})
    vector_count = ns_stats.get('vector_count', 0)
    print(f"📊 Namespace '{namespace}' now has {vector_count} vectors")

# ============================================================================
# INGESTION FUNCTIONS - PDF
# ============================================================================

def ingest_pdf(entry, json_dir=None):
    """
    Ingest a single PDF from a data_sources.json entry.
    """
    file_path = entry.get('source')
    
    if json_dir:
        full_path = (json_dir / file_path).resolve()
    else:
        full_path = get_full_path(file_path).resolve()
    
    namespace = entry.get('namespace', DEFAULT_NAMESPACE)
    meta = entry.get('meta', {})
    
    if not file_path or not full_path.exists():
        print(f"⚠️  File not found: {full_path}")
        return
    
    display_name = meta.get('name') or full_path.stem
    print(f"\n{'─'*70}\n[PDF] {display_name}\n{'─'*70}")
    
    try:
        # STEP 1: Load PDF
        print(f"📖 Loading PDF...")
        loader = PyPDFLoader(str(full_path))
        docs = loader.load()
        print(f"✅ Loaded {len(docs)} pages")

        # STEP 2: Extract section headers
        print(f"🔍 Extracting section headers...")
        headings = parse_pdf_with_sections(str(full_path))
        print(f"✅ Found {len(headings)} sections")

        # STEP 3: Split into chunks
        print(f"✂️  Splitting into chunks...")
        chunks = splitter.split_documents(docs)
        print(f"✅ Created {len(chunks)} chunks")
        
        # STEP 4: Add metadata to each chunk
        print(f"🏷️  Adding metadata...")
        for i, chunk in enumerate(chunks):
            current_page = chunk.metadata.get('page', 0)
            section_title = next(
                (h['text'] for h in reversed(headings) if h['page'] <= current_page),
                "General"
            )
            
            metadata = build_chunk_metadata_validated(
                file_path_or_url=str(file_path),
                chunk_index=i,
                chunk=chunk,
                meta=meta,
                section_title=section_title
            )
            chunk.metadata.update(metadata)
            chunk.id = metadata["chunk_id"]

        # STEP 5: Embed and upsert to Pinecone
        upsert_chunks_batched(chunks, index_name, embeddings, namespace)
        
    except Exception as e:
        print(f"❌ Error processing {full_path}: {e}")

# ============================================================================
# INGESTION FUNCTIONS - WEB (single URL or web crawl + scrape)
# ============================================================================

def ingest_web(entry):
    config = entry.get('config', {})
    namespace = entry.get('namespace', DEFAULT_NAMESPACE)

    # Crawler configs
    is_crawl = config.get('is_crawl', False)
    skip_ingesting_seed = config.get('skip_ingesting_seed', False)
    max_depth = config.get('max_depth', 1) if is_crawl else 0
    container_selector = config.get('container_selector', None)

    # Instantiate a single WebScraper to ingest all urls in the entry 
    default_threads = 3 if is_crawl else 1
    max_threads = config.get('max_threads', default_threads)
    scraper = WebScraper(max_threads=max_threads)

    sources = entry.get('source', [])
    urls_to_process = sources if isinstance(sources, list) else [sources]
    
    total_ingested = 0

    try:
        for url in urls_to_process:
            # STEP 1: Scrape web from source URLs, and collect additional URLs to ingest if crawling
            print(f"\n🕸️  [WEB] Processing: {url}")
            pages = scraper.crawl_and_scrape(
                url, 
                max_depth=max_depth, # When depth = 0, only scrapes the current URL (no crawling)
                skip_ingesting_seed=skip_ingesting_seed, 
                container_selector=container_selector
            )
            
            # STEP 2: Convert each webpage to Markdown
            for page in pages:
                current_url = page.get('url')
                markdown_text = page['markdown']
                page_title = page.get('title', 'Untitled')
                
                if not markdown_text:
                    continue

                # STEP 3: Create Document object
                doc = Document(
                    page_content=markdown_text,
                    metadata={
                        "source": current_url, 
                        "title": page_title.strip() if page_title else ""
                    }
                )
                
                # STEP 4: Split into chunks
                chunks = splitter.split_documents([doc])           
                for i, chunk in enumerate(chunks):
                    chunk_meta = build_chunk_metadata_validated(
                        file_path_or_url=current_url,
                        chunk_index=i,
                        chunk=chunk,
                        meta={**entry.get('meta', {}), "url": current_url}
                    )
                    chunk.metadata.update(chunk_meta)
                    chunk.id = chunk_meta["chunk_id"]

                # STEP 5: Embed and upsert to Pinecone
                upsert_chunks_batched(chunks, index_name, embeddings, namespace)
                total_ingested += 1
    finally:
        scraper.close()
        return total_ingested


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def run_ingestion_from_json(json_path_str):
    """
    Read data_sources.json and ingest all entries (PDF, URL, and web scrape).
    
    Processes entries in order:
    - type: "pdf" → calls ingest_pdf()
    - type: "web" → calls ingest_web()
    - Other → skipped with warning
    
    Relative paths in "source" fields are resolved relative to the JSON file's directory.
    
    Args:
        json_file_path: Path to data_sources.json (absolute or relative)
    """
    json_path = Path(json_path_str).resolve()
    
    if not json_path.exists():
        print(f"❌ JSON file not found at: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data_sources = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON: {e}")
            return

    # Store JSON directory for resolving relative paths
    json_dir = json_path.parent

    # Process all entries
    pdf_count = 0
    url_count = 0
    
    print(f"🚀 Starting ingestion for {len(data_sources)} source definitions...")
    for entry in data_sources:
        entry_type = entry.get('type')
        try:
            if entry_type == 'pdf':
                ingest_pdf(entry, json_dir)
                pdf_count += 1
            elif entry_type == 'web':
                url_count += ingest_web(entry)
            else:
                print(f"🚫 Skipping unknown type '{entry_type}': {entry.get('source', entry.get('seed', 'Unknown'))}")
        except Exception as e:
            print(f"❌ Error processing entry: {e}")
    
    print(f"\n{'='*70}")
    print(f"✅ Batch ingestion complete!")
    print(f"   PDFs processed: {pdf_count}")
    print(f"   Web sources processed: {url_count}")
    print(f"   Total: {pdf_count + url_count} sources")
    print(f"{'='*70}")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Get JSON file from command-line argument or use default
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # Default to data_sources.json in script directory
        current_file = Path(__file__).resolve()
        rag_dir = current_file.parent
        json_file = str(rag_dir / "data_sources.json")
    
    # Resolve to absolute path
    json_path = Path(json_file).resolve()
    
    print(f"{'='*70}")
    print(f"🚀 Starting Batch Ingestion to Pinecone from {json_path}")
    print(f"{'='*70}")
    
    run_ingestion_from_json(str(json_path))
