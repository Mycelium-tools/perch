# batchIngest.py
#
# Batch document ingestion script for Perch's RAG pipeline.
#
# Reads document metadata from data_sources.json and ingests all PDFs into
# Pinecone with rich metadata (organization, doc_type, tags, sections, etc.).
#
# For ingesting a single document, see ingest.py.
#
# Usage:
#   python batchIngest.py

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Import utility functions and main ingest function from ingest_utils
from ingest_utils import *

from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# ============================================================================
# PINECONE CONFIGURATION
# ============================================================================

# Default namespace to ingest to
DEFAULT_NAMESPACE = 'animal_policies'

# Embedding model
model_name = 'multilingual-e5-large'

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
if index_name not in pc.list_indexes().names():
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

# ============================================================================
# BATCH INGESTION FUNCTIONS
# ============================================================================

def ingest_pdf(entry):
    """
    Ingest a single PDF from a data_sources.json entry
    
    Args:
        entry: Dict from data_sources.json with keys:
               - source: relative path to PDF
               - namespace: Pinecone namespace (default: "animal_policies")
               - meta: metadata dict (name, url, organization, doc_type, tags, pub_date)
    """
    file_path = entry.get('source')
    full_path = get_full_path(file_path).resolve()
    namespace = entry.get('namespace', DEFAULT_NAMESPACE)
    meta = entry.get('meta', {})
    
    # Validate file exists
    if not file_path or not full_path.exists():
        print(f"⚠️  File not found or path missing: {full_path}")
        return
    
    display_name = meta.get('name') or full_path.stem
    print(f"\n{'─'*70}")
    print(f"Processing: {display_name}")
    print(f"{'─'*70}")
    
    try:
        # ─────────────────────────────────────────────────────────────────
        # STEP 1: Load PDF
        # ─────────────────────────────────────────────────────────────────
        print(f"📖 Loading PDF...")
        loader = PyPDFLoader(str(full_path))
        docs = loader.load()
        print(f"✅ Loaded {len(docs)} pages")

        # ─────────────────────────────────────────────────────────────────
        # STEP 2: Generate identifiers and timestamps
        # ─────────────────────────────────────────────────────────────────
        source_hash = get_source_hash(file_path)
        ingestion_date = datetime.now().strftime("%Y-%m-%d")

        # ─────────────────────────────────────────────────────────────────
        # STEP 3: Extract section headers
        # ─────────────────────────────────────────────────────────────────
        print(f"🔍 Extracting section headers...")
        headings = extract_section_titles_by_font_size(file_path)
        print(f"✅ Found {len(headings)} sections")

        # ─────────────────────────────────────────────────────────────────
        # STEP 4: Split into chunks
        # ─────────────────────────────────────────────────────────────────
        print(f"✂️  Splitting into chunks...")
        chunks = splitter.split_documents(docs)
        print(f"✅ Created {len(chunks)} chunks")
        
        # ─────────────────────────────────────────────────────────────────
        # STEP 5: Add metadata to each chunk
        # ─────────────────────────────────────────────────────────────────
        print(f"🏷️  Adding metadata...")
        for i, chunk in enumerate(chunks):
            # Determine section for this chunk
            current_page = chunk.metadata.get('page', 0)
            section = next(
                (h['text'] for h in reversed(headings) if h['page'] <= current_page),
                "General"
            )
            
            # Build metadata
            metadata = build_chunk_metadata(
                file_path=file_path,
                source_hash=source_hash,
                chunk_index=i,
                chunk=chunk,
                meta=meta,
                ingestion_date=ingestion_date
            )
            metadata["section"] = section
            
            # Attach metadata to chunk
            chunk.metadata.update(metadata)
            chunk.id = metadata["chunk_id"]

        # ─────────────────────────────────────────────────────────────────
        # STEP 6: Embed and upsert to Pinecone
        # ─────────────────────────────────────────────────────────────────
        print(f"🚀 Embedding and upserting...")
        PineconeVectorStore.from_documents(
            chunks,
            index_name=index_name,
            embedding=embeddings,
            namespace=namespace
        )
        print(f"✅ Successfully ingested {len(chunks)} chunks into '{namespace}'")
        
        # Print stats
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        ns_stats = stats.get('namespaces', {}).get(namespace, {})
        vector_count = ns_stats.get('vector_count', 0)
        print(f"📊 Namespace '{namespace}' now has {vector_count} vectors")
        
    except Exception as e:
        print(f"❌ Error processing {full_path}: {e}")
        import traceback
        traceback.print_exc()


def run_ingestion_from_json(json_file_path):
    """
    Read data_sources.json and ingest all PDF entries.
    
    Args:
        json_file_path: Path to data_sources.json
    """
    if not os.path.exists(json_file_path):
        print(f"❌ JSON file not found at: {json_file_path}")
        return

    with open(json_file_path, 'r', encoding='utf-8') as f:
        try:
            data_sources = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON: {e}")
            return

    # Process only PDF entries
    pdf_count = 0
    for entry in data_sources:
        if entry.get('type') == 'pdf':
            ingest_pdf(entry)
            pdf_count += 1
        else:
            print(f"🚫 Skipping non-PDF entry: {entry.get('source', 'Unknown')}")
    
    print(f"\n{'='*70}")
    print(f"✅ Batch ingestion complete! Processed {pdf_count} PDFs")
    print(f"{'='*70}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    rag_dir = current_file.parent
    JSON_INPUT_FILE = rag_dir / "data_sources.json"
    
    print(f"{'='*70}")
    print(f"🚀 Starting Resource Ingestion to Pinecone from {JSON_INPUT_FILE}")
    print(f"{'='*70}")
    
    run_ingestion_from_json(JSON_INPUT_FILE)