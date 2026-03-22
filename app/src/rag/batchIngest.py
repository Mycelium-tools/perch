import os
import json
import hashlib
import pdfplumber
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone client and config
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

# Pinecone index to deposit resource embeddings into
index_name = "perch"
model_name = 'multilingual-e5-large'

embeddings = PineconeEmbeddings(
    model=model_name,
    pinecone_api_key=os.environ.get('PINECONE_API_KEY')
)

# Ensure index exists in Pinecone
if index_name not in pc.list_indexes().names():
    print(f"Creating index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=1024, # Dimension for multilingual-e5-large
        metric="cosine",
        spec=spec
    )

# Attempt to chunk PDF based on formatting/section indicators
splitter = RecursiveCharacterTextSplitter(
    chunk_size=750, # Dense policy language requires larger chunks to preserve meaning
    chunk_overlap=150, # Ensures concepts split across boundaries aren't lost
    separators=["\n\n", "\n", ". ", " ", ""] # Prefers paragraph/line boundaries over arbitrary splits
)

def extract_section_titles_by_font(file):
    """Extract section titles by identifying largest font sizes."""
    file_path = get_full_path(file)
    try:
        with pdfplumber.open(file_path) as pdf:
            headings = []
            
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if not text:
                    continue
                
                lines = text.split('\n')
                
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    
                    # Match common heading patterns
                    if any(pattern in stripped for pattern in [
                        '§',           # Legal section
                        'Chapter',
                        'Section',
                        'Article',
                        'Subchapter',
                        'Part ',
                    ]):
                        if len(stripped) < 150:  # Reasonable heading length
                            headings.append({
                                'text': stripped,
                                'page': page_num
                            })
                    
                    # Match all-caps lines (often titles)
                    elif (stripped.isupper() and 
                          len(stripped) > 5 and 
                          len(stripped) < 100 and
                          len(stripped.split()) < 15):
                        headings.append({
                            'text': stripped,
                            'page': page_num
                        })
            
            return headings
    
    except Exception as e:
        print(f"Could not extract structure from {file_path}: {e}")
        return []

def get_source_hash(input_string):
    """Generates a unique MD5 hash for the source path."""
    return hashlib.md5(input_string.encode()).hexdigest()

def build_chunk_metadata(file_path, source_hash, chunk_index, chunk, meta, ingestion_date):
    """
    Constructs metadata for a single document chunk.
    
    Metadata fields can be used in Pinecone for:
    - **Filtering**: Use in hybrid search to restrict results by organization, doc_type, or date range
      Example: retriever.as_retriever(search_kwargs={"filter": {"source_organization": "Faunalytics"}})
    - **Display**: Return to frontend for source attribution and context
    - **Deduplication**: source_hash + chunk_index uniquely identifies each chunk
    - **Analytics**: Track ingestion dates, document types, and coverage across namespaces
    See https://docs.pinecone.io/guides/assistant/files-overview for more infor
    
    Args:
        file_path: Original file path (used for hash)
        source_hash: MD5 hash of file_path for deduplication
        chunk_index: Position of this chunk in the document
        chunk: LangChain Document object with page metadata
        meta: Source metadata from JSON (name, url, organization, etc.)
        ingestion_date: ISO date string of ingestion time
    
    Returns:
        Dictionary of metadata to attach to the chunk
    """
    return {
        # Source identification
        "source_name": meta.get('name') or Path(file_path).stem,  # Display name for UI
        "source_url": meta.get("url"),  # Link back to original source
        "source_organization": meta.get("organization"),  # Filter by org: {"source_organization": "Faunalytics"}
        "source_hash": source_hash,  # Deduplication: uniquely identifies this PDF/URL
        
        # Chunk positioning
        "chunk_index": chunk_index,  # Sequence within document
        "page_number": chunk.metadata.get("page", 0) + 1,  # Human-readable page ref

        # Document classification
        "doc_type": meta.get("doc_type"),  # Filter by type: {"doc_type": "legislation"} vs "report"
        "primary_focus": meta.get("primary_focus"),  # Topic tag: "cage-free eggs", "fur ban", etc.
        "tags": meta.get("tags", []),  # Multi-value filter: filter for multiple tags at once
        
        # Temporal metadata
        "publication_date": meta.get("pub_date"),  # Filter for recent policies: {"publication_date": {"$gte": "2023-01-01"}}
        "ingestion_date": ingestion_date,  # Track when document was added to index
        
        # Chunk identification (for Pinecone consistency)
        "chunk_id": f"{source_hash}_{chunk_index}",
    }


def extract_section_title(content):
    """Extract first meaningful line as section reference."""
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    # If first line looks like a section/definition header, use it
    first = lines[0] if lines else "General"
    if any(marker in first for marker in ['§', 'Subchapter', 'Chapter', 'Section', 'Article']):
        return first
    
    return "General"

def ingest_pdf(entry):
    """
    Ingests a single PDF using the structured JSON entry format.
    """
    file_path = entry.get('source')
    full_path = get_full_path(file_path).resolve()
    namespace = entry.get('namespace', 'animal_policies')
    meta = entry.get('meta', {})
    
    if not file_path or not full_path.exists():
        print(f"⚠️ File not found or path missing: {full_path}")
        return

    # Fallback logic: use file name w/o suffix if meta['name'] is missing or empty
    display_name = meta.get('name') or full_path.stem
    print(f"Processing: {display_name}")
    
    try:
        # Convert Path object to string for the loader
        loader = PyPDFLoader(str(full_path))
        docs = loader.load()

        source_hash = get_source_hash(file_path)
        ingestion_date = datetime.now().strftime("%Y-%m-%d")
        
        # Extract headings once per document
        headings = extract_section_titles_by_font(file_path)
        
        chunks = splitter.split_documents(docs)        
        for i, chunk in enumerate(chunks):
            # Find the most recent heading before this chunk
            current_page = chunk.metadata.get('page', 0)
            section = next(
                (h['text'] for h in reversed(headings) if h['page'] <= current_page),
                "General"
            )
            # For debugging purposes only:
            # print(section)
            
            # Build and apply metadata
            metadata = build_chunk_metadata(
                file_path=file_path,
                source_hash=source_hash,
                chunk_index=i,
                chunk=chunk,
                meta=meta,
                ingestion_date=ingestion_date
            )
            metadata["section"] = section  # Add section context
            chunk.metadata.update(metadata)
            chunk.id = metadata["chunk_id"]

        # Embed chunks and upsert to Pinecone
        # PineconeVectorStore.from_documents embeds each chunk and upserts the
        # resulting vectors in a single call. If the namespace does not exist,
        # Pinecone creates it automatically.
        PineconeVectorStore.from_documents(
            chunks,
            index_name=index_name,
            embedding=embeddings,
            namespace=namespace
        )
        print(f"✅ Successfully ingested {len(chunks)} chunks into '{namespace}'.")
        # This shows counts per namespace
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(stats)
        
    except Exception as e:
        print(f"❌ Error processing {full_path}: {e}")

def get_full_path(relative_path):
    # base_dir is where this script is, assuming 'sources/' is relative to it
    base_dir = Path(__file__).resolve().parent
    full_path = (base_dir / relative_path).resolve()
    return full_path
    
def run_ingestion_from_json(json_file_path):
    """
    Reads the structured JSON file and triggers ingestion for each PDF entry.
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

    # Strictly process only PDF types from the JSON
    for entry in data_sources:
        if entry.get('type') == 'pdf':
            ingest_pdf(entry)
        else:
            print(f"🚫 Skipping non-PDF entry: {entry.get('source', 'Unknown source')}")

if __name__ == "__main__":
    # Specify the path to your external JSON file here
    current_file = Path(__file__).resolve()
    rag_dir = current_file.parent
    JSON_INPUT_FILE = rag_dir / "data_sources.json" 
    
    print(f"--- Starting Bulk Ingestion from {JSON_INPUT_FILE} ---")
    run_ingestion_from_json(JSON_INPUT_FILE)
    print("--- Ingestion Process Complete ---")