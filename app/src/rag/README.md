# Perch RAG Ingestion Pipeline

This folder contains the document ingestion pipeline for Perch, the AI assistant for animal advocacy policy research. The pipeline supports 2 input types:

1. **PDFs** — Extract text, automatically detect section headers, chunk semantically
2. **Web Sources** — Fetch webpages, convert to markdown, chunk, and embed. Configurable to crawl a site up to N levels deep, ingest each discovered page separately

All documents are chunked with the `multilingual-e5-large` embedding model and stored in Pinecone with rich metadata for retrieval-augmented generation.

## Quick Start

### 1. Organize Your Documents

```
sources/
├── data_sources.json       ← Configuration file
├── pdfs/                   ← PDF files
│   ├── document1.pdf
│   └── document2.pdf
```

### 2. Create data_sources.json

In the `sources/` directory, create `data_sources.json` with entries for documents to ingest. See [Data Sources Format](#data-sources-format) below for detailed examples.

### 3. Run Ingestion

```bash
python ingest.py sources/data_sources.json
```

Or if running from the script directory:

```bash
python ingest.py path/to/sources/data_sources.json
```

The script will:

- Read `data_sources.json`
- Resolve PDF paths relative to the JSON file's directory
- Download/process each document (PDF, Web URLs)
- Chunk documents semantically (chunk_size=750, overlap=150)
- Embed chunks using multilingual-e5-large
- Upsert vectors to Pinecone with full metadata
- Show progress and final stats per namespace

---

## Data Sources Format

`data_sources.json` is a JSON array where each entry describes a single document to ingest. Three types are supported:

### Type 1: PDF File

```json
{
  "type": "pdf",                          // Required: document type
  "source": "pdfs/fur_sale_ban_2019.pdf", // Required: path relative to JSON file
  "namespace": "animal_policies",         // Optional: Pinecone namespace (default "animal_policies")
  "meta": {                               // Optional: See [Metadata Fields](#metadata-fields) below
    "name": "Proposed Int. No. 1476-A",   
    "url": "https://legistar.council.nyc.gov/...", 
    "primary_focus": "Farmed Animals",    
    "pub_date": "2025-09-25",             
    "doc_type": "legislation",            
    "tags": ["nyc", "fur", "farmed animals"] 
  }
}
```

### Type 2: Web sources

```json
{
  "type": "web",                         // Required
  "source": [                            // Required: full URL(s)
    "https://example.org/1",
    "https://example.org/2"
  ],
  "config": {                            // Optional web scraper/crawler options
    "container_selector": "article",     // CSS selector to isolate main content (default: main, article, or body)
    "is_crawl": true,                    // Must be true for the following scraper configs to apply. (default = False, for single URL ingestion)
                                         // Enables web crawling and ingesting discovered resources
    "skip_ingesting_seed": true,         // Skip ingesting seed page contents, e.g. if the source only provides a menu (default false)
    "max_depth": 1,                      // Crawl depth: 0=seed only, 1=seed+links, 2=2 levels (default 1 when crawling)
    "max_threads": 3                    // Concurrent fetchers. (default = 3 when crawling)
  },
  "namespace": "animal_policies",        // Optional (default "animal_policies")
  "meta": {                              // Optional metadata applied to all discovered pages
    "organization": "Faunalytics",
    "doc_type": "research",
    "tags": ["faunalytics"]
  }
}
```

**Important:** Each discovered URL is ingested **separately** with its own metadata. If crawling returns 35 URLs, you get 35 distinct documents in Pinecone, each with its own URL in the metadata.

---

## Metadata Fields

These fields can be included in the `meta` object for any document type:


| Field           | Type   | Description                                                                                 |
| --------------- | ------ | ------------------------------------------------------------------------------------------- |
| `name`          | string | Human-readable document title. If omitted, uses filename or page title.                     |
| `url`           | string | Link to original source for attribution. Auto-populated for URLs/scraped pages.             |
| `organization`  | string | Source organization. Used for filtering and attribution.                                    |
| `doc_type`      | string | Classification: `"legislation"`, `"report"`, `"case_study"`, `"guide"`, `"research"`, etc.  |
| `primary_focus` | string | Policy area: `"cage-free eggs"`, `"broiler welfare"`, `"fur ban"`, etc.                     |
| `pub_date`      | string | Publication date in ISO 8601 format (`YYYY-MM-DD`). Used for temporal filtering.            |
| `tags`          | array  | Multi-value tags for fine-grained filtering (e.g., `["California", "city-level", "2020"]`). |


**Example metadata:**

```json
{
  "name": "Chinese Consumers' Attitudes Toward Animal Welfare",
  "organization": "Faunalytics",
  "doc_type": "report",
  "primary_focus": "Consumer attitudes",
  "pub_date": "2023-06-15",
  "tags": ["China", "consumer research", "attitudes"]
}
```

---

## Section Title Extraction (PDFs Only)

For PDF documents, the script automatically detects section headers using three complementary methods:

1. **Font Size Detection** — Finds text larger than the median font size
2. **Bold Detection** — Identifies lines where entire text is bold
3. **Pattern Matching** — Matches legal symbols (`§`), keywords (`Chapter`, `Section`, `Article`), and ALL-CAPS titles

Each chunk is tagged with its parent section, enabling section-based filtering:

```
§ 17-330 Definitions         ← Detected section header
§ 17-326 License required    ← Another section
Article IV: Provisions       ← Section from pattern matching
General                      ← Unmapped content (intro/preamble)
```

This works best for structured documents (legislation, reports) and degrades gracefully for unstructured content (marked as `"General"`).

---

## Chunk Metadata

Each chunk carries rich metadata attached to the vector:

```json
{
  "source_name": "Proposed Int. No. 1476-A",
  "source_url": "https://legistar.council.nyc.gov/...",
  "source_organization": "NYC City Council",
  "source_hash": "abc123def456",
  "chunk_index": 0,
  "page_number": 1,
  "section": "§ 17-330 Definitions",
  "doc_type": "legislation",
  "primary_focus": "Farmed Animals",
  "tags": ["nyc", "fur"],
  "publication_date": "2025-09-25",
  "ingestion_date": "2026-03-27",
  "chunk_id": "abc123def456_0"
}
```

All metadata is queryable for filtering, attribution, and analytics.

---

## Pinecone Configuration

The script automatically creates the Pinecone index if it doesn't exist:

- **Index name**: `perch`
- **Embedding model**: `multilingual-e5-large` (1024 dimensions)
- **Distance metric**: Cosine similarity
- **Namespaces**: Organize documents by topic (e.g., `animal_policies`, `corporate_campaigns`)

### Checking Index Stats

```bash
python -c "
from pinecone import Pinecone
pc = Pinecone(api_key='your_api_key')
index = pc.Index('perch')
stats = index.describe_index_stats()
print(f\"Total vectors: {stats['total_vector_count']}\")
print(f\"Namespaces: {list(stats['namespaces'].keys())}\")
"
```

---

## Rate Limiting & Batching

The ingestion pipeline batches embeddings to respect Pinecone's rate limits:

- **Batch size**: 50 chunks per request
- **Batch delay**: 2 seconds between batches

To adjust these settings, edit the top of `ingest.py`:

```python
BATCH_SIZE = 50   # Chunks per batch
BATCH_DELAY = 2   # Seconds between batches
```

If you still hit 429 errors (rate limited), reduce `BATCH_SIZE` to 25 or 10, and increase `BATCH_DELAY` to 5 or 10.

---

## Running Ingestion

### Basic Ingestion

```bash
# From anywhere, use full or relative path to JSON file
python ingest.py sources/data_sources.json

# Or with full path
python ingest.py /home/user/perch/sources/data_sources.json
```

The script will resolve all PDF paths relative to the JSON file's directory.

### Example Directory Structure

```
project/
├── ingest.py
├── chunking_utils.py
└── sources/
    ├── data_sources.json
    ├── pdfs/
    │   ├── document1.pdf
    │   └── document2.pdf
    └── ...other data files
```

Run from project root:

```bash
python ingest.py sources/data_sources.json
```
<details>
<summary>Sample output:</summary>

```
======================================================================
🚀 Starting Batch Ingestion to Pinecone from ./data_sources.json
======================================================================
✅ Using existing index: perch

──────────────────────────────────────────────────────────────────────
[PDF] Proposed Int. No. 1476-A
──────────────────────────────────────────────────────────────────────
📖 Loading PDF...
✅ Loaded 5 pages
🔍 Extracting section headers...
✅ Found 3 sections
✂️  Splitting into chunks...
✅ Created 12 chunks
🏷️  Adding metadata...
🚀 Embedding and upserting...
   ✅ Batch 1/1 (12 chunks) upserted
✅ Successfully ingested 12 chunks into 'animal_policies'
📊 Namespace 'animal_policies' now has 1524 vectors


🕸️  [WEB] Processing: https://www.betterfoodfoundation.org/initiatives/plant-based-ngo-network/
   ✅ [MARKED TO INGEST] https://www.betterfoodfoundation.org/initiatives/plant-based-ngo-network (3775 chars)
   📦 Upserting 7 NEW chunks from 'Plant-Based NGO Network | Better Food Foundation' in 1 batches...
   ✅ Batch 1/1 upserted
📊 Namespace 'animal_policies' now has 10459 vectors
```
</details>
---

## Chunking Strategy

Documents are split using `RecursiveCharacterTextSplitter` with:

- **Chunk size**: 750 characters
- **Overlap**: 150 characters
- **Priority separators**: Paragraph breaks → Line breaks → Sentences → Words

This is tuned for dense policy language where concepts must stay together. For other content types, edit `chunking_utils.py`:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

---

## Document Deduplication

Documents are deduplicated at the chunk level using a hash of the source (file path or URL):

- `source_hash = MD5(file_path or URL)`
- `chunk_id = f"{source_hash}_{chunk_index}"`

~~Re-running ingestion with the same file/URL will **upsert** (overwrite) existing chunks. No duplicates are created.~~
Re-running ingestion with the same file/URL will **skip** upsertion in order to save Pinecone resources. TODO: add --force flag to upsert

To fully remove a document:

1. Delete its entry from `data_sources.json`
2. Manually delete its vectors from Pinecone:
  ```bash
   python -c "
   from pinecone import Pinecone
   pc = Pinecone(api_key='your_api_key')
   index = pc.Index('perch')
   # Delete all chunks with source_hash=abc123
   index.delete(filter={'source_hash': 'abc123'}, namespace='animal_policies')
   "
  ```

---

## Troubleshooting

### Ingestion fails with 403 or 404

- PDFs: Check file path exists relative to script directory
- URLs: Website is blocking automated access; try a different URL
- Web crawl: Some pages may 403; the script continues with others

### Retrieval quality is low

- Increase `chunk_size` in `chunking_utils.py` if policy concepts are split
- Add more specific `tags` and `primary_focus` for metadata filtering
- Verify `pub_date` is accurate for temporal filtering
- Check that documents were actually ingested: `index.describe_index_stats()`

### Missing section headers in PDF chunks

- PDF likely has unusual formatting; check that `extract_section_titles_by_font_size()` output has headers
- Mark sections as `"General"` if extraction fails; re-parse manually if needed

### Rate limiting (429 errors)

- Reduce `BATCH_SIZE` (try 25 or 10)
- Increase `BATCH_DELAY` (try 5 or 10 seconds)
- Edit these constants at the top of `ingest.py`

---

## Integration with Query Pipeline

Once documents are ingested, the query pipeline (`query.py`) automatically:

1. Embeds user queries using `multilingual-e5-large` (same model)
2. Searches Pinecone for most-similar chunks
3. Returns top-k results with full metadata for attribution
4. Passes chunks to LLM for context in RAG


<details>
<summary>Example retrieval with metadata filtering:</summary>

```python
from langchain_pinecone import PineconeVectorStore

vectorstore = PineconeVectorStore(
    index_name="perch",
    embedding=embeddings,
    namespace="animal_policies"
)

# Simple similarity search
docs = vectorstore.similarity_search("broiler welfare", k=5)

# With metadata filtering (org)
docs = vectorstore.similarity_search(
    "broiler welfare",
    k=5,
    filter={"source_organization": "Faunalytics"}
)

# With date range filtering
docs = vectorstore.similarity_search(
    "broiler welfare",
    k=5,
    filter={"publication_date": {"$gte": "2020-01-01"}}
)

# With tag filtering (multi-value)
docs = vectorstore.similarity_search(
    "broiler welfare",
    k=5,
    filter={"tags": {"$in": ["USA", "legislation"]}}
)
```
</details>


---

## Performance Notes

- **Ingestion speed**: ~10 chunks/second (depends on network and chunk size)
- **Web scraping**: Scales with `max_threads` and `max_depth`. Depth=2 can return 100+ URLs.
- **Storage**: Pinecone charges per million vectors. A 100-page PDF (~300 chunks) uses minimal capacity.
- **Query latency**: Metadata filtering reduces retrieval time by narrowing candidates before similarity search.

---

## Future Improvements

- Support for DOCX, TXT, Markdown files
- Automatic table and figure extraction from PDFs
- Metadata validation and schema enforcement
- Change tracking for re-ingested documents
- Bulk operations for faster ingestion of large document sets

---

## Contributing

When adding new documents:

- Use consistent `doc_type` values (keep it simple: legislation, report, case_study, guide, research)
- Add descriptive `tags` for cross-filtering and discovery
- For PDFs, ensure files are machine-readable (not scanned images without OCR)
- Set `pub_date` in ISO format (`YYYY-MM-DD`)

---

## License

Internal use only for Perch animal advocacy platform.