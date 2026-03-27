# Perch RAG Ingestion Pipeline

This folder contains the document ingestion pipeline for Perch, the AI assistant for animal advocacy policy research. The pipeline extracts text from PDF documents, chunks them semantically, embeds them with multilingual-e5-large, and stores them in Pinecone for retrieval-augmented generation.

## Quick Start
```
### 1. Prepare Documents

Place PDF files in the `sources/` directory and add entries to `data_sources.json` (see [Data Sources Format](#data-sources-format) below).

### 2. Run Ingestion

```bash
python batchIngest.py
```

The script will:
- Read `data_sources.json`
- Load and chunk each PDF
- Embed chunks using multilingual-e5-large
- Upsert vectors to Pinecone with rich metadata
- Print ingestion stats per namespace

## Data Sources Format

`data_sources.json` is a JSON array where each entry describes a single PDF document, e.g.:

```json
[
  {
    "type": "pdf",
    "source": "sources/fur_sale_ban_2019.pdf",
    "namespace": "animal_policies",
    "meta": {
      "name": "Proposed Int. No. 1476-A",
      "url": "https://legistar.council.nyc.gov/LegislationDetail.aspx?ID=3903503&GUID=EBE55293-8737-4620-945A-308ADC3A23DC",
      "organization": "NYC City Council",
      "primary_focus": "Farmed Animals",
      "pub_date": "2025-09-25",
      "doc_type": "report",
      "tags": ["nyc", "fur", "farmed animals", "animal advocacy", "legislation"]
    }
  }
]
```

### Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Must be `"pdf"`. Non-PDF types are skipped. |
| `source` | string | Yes | Relative path to PDF file from script directory (e.g., `"sources/file.pdf"`). |
| `namespace` | string | No | Pinecone namespace for organizing docs. Defaults to `"animal_policies"`. |
| `meta.name` | string | No | Human-readable document title. If omitted, uses PDF filename. |
| `meta.url` | string | No | Link to original source for attribution in UI. |
| `meta.organization` | string | No | Source organization. Used for filtering and attribution. |
| `meta.doc_type` | string | No | Classification: `"legislation"`, `"report"`, `"case_study"`, `"guide"`, etc. |
| `meta.primary_focus` | string | No | Policy area: `"cage-free eggs"`, `"broiler welfare"`, `"fur ban"`, etc. |
| `meta.pub_date` | string (ISO 8601) | No | Publication date. Format: `"YYYY-MM-DD"`. Used for temporal filtering. |
| `meta.tags` | array of strings | No | Multi-value tags for fine-grained filtering (e.g., `["California", "city-level", "2020"]`). |


## Section Title Extraction
 
The script uses `pdfplumber` to automatically detect section headers in your PDFs by matching common patterns:
 
- **Legal documents** (legislation, regulations): Matches `§ 17-330`, `Chapter`, `Article`, `Subchapter`
- **Research documents** (reports, studies): Matches ALL-CAPS section titles and numbered sections
- **Preamble/intro text**: Automatically tagged as `"General"`
 
Each chunk is tagged with its parent section, enabling section-based filtering in queries. Example output:
 
```
Processing: Chinese Consumers' Attitudes Toward Animal Welfare
General          ← Preamble/intro chunks
General          ← Introduction chunks  
METHOD & RESULTS  ← Section-specific chunks
METHOD & RESULTS  ← More chunks from same section
```

## Metadata Schema

Each chunk carries rich metadata for filtering, attribution, and analytics:

```json
{
  "source_name": "California Animal Welfare Act (2015)",
  "source_url": "https://...",
  "source_organization": "California State Legislature",
  "source_hash": "abc123def456",
  "chunk_index": 0,
  "page_number": 1,
  "section": "§ 17-330 Definitions",
  "doc_type": "legislation",
  "primary_focus": "cage-free eggs",
  "tags": ["California", "cage-free", "eggs"],
  "publication_date": "2015-01-01",
  "ingestion_date": "2026-03-23",
  "chunk_id": "abc123def456_0"
}
```

## Pinecone Index Configuration

The script automatically creates the Pinecone index if it doesn't exist:

- **Index name**: `perch`
- **Embedding model**: `multilingual-e5-large` (1024 dimensions)
- **Distance metric**: Cosine similarity
- **Namespaces**: Organize documents by topic (e.g., `animal_policies`, `corporate_campaigns`)

### Viewing Index Stats

After ingestion, check Pinecone stats:

```bash
python -c "
from pinecone import Pinecone
pc = Pinecone(api_key='your_api_key')
index = pc.Index('perch')
print(index.describe_index_stats())
"
```

## Advanced Usage

### Re-ingesting Documents

To update a document without duplicates:
1. The script uses `source_hash` (MD5 of file path) to identify documents
2. Re-running ingest with the same file path will upsert (overwrite) existing chunks
3. To remove a document, delete its entries from `data_sources.json` and manually delete vectors from Pinecone

## Optional: Discovering Documents from OSF

The `search_osf.py` script helps discover relevant research projects on the Open Science Framework:
```bash
python search_osf.py "animal welfare" --construct_meta --output osf_candidates.json
```

This generates `osf_candidates.json` with candidate projects. Review and manually:
1. Verify projects are relevant
2. Update `primary_focus` and `doc_type` fields
3. Move relevant entries to `data_sources.json`

Note: Currently searches metadata only. Projects must be downloaded separately and added as PDFs.

## Troubleshooting

### Low retrieval quality
- Increase `chunk_size` if policy concepts are split across chunks
- Add more specific `tags` and `primary_focus` for better metadata filtering
- Check that `publication_date` is accurate for temporal filtering

### Missing metadata in retrievals
- Verify `data_sources.json` has `meta` fields populated
- Check Pinecone index stats to confirm chunks were stored: `index.describe_index_stats()`

## Performance Considerations

- **Ingestion speed**: ~10 chunks/second (depends on network and chunk size)
- **Storage cost**: Pinecone charges per million vectors. A 100-page PDF (~300 chunks) uses minimal capacity
- **Query latency**: Metadata filtering reduces retrieval time by narrowing candidate vectors before similarity search

## Integration with Query Pipeline

Once documents are ingested, the retrieval pipeline in `query.py` automatically:
1. Embeds user queries using the same multilingual-e5-large model
2. Searches Pinecone for most-similar chunks
3. Returns top-k results with full metadata
4. Passes chunks to LLM with context for RAG

See `query.py` for retriever configuration and filtering examples.

### (Not implemented) Using Metadata in Queries

In `query.py`, metadata enables hybrid search and filtering:

```python
# Filter by organization
retriever = docsearch.as_retriever(
    search_kwargs={"filter": {"source_organization": "Faunalytics"}}
)

# Filter by document type
retriever = docsearch.as_retriever(
    search_kwargs={"filter": {"doc_type": "legislation"}}
)

# Filter by tags (multi-value)
retriever = docsearch.as_retriever(
    search_kwargs={"filter": {"tags": {"$in": ["California", "city-level"]}}}
)

# Combine filters
retriever = docsearch.as_retriever(
    search_kwargs={
        "filter": {
            "doc_type": "legislation",
            "publication_date": {"$gte": "2020-01-01"}
        }
    }
)
```

## Future Improvements

- [ ] Support for other formats (DOCX, TXT, web scraping)
- [ ] Automatic table and figure extraction from PDFs
- [ ] Deduplication detection across similar documents
- [ ] Metadata validation and schema enforcement
- [ ] Batch re-ingestion with change tracking

## Contributing

When adding new documents:
- Use consistent `doc_type` values (standardize enum in code)
- Add descriptive `tags` for cross-filtering
- Verify PDFs are machine-readable (not scanned images)

## License

Internal use only for Perch animal advocacy platform.