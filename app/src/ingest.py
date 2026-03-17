# ingest.py
#
# One-off document ingestion script for Perch's RAG pipeline.
#
# Loads a single PDF, splits it into overlapping chunks, embeds each chunk
# using multilingual-e5-large, and upserts the vectors into a Pinecone index.
# Run this script directly to populate the vector store before the API can
# retrieve relevant context.
#
# For ingesting multiple documents at once, see batchIngest.py.
#
# Usage:
#   python ingest.py
# (ingests `pdfTest` into `index_name` / `namespace` defined at module level)

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os

# Input data for ingestion — used when running this script directly
pdfTest = "rag_sources/An Experimental Investigation of the Impact of Video Media on Pork Consumption.pdf"
topic = "pork"
source = "Faunlytics"
year = "2017"
# Namespace acts as a logical partition within the index — different document
# collections can live in the same index under separate namespaces.
namespace = "animal_policies"


# Embedding model — multilingual-e5-large produces 1024-dimensional vectors
model_name = 'multilingual-e5-large'

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Serverless spec — cloud/region determine where the index is hosted.
# Defaults to AWS us-east-1 if env vars are not set.
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

# Target Pinecone index
index_name = "perch"

def ingest_document(file_path, index_name, namespace):
    # Note: `index_name` and `namespace` here are local parameters that shadow
    # the module-level variables of the same name.

    # 1. Load PDF
    # PyPDFLoader loads one Document per page.
    # Note: the original comment said "PDF or markdown" but this loader only
    # handles PDFs — use a different loader (e.g. UnstructuredMarkdownLoader)
    # for markdown files.
    file_id = hash(file_path)
    print(f"Loading file: {file_path} with ID: {file_id}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from PDF.")

    # 2. Split into overlapping chunks
    # chunk_size=500 (characters) keeps chunks small enough for the embedding
    # model's context window. chunk_overlap=50 ensures context isn't lost at
    # chunk boundaries.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks for embedding.")

    # Attach metadata to every chunk so vectors can be filtered or identified
    # after retrieval. These fields are hardcoded for the fur ban document —
    # update them for different documents.
    metadata_fields = {
        "topic": f"{topic}",
        "year": year,
        "source": f"{source}",
        "chunkCount": len(chunks)
    }

    for (chunk_count, chunk) in enumerate(chunks):        
        # Add chunk metadata
        chunk.metadata.update(metadata_fields)
        
        # Set chunk ID for Pinecone upsert
        chunk.id = f"{file_id}-{chunk_count}"

    # 3. Initialize embedding model
    # PineconeEmbeddings wraps the hosted Pinecone inference API.
    embeddings = PineconeEmbeddings(
        model=model_name,
        pinecone_api_key=os.environ.get('PINECONE_API_KEY')
    )

    # 4. Create Pinecone index if it doesn't already exist
    # `embeddings.dimension` returns the output dimension of the model (1024 for
    # multilingual-e5-large). cosine metric is standard for semantic similarity.
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=embeddings.dimension,
            metric="cosine",
            spec=spec
        )
        print(f"Created new index: {index_name}")
    else:
        print(f"Using existing index: {index_name}")

    # 5. Embed chunks and upsert to Pinecone
    # PineconeVectorStore.from_documents embeds each chunk and upserts the
    # resulting vectors in a single call. If the namespace does not exist,
    # Pinecone creates it automatically.
    PineconeVectorStore.from_documents(
        chunks,
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )

    print(f"Upserted {len(chunks)} chunks into index '{index_name}' under namespace '{namespace}'.")

    # Print index stats to confirm the upsert was successful
    stats = pc.Index(index_name).describe_index_stats()
    print(f"Index stats after upsert: {stats}")

if __name__ == "__main__":
    ingest_document(pdfTest, index_name, namespace)
