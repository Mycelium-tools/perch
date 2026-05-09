# app_api.py
#
# FastAPI backend for Perch.
#
# Exposes a single POST endpoint, /ask, that accepts a user question and
# returns a RAG-generated answer using the retrieval_chain defined in
# app/src/query.py.
#
# The chain is initialized at import time (module-level code in query.py runs
# on startup), so the Pinecone index must be populated before the server starts.
#
# Deployment: served via uvicorn, hosted on Railway (see railway.json).
# The frontend (Next.js on Vercel) calls this API from the browser.
#
# A session-based multi-turn conversation version of /ask is preserved in
# commented-out code at the bottom of this file.

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
import time
import os
import re
from pathlib import Path
from urllib.parse import unquote, quote
from app.src.rag.query import retrieval_chain  # retrieval chain from RAG pipeline


app = FastAPI()


# CORS middleware: allows the Vercel frontend (a different origin) to call this API.
# allow_origins=["*"] permits requests from any origin — acceptable for development,
# but should be restricted to the specific frontend domain in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve local ingested PDFs so legacy source_url paths (e.g., ../pdfs/foo.pdf) can resolve.
PDF_DIR = Path(__file__).resolve().parent / "app" / "src" / "rag" / "ingestion" / "pdfs"
if PDF_DIR.exists():
    app.mount("/pdfs", StaticFiles(directory=str(PDF_DIR)), name="pdfs")

# In-memory store mapping session_id -> list of message dicts.
# Resets on server restart; TODO: use a database for persistence in production.
user_histories = {}

# Controls for debug logging based on env vars
def _env_flag(name: str, default: int = 0) -> bool:
    return os.environ.get(name, default)
DEBUG_ALL = _env_flag("PERCH_DEBUG", 0)
DEBUG_RETRIEVAL = DEBUG_ALL or _env_flag("PERCH_RETRIEVAL_DEBUG", 0)
DEBUG_TIMING = DEBUG_ALL or _env_flag("PERCH_TIMING_DEBUG", 0)


def _normalize_source_ref(value: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    raw = unquote(raw)
    raw = raw.replace("http://", "https://")
    raw = re.sub(r"^https://www\.", "https://", raw)
    raw = raw.split("#", 1)[0].split("?", 1)[0]
    return raw.rstrip("/")

def _api_base_url() -> str:
    return (
        os.environ.get("NEXT_PUBLIC_API_URL")
        or os.environ.get("PUBLIC_API_URL")
        or os.environ.get("RAILWAY_PUBLIC_DOMAIN") and f"https://{os.environ.get('RAILWAY_PUBLIC_DOMAIN')}"
        or "https://pawlicy-gpt-production.up.railway.app"
    )

def _resolve_source_url(value: str) -> str:
    """
    Resolve source URLs for citations:
    - keep http(s) as-is
    - map local/relative pdf paths to backend /pdfs route
    """
    url = (value or "").strip()
    if not url:
        return ""
    if url.startswith(("http://", "https://")):
        return url
    if ".pdf" in url.lower():
        filename = unquote(os.path.basename(url))
        return f"{_api_base_url().rstrip('/')}/pdfs/{quote(filename)}"
    return ""


def _safe_year(meta: dict) -> str:
    y = str(meta.get("publication_year", "") or "").strip()
    if y.isdigit() and len(y) == 4:
        return y
    d = str(meta.get("publication_date", "") or "").strip()
    if len(d) >= 4 and d[:4].isdigit():
        return d[:4]
    return "n.d."


def normalize_citations(answer: str, context_docs) -> str:
    """
    Rewrite citation-style markdown links into canonical org/year citations.
    Canonical format:
      ([Organization, Year](https://...)) when public URL exists
      (Organization, Year) otherwise
    """
    if not answer:
        return answer

    source_by_ref = {}
    for doc in context_docs or []:
        meta = doc.metadata or {}
        org = (meta.get("source_organization") or "").strip()
        if not org:
            continue
        src_url = _resolve_source_url(meta.get("source_url", ""))
        year = _safe_year(meta)
        entry = {"org": org, "year": year, "url": src_url}
        for key in {
            _normalize_source_ref(src_url),
            _normalize_source_ref(meta.get("source_name", "")),
            _normalize_source_ref((meta.get("source_name", "") or "").replace(".pdf", "")),
        }:
            if key:
                source_by_ref[key] = entry

    link_re = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    def _replace_link(m):
        href = (m.group(2) or "").strip()
        key = _normalize_source_ref(href)
        entry = source_by_ref.get(key)
        if not entry:
            # Try loose basename matching for local pdf paths.
            base = _normalize_source_ref(os.path.basename(href).replace(".pdf", ""))
            entry = source_by_ref.get(base)
        if not entry:
            return m.group(0)

        org = entry["org"]
        year = entry["year"]
        src_url = entry["url"]
        if src_url:
            return f"[{org}, {year}]({src_url})"
        return f"{org}, {year}"

    normalized = link_re.sub(_replace_link, answer)

    # Ensure citation links are parenthetical.
    normalized = re.sub(
        r"(?<!\()\[([^\]]+,\s*(?:\d{4}|n\.d\.))\]\((https?://[^)]+)\)(?!\))",
        r"([\1](\2))",
        normalized,
    )
    normalized = re.sub(
        r"(?<!\()([A-Za-z][A-Za-z&,\s\-']+,\s*(?:\d{4}|n\.d\.))(?!\))",
        lambda m: f"({m.group(1)})" if "http" not in m.group(1).lower() else m.group(1),
        normalized,
    )
    return normalized


def log_retrieved_docs(context_docs, *, header: str = ""):
    if header:
        print(f"\n--- RETRIEVAL DEBUG {header} ---")
    if not context_docs:
        print("❌ NO DOCUMENTS RETRIEVED")
        if header:
            print("-------------------------------------------\n")
        return

    for i, doc in enumerate(context_docs):
        name = doc.metadata.get("source_name", "Unknown Name")
        url = doc.metadata.get("source_url", "No URL")
        chunk_id = doc.metadata.get("chunk_id", "No ID")
        snippet = doc.page_content[:250].replace('\n', ' ')
        print(f"[{i+1}] NAME: {name}")
        print(f"    URL:  {url}")
        print(f"    chunk_id:  {chunk_id}")
        print(f"    TEXT: {snippet}...")
    if header:
        print("-------------------------------------------\n")

@app.get("/health")
async def health():
    return {"status": "OK"}

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    user_input = data.get("question", "")

    # Ensure conversation history persists per session
    session_id = data.get("session_id", "default") # TODO implement session IDs
    history = user_histories.get(session_id, [])
    
    # Invoke the RAG chain: embeds the question, retrieves relevant chunks from
    # Pinecone, and passes them to Gemini via the custom prompt in query.py.
    # Returns {"answer": str, "context": list[Document]}.
    result = retrieval_chain.invoke({
        "input": user_input,
        "chat_history": history
    })
    result["answer"] = normalize_citations(result.get("answer", ""), result.get("context", []))

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": result["answer"]})
    user_histories[session_id] = history

    if DEBUG_RETRIEVAL:
        log_retrieved_docs(result["context"], header=f"FOR: '{user_input}'")

    return {
        "answer": result["answer"],
        "context": result["context"]  # list of LangChain Document objects (retrieved chunks)
    }    

@app.post("/ask/stream")
async def ask_question_stream(request: Request):
    # Streaming version of /ask — emits SSE tokens as they arrive from the LLM
    data = await request.json()
    user_input = data.get("question", "")

    # Ensure conversation history persists per session
    session_id = data.get("session_id", "default") # TODO implement session IDs
    history = user_histories.get(session_id, [])

    async def generate():
        request_start = time.perf_counter()
        context_docs = []
        full_answer = ""
        sent_docs_status = False
        sent_sources_early = False
        docs_retrieved_at = None
        first_answer_token_at = None

        async for chunk in retrieval_chain.astream({"input": user_input, "chat_history": history}):
            if "context" in chunk:
                context_docs = chunk["context"]
                if context_docs and not sent_docs_status:
                    docs_retrieved_at = time.perf_counter()
                    yield f"data: {json.dumps({'type': 'status', 'stage': 'docs_retrieved'})}\n\n"
                    sent_docs_status = True
                if context_docs and not sent_sources_early:
                    serialized_early = [{"metadata": d.metadata, "page_content": d.page_content} for d in context_docs]
                    yield f"data: {json.dumps({'type': 'sources', 'context': serialized_early})}\n\n"
                    sent_sources_early = True
                if DEBUG_RETRIEVAL:
                    log_retrieved_docs(context_docs)
            if "answer" in chunk and chunk["answer"]:
                if first_answer_token_at is None:
                    first_answer_token_at = time.perf_counter()
                full_answer += chunk["answer"]
                yield f"data: {json.dumps({'type': 'text', 'content': chunk['answer']})}\n\n"

        normalized_answer = normalize_citations(full_answer, context_docs)
        if normalized_answer != full_answer:
            yield f"data: {json.dumps({'type': 'replace_answer', 'content': normalized_answer})}\n\n"
        full_answer = normalized_answer
        
        # Save chat history to session
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": full_answer})
        user_histories[session_id] = history

        # Send serialized source documents after the answer stream ends
        serialized = [{"metadata": d.metadata, "page_content": d.page_content} for d in context_docs]
        yield f"data: {json.dumps({'type': 'sources', 'context': serialized})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

        if DEBUG_TIMING:
            # Timing logs for latency analysis
            end_time = time.perf_counter()
            total_ms = (end_time - request_start) * 1000
            docs_ms = ((docs_retrieved_at - request_start) * 1000) if docs_retrieved_at else None
            first_token_ms = ((first_answer_token_at - request_start) * 1000) if first_answer_token_at else None
            docs_to_first_token_ms = (
                (first_answer_token_at - docs_retrieved_at) * 1000
                if docs_retrieved_at and first_answer_token_at
                else None
            )
            print(
                "[TIMING] /ask/stream "
                f"total_ms={total_ms:.1f} "
                f"docs_retrieved_ms={'NA' if docs_ms is None else f'{docs_ms:.1f}'} "
                f"first_token_ms={'NA' if first_token_ms is None else f'{first_token_ms:.1f}'} "
                f"docs_to_first_token_ms={'NA' if docs_to_first_token_ms is None else f'{docs_to_first_token_ms:.1f}'} "
                f"question_len={len(user_input)} "
                f"retrieved_docs={len(context_docs)}"
            )

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    # Entry point for local development. In production, Railway runs uvicorn
    # directly via the startCommand in railway.json.
    uvicorn.run(app, host="0.0.0.0", port=8000)
