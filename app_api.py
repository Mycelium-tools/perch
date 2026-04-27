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
import uvicorn
import json
from app.src.rag.query import retrieval_chain  # retrieval chain from RAG pipeline
from app.src.rag.query import llm  # NOTE: llm is imported but not used in the active endpoint;
                                # it is only referenced in the commented-out history version below


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

# In-memory store mapping session_id -> list of message dicts.
# Resets on server restart; TODO: use a database for persistence in production.
user_histories = {}

def build_history_text(history):
    # Formats a list of {"role": "user"|"assistant", "content": str} dicts
    # into a plain-text conversation transcript for injection into the prompt.
    # NOTE: also only used in the commented-out endpoint below.
    return "\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in history
    )

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

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": result["answer"]})
    user_histories[session_id] = history

    # DEBUG LOGGING: Loop through retrieved chunks and log to terminal
    print(f"\n--- RETRIEVAL DEBUG FOR: '{user_input}' ---")
    if not result["context"]:
        print("❌ NO DOCUMENTS RETRIEVED")
    else:
        for i, doc in enumerate(result["context"]):
            name = doc.metadata.get("source_name", "Unknown Name")
            url = doc.metadata.get("source_url", "No URL")
            # Print a snippet of the text to verify the content matches the metadata
            snippet = doc.page_content[:250].replace('\n', ' ')
            print(f"[{i+1}] NAME: {name}")
            print(f"    URL:  {url}")
            print(f"    TEXT: {snippet}...")
    print("-------------------------------------------\n")

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
        context_docs = []
        full_answer = ""
        async for chunk in retrieval_chain.astream({"input": user_input, "chat_history": history}):
            if "context" in chunk:
                context_docs = chunk["context"]
                # --- DEBUG LOGGING START ---
                for i, doc in enumerate(context_docs):
                    name = doc.metadata.get("source_name", "Unknown Name")
                    url = doc.metadata.get("source_url", "No URL")
                    chunk_id = doc.metadata.get("chunk_id", "No ID")
                    # Print a snippet of the text to verify the content matches the metadata
                    snippet = doc.page_content[:250].replace('\n', ' ')
                    print(f"[{i+1}] NAME: {name}")
                    print(f"    URL:  {url}")
                    print(f"    chunk_id:  {chunk_id}")
                    print(f"    TEXT: {snippet}...")
                # --- DEBUG LOGGING END ---
            if "answer" in chunk and chunk["answer"]:
                full_answer += chunk["answer"]
                yield f"data: {json.dumps({'type': 'text', 'content': chunk['answer']})}\n\n"
        
        # Save chat history to session
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": full_answer})
        user_histories[session_id] = history

        # Send serialized source documents after the answer stream ends
        serialized = [{"metadata": d.metadata, "page_content": d.page_content} for d in context_docs]
        yield f"data: {json.dumps({'type': 'sources', 'context': serialized})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    # Entry point for local development. In production, Railway runs uvicorn
    # directly via the startCommand in railway.json.
    uvicorn.run(app, host="0.0.0.0", port=8000)




# --- Commented-out: session-based multi-turn conversation version of /ask ---
# This version tracks conversation history per session_id and passes it to
# the chain. Requires adding a {history} variable to custom_prompt in query.py.
#
# @app.post("/ask")
# async def ask_question(request: Request):
#     data = await request.json()
#     print("Received data:", data)
#     user_input = data.get("question", "")
#     session_id = data.get("session_id", "default")  # session/user id from frontend
#
#     # Get or create history for this session
#     history = user_histories.setdefault(session_id, [])
#     history.append({"role": "user", "content": user_input})
#
#     # Format history as plain text for prompt injection
#     history_text = build_history_text(history)
#
#     # Call the chain with history included
#     result = retrieval_chain.invoke({
#         "history": history_text,
#         "input": user_input,
#         "context": ""  # let the chain handle context retrieval
#     })
#
#     history.append({"role": "assistant", "content": result["answer"]})
#
#     return {
#         "answer": result["answer"],
#         "context": result["context"]
#     }
