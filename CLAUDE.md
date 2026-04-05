# Perch — Project Context for LLM Agents

## Overview

Perch is an AI-powered policy assistant for animal advocacy organizations. It uses Retrieval-Augmented Generation (RAG) to answer questions about animal welfare policy, legislation, and advocacy strategy, grounding responses in a curated corpus of policy documents.

---

## Tech Stack

**Backend**
- Python 3.11.9, FastAPI, Uvicorn
- LangChain (`langchain`, `langchain-openai`, `langchain-core`, `langchain-classic`)
- Pinecone (vector database, index: `perch`, namespace: `animal_policies`)
- OpenAI API (`gpt-5-mini`, embeddings via `multilingual-e5-large` through Pinecone)

**Frontend**
- Next.js 15 (React 19, TypeScript)
- Tailwind CSS v4
- NextAuth.js v4 (credential-based auth, JWT sessions)
- Supabase (user database)
- `react-markdown` for rendering LLM responses

**Infrastructure**
- Backend: Render (`render.yaml` at repo root)
- Frontend: Vercel (`vercel.json` at repo root)

---

## Directory Structure

```
perch/
├── app_api.py                    # FastAPI app — /ask and /ask/stream endpoints
├── requirements.txt              # Python dependencies
├── render.yaml                   # Render deployment config (backend)
├── vercel.json                   # Vercel deployment config (frontend)
├── .env                          # Backend env vars (OPENAI_API_KEY, PINECONE_API_KEY)
│
├── app/
│   └── src/
│       ├── query.py              # LangChain RAG chain (retrieval_chain, llm)
│       └── rag/
│           ├── ingest.py         # PDF ingestion → Pinecone
│           ├── chunking_utils.py # Section header extraction from PDFs
│           ├── search_osf.py     # OSF API search for policy documents
│           └── data_sources.json # Registry of ingested documents (metadata)
│
└── app/src/nextjs-frontend/
    ├── package.json
    └── src/
        ├── app/
        │   ├── page.tsx          # Main chat UI
        │   ├── layout.tsx        # Root layout with SessionProvider
        │   └── api/auth/         # NextAuth + Supabase registration routes
        ├── components/
        │   ├── ClientLayout.tsx  # ChatContext provider (chats, history, activeId)
        │   ├── Sidebar.tsx       # Chat history sidebar
        │   ├── Header.tsx
        │   └── BirdLoader.tsx    # Loading animation
        └── lib/
            └── supabase.ts       # Supabase client
```

---

## Key Files

| File | Role |
|------|------|
| `app_api.py` | FastAPI entry point; exposes `/ask` (JSON) and `/ask/stream` (SSE) |
| `app/src/query.py` | Builds `retrieval_chain` and `llm`; imported at startup by `app_api.py` |
| `app/src/rag/ingest.py` | One-time script to ingest PDFs into Pinecone |
| `app/src/nextjs-frontend/src/app/page.tsx` | Chat interface; handles submit, streaming, markdown rendering |
| `app/src/nextjs-frontend/src/components/ClientLayout.tsx` | React context for chat state |

---

## Data Flow

```
User types question (page.tsx)
  → POST /ask/stream  (app_api.py)
  → retrieval_chain.astream({"input": question})  (query.py)
      → Embed question with multilingual-e5-large
      → Similarity search in Pinecone (threshold: 0.8)
      → Pass top-k chunks + question to gpt-5-mini via custom_prompt
  → SSE stream: {"type":"text","content":"..."} per token
  → SSE final: {"type":"sources","context":[...]}
  → Frontend renders tokens incrementally with ReactMarkdown
```

---

## Development Setup

```bash
# Backend
cd /path/to/perch
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app_api:app --reload --port 8000

# Frontend
cd app/src/nextjs-frontend
npm install
npm run dev   # runs on http://localhost:3000

# Ingest documents into Pinecone (one-time / as needed)
cd app/src/rag
python ingest.py
```

---

## Environment Variables

**Backend** (`.env` at repo root):
```
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_CLOUD=aws          # optional, default: aws
PINECONE_REGION=us-east-1   # optional
```

**Frontend** (`app/src/nextjs-frontend/.env.local`):
```
NEXT_PUBLIC_SUPABASE_URL=...
NEXT_PUBLIC_SUPABASE_ANON_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...
NEXTAUTH_SECRET=...
NEXT_PUBLIC_API_URL=...     # optional; defaults to localhost:8000 or Render URL
```

---

## Deployment

- **Backend**: Render — build: `pip install -r requirements.txt`, start: `uvicorn app_api:app --host 0.0.0.0 --port $PORT`
- **Frontend**: Vercel — auto-detected Next.js, no custom config needed
- Production backend URL: `https://pawlicy-gpt-production.up.railway.app` (hardcoded fallback in `page.tsx`)

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/ask` | Full JSON response `{answer, context}` |
| POST | `/ask/stream` | SSE stream — text chunks then sources |

**SSE event format** (`/ask/stream`):
```
data: {"type": "text", "content": "<token>"}
data: {"type": "sources", "context": [{metadata, page_content}, ...]}
data: {"type": "done"}
```

---

## Known Issues / TODOs

- `allow_origins=["*"]` in CORS should be restricted to the Vercel frontend domain in production
- `.env` file is tracked in git (should be gitignored)
- Chat history is client-side only (localStorage + React state) — resets on refresh, no DB persistence
- `promptSuggestions` in `page.tsx` all have placeholder text — not yet populated
- The `hub.pull("langchain-ai/retrieval-qa-chat")` in `query.py` is fetched but unused
- Multi-turn conversation (session history) is implemented but commented out in `app_api.py`
