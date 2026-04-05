# query.py
#
# RAG query pipeline for Perch.
#
# This module builds and exports `retrieval_chain`, the main LangChain chain
# used to answer user questions. When invoked with {"input": <question>}, it:
#   1. Embeds the question using the multilingual-e5-large model
#   2. Retrieves the top-k most similar document chunks from Pinecone
#   3. Passes the retrieved chunks + question to Gemini via custom_prompt
#   4. Returns {"answer": <str>, "context": <list of Documents>}
#
# `retrieval_chain` and `llm` are imported by app_api.py at startup.
# The Pinecone index ("policy-docs") and namespace ("horse_carriage") must
# already exist and be populated before this module is imported.

from dotenv import load_dotenv
load_dotenv()

import os
import time  # only used in commented-out debug code below

from pinecone import Pinecone
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic import hub
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

system_prompt = (
    "CONTEXT FROM DOCUMENTS:\n{context}\n\n"
    "You are an expert advisor for animal advocacy organizations. Your role is to provide "
    "actionable, evidence-based guidance on animal welfare policy, legislation, and advocacy strategy. "
    "Ground your responses in research, case studies, and real-world examples found in the context.\n"
    "\n\n"
    "INSTRUCTIONS:\n"
    "1. **Structure**: Use ## Headers to organize sections. Use * for bullet points and ** for bold emphasis on key terms.\n"
    "2. **Formatting**: Each section header should be on its own line followed by a blank line. Bullet points should be concise (one idea per bullet).\n"
    "3. **Specificity**: Avoid generic advice. Reference concrete examples, jurisdictions, or policy mechanisms when possible. If you don't have specific information, say so.\n"
    "4. **Sources**: Never mention 'the documents' or 'the context provided.' If no relevant sources exist for your answer, acknowledge the gap and provide general knowledge if helpful.\n"
    "5. **Tone**: Professional and grounded. Assume the user has domain expertise in animal advocacy.\n"
)

# Custom prompt passed to the LLM.
# 1. Use ChatPromptTemplate for better GPT-5 instruction following.
# 2. Use .strip() or left-aligned text to avoid passing "tab" spaces to the LLM.

# input_variables must match the keys the chain injects:
#   - {context}: the retrieved document chunks, formatted as a single string by create_stuff_documents_chain
#   - {input}: the user's question, passed directly from retrieval_chain.invoke({"input": ...})
custom_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt.strip()),
    ("human", "From an animal advocacy perspective, {input}"),
])

# Pinecone config — must match what was used at ingest time.
# Changing `namespace` here switches which set of documents is retrieved.
index_name = "perch"
namespace = "animal_policies"
model_name = 'multilingual-e5-large'

# Initialize Pinecone client using API key from environment
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Initialize embedding model — multilingual-e5-large produces 1024-dimensional vectors
embeddings = PineconeEmbeddings(
    model=model_name,
    pinecone_api_key=os.environ.get('PINECONE_API_KEY')
)

# Connect to an existing Pinecone index and namespace as a LangChain vector store.
# This does not load vectors into memory — queries are made remotely at retrieval time.
docsearch = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace
)

# Pulls the standard retrieval-QA prompt from LangChain Hub.
# NOTE: this prompt is never actually used — custom_prompt is passed to the chain instead.
# Safe to remove if custom_prompt is the intended prompt.
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# Wrap the vector store as a LangChain retriever.
# retriever = docsearch.as_retriever()

# OpenAI LLM setup
# Use gpt-5-mini as the base model
# temperature=0.0 makes responses deterministic (no sampling randomness).
llm = ChatOpenAI(
    model_name="gpt-5-mini",
    temperature=0.0,
    streaming=True,  # enable token-by-token streaming for /ask/stream endpoint
)

# Format documents into a string containing source name and URL that the LLM can easily parse
document_prompt = PromptTemplate(
    input_variables=["page_content", "source_name", "source_url"],
    template="--- SOURCE: {source_name} ---\nURL: {source_url}\nCONTENT: {page_content}\n"
)

# create_stuff_documents_chain: combines retrieved Document chunks into a single
# context string, then calls the LLM with custom_prompt.
combine_docs_chain = create_stuff_documents_chain(
    llm, 
    custom_prompt,
    document_variable_name="context", 
    document_prompt=document_prompt
)

# Optionally filter retrieved chunks by similarity score (0–1).
# Score threshold of 0.8 means only chunks with cosine similarity >= 0.8 are returned.
retriever = docsearch.as_retriever(search_kwargs={"score_threshold": 0.8})

# create_retrieval_chain: wraps the retriever + combine_docs_chain into one pipeline.
# Calling retrieval_chain.invoke({"input": question}) returns {"answer": ..., "context": ...}.
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
