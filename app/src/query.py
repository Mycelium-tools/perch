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
from langchain_openai import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate

# Custom prompt passed to the LLM.
# input_variables must match the keys the chain injects:
#   - {context}: the retrieved document chunks, formatted as a single string by create_stuff_documents_chain
#   - {input}: the user's question, passed directly from retrieval_chain.invoke({"input": ...})
custom_prompt = PromptTemplate(
    input_variables=["context", "input"],
    template="""
        You are an expert assistant.

        First, provide a detailed answer to the question based on your general knowledge.

        Then if applicable, supplement your answer with the following relevant facts from these documents:
        {context}

        However, if the query is clearly not related to or present in the documents, answer solely based on your own knowledge.

        Regardless, in your response to the user, do not mention that you are using these documents. Instead, seamlessly integrate the information into your answer. If possible, format your response in a way that is easy to read, such as using headers, bullet points or numbered lists.

        If there is any conflict between your knowledge and the documents, prioritize the documents.

        Question:
        {input}

        Answer:
        """
    )

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

# index = pc.Index(index_name)  # Raw Pinecone index client — used in debug code below

# Pulls the standard retrieval-QA prompt from LangChain Hub.
# NOTE: this prompt is never actually used — custom_prompt is passed to the chain instead.
# Safe to remove if custom_prompt is the intended prompt.
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# Wrap the vector store as a LangChain retriever.
# docsearch is already defined above — the comment below is outdated.
retriever = docsearch.as_retriever()

# OpenAI LLM setup
# Use gpt-5-nano as the cheapest and least resource intensive model
# temperature=0.0 makes responses deterministic (no sampling randomness).
llm = ChatOpenAI(
    model_name="gpt-5-nano", 
    temperature=0.0 
    )


# create_stuff_documents_chain: combines retrieved Document chunks into a single
# context string, then calls the LLM with custom_prompt.
combine_docs_chain = create_stuff_documents_chain(
    llm, custom_prompt
)

# Optionally filter retrieved chunks by similarity score (0–1).
# Score threshold of 0.7 means only chunks with cosine similarity >= 0.7 are returned.
# retriever = docsearch.as_retriever(search_kwargs={"score_threshold": 0.7})

# create_retrieval_chain: wraps the retriever + combine_docs_chain into one pipeline.
# Calling retrieval_chain.invoke({"input": question}) returns {"answer": ..., "context": ...}.
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
