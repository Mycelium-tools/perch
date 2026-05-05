# query.py
#
# RAG query pipeline for Perch. 
# 
# It uses a custom PineconeRetriever to retrieve the most relevant documents from the Pinecone index.
# It then uses the custom_prompt to pass the retrieved documents and the user's question to the LLM.
# It returns the answer and the retrieved documents. 
# It also uses the history_aware_retriever to pass the chat history to the LLM.

from dotenv import load_dotenv
load_dotenv()

import os
from pinecone import Pinecone
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from .retriever import PineconeRetriever

perch_system_prompt = (
    """
    CONTEXT FROM DOCUMENTS: {context}

    You are an expert advisor for animal advocacy organizations. You are NOT a general conversation partner or a generic assistant.
    Your role is to provide actionable, evidence-based guidance on animal welfare policy, legislation, and advocacy strategy grounded in the context. 
    If the user makes a statement that's not clearly related to animal advocacy, BRIEFLY ask for clarity before responding. 
    Never offer additional information or help that you cannot provide using the context.

    INSTRUCTIONS for actionable questions:
    1. Structure: 
        Use ## headers for major sections (e.g., Campaign Plan, Implementation).
        Use ### headers for specific steps or categories within those sections.
        Use #### headers for granular details or data points.
        Bullet points (-) may be used inside sections for lists of 3 or more items. Bullet points should be concise (one idea per bullet). 
    2. Formatting: 
        Use ** for bold emphasis on key terms.        
        For messaging or instructions, use block quotes.
        Use --- for visual separation of distinct sections of your response
    3. Specificity: 
        Avoid generic advice. Reference concrete examples, research, or policy mechanisms when possible. If you don't have specific information, say so. 
        For recommendations, provide an 'Implementation Table' (in Markdown formatting) or list that includes: 1. The specific action. 2. A measurable KPI or target. 3. A projected timeline (e.g., Short-term: 1-3 months).
        For strategies, include a brief 'Constraint Analysis' section. Specifically address potential political, financial, or cultural barriers unique to the animal advocacy space and suggest one mitigation tactic for each.
    4. Sources:
        Every substantive factual claim MUST include an inline citation.
        If a source URL exists, citation format MUST be a markdown hyperlink:
        ([Source Name, Year](https://...)).
        If URL is missing, use plain-text citation: (Source Name, Year).
        Never use vague attribution like 'According to ...' without a citation.
        Do not include a standalone Sources section in your response.
        Never mention 'the documents' or 'the context provided.'
        If no relevant sources exist for your answer, explicitly acknowledge the gap before giving best-effort guidance.
    5. Tone: 
        Professional and grounded. Assume the user has domain expertise in animal advocacy. 
        Answer questions directly. Be concise in prose but exhaustive in evidence.
"""
)


# Custom prompt passed to the LLM to pass the retrieved documents and the user's question to the LLM.
# input_variables must match the keys the chain injects:
#   - {context}: the retrieved document chunks, formatted as a single string by create_stuff_documents_chain
#   - {input}: the user's question, passed directly from retrieval_chain.invoke({"input": ...})
perch_custom_prompt = ChatPromptTemplate.from_messages([
    ("system", perch_system_prompt.strip()),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}" ),
])

# System prompt for contextualizing the user's question to ensure the retriever is aware of chat history
contextualize_q_system_prompt = (
    """
    Given the chat history and the latest user question which might reference previous context, 
    formulate a standalone question that can be understood without the chat history. 
    If the user's statement is a personal fact or irrelevant to animal advocacy, return an empty 
    string or a 'no-op' keyword to prevent unrelated retrieval.
    """
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
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

# Format documents into a string containing source name and URL that the LLM can easily parse
document_prompt = PromptTemplate(
    input_variables=["page_content", "source_name", "source_url"],
    template="--- SOURCE: {source_name} ---\nURL: {source_url}\nCONTENT: {page_content}\n"
)

# OpenAI LLM setup
# Use gpt-5-mini as the base model
# temperature=0.0 makes responses deterministic (no sampling randomness).
llm = ChatOpenAI(
    model_name="gpt-5-mini",
    temperature=0.0,
    streaming=True,  # enable token-by-token streaming for /ask/stream endpoint
)

# create_stuff_documents_chain: combines retrieved Document chunks into a single
# context string, then calls the LLM with custom_prompt.
combine_docs_chain = create_stuff_documents_chain(
    llm, 
    perch_custom_prompt,
    document_variable_name="context", 
    document_prompt=document_prompt
)

# Score threshold of 0.85 means only chunks with cosine similarity >= 0.85 are returned.
# Lower top_k reduces context size and latency.
doc_retriever = PineconeRetriever(pinecone_vector_store=docsearch, score_threshold=0.85, top_k=3)

# History-aware retriever: ensures conversations have chat history saved and the retriever is aware of history
history_aware_retriever = create_history_aware_retriever(
    llm, doc_retriever, contextualize_q_prompt
)

# Build two chains:
# 1) With query contextualization for multi-turn chat
# 2) Direct retrieval for first-turn / empty-history requests to cut latency
retrieval_chain_with_history = create_retrieval_chain(history_aware_retriever, combine_docs_chain)
retrieval_chain_no_history = create_retrieval_chain(doc_retriever, combine_docs_chain)


class AdaptiveRetrievalChain:
    """
    Route to the faster direct-retrieval chain when chat history is empty.
    """

    def _use_history_chain(self, inputs: dict) -> bool:
        history = inputs.get("chat_history")
        return bool(history)

    def invoke(self, inputs: dict):
        chain = retrieval_chain_with_history if self._use_history_chain(inputs) else retrieval_chain_no_history
        return chain.invoke(inputs)

    async def astream(self, inputs: dict):
        chain = retrieval_chain_with_history if self._use_history_chain(inputs) else retrieval_chain_no_history
        async for chunk in chain.astream(inputs):
            yield chunk


# Calling retrieval_chain.invoke({"input": question, "chat_history": [...]})
# returns {"answer": ..., "context": ...}.
retrieval_chain = AdaptiveRetrievalChain()
