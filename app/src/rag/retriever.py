import cohere
from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

class PineconeRetriever(BaseRetriever):
    """
    A custom Pinecone retriever that uses Cohere's reranking to return the top-k most relevant documents.

    Args:
        pinecone_vector_store: The Pinecone vector store to search.
        score_threshold: The threshold for the cosine similarity score.
        top_k: The number of documents to return.
    """

    pinecone_vector_store: PineconeVectorStore
    score_threshold: float
    top_k: int

    def __init__(
        self,
        pinecone_vector_store: PineconeVectorStore,
        score_threshold: float,
        top_k: int,
        **kwargs
    ):
        super().__init__(
            pinecone_vector_store=pinecone_vector_store,
            score_threshold=score_threshold,
            top_k=top_k,
            **kwargs
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 1. Initial Retrieval (Fetch more candidates for reranking)
        # Note: VectorStore.similarity_search returns Document objects
        initial_docs = self.pinecone_vector_store.similarity_search(
            query, 
            k=self.top_k * 3 # fetch 3x more documents than we need to ensure we have enough for reranking
        )
        
        if not initial_docs:
            return []

        # 2. Rerank
        return self._rerank(query, initial_docs, top_k=self.top_k)

    def _rerank(
        self, query: str, documents: List[Document], top_k: int
    ) -> List[Document]:
        co = cohere.ClientV2()
        
        # Cohere needs strings, so extract page_content
        doc_texts = [doc.page_content for doc in documents]
        
        response = co.rerank(
            model="rerank-v3.5",
            query=query,
            documents=doc_texts,
            top_n=top_k,
        )

        # Map back to original Document objects to preserve metadata
        reranked_docs = []
        for result in response.results:
            reranked_docs.append(documents[result.index])

        return reranked_docs
