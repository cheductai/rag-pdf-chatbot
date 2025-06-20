"""Document retrieval logic for RAG system using LangChain integration."""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass

from .vector_store import create_vector_store
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Data class for retrieval results."""
    documents: List[Dict[str, Any]]
    query: str
    total_found: int
    retrieval_time: float

class DocumentRetriever:
    """Handles document retrieval for RAG system."""
    
    def __init__(self):
        self.vector_store = create_vector_store()
        self.config = settings.get_openai_config()
        self.top_k = self.config["top_k_documents"]
    
    def retrieve(self, query: str, k: int = None) -> RetrievalResult:
        """Retrieve relevant documents for a query using LangChain FAISS."""
        try:
            import time
            start_time = time.time()
            
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            
            # Use provided k or default
            k = k or self.top_k
            
            # Check if vector store has documents
            if self.vector_store.is_empty():
                logger.warning("Vector store is empty - no documents to retrieve")
                return RetrievalResult(
                    documents=[],
                    query=query,
                    total_found=0,
                    retrieval_time=time.time() - start_time
                )
            
            # Search for similar documents using text query (LangChain FAISS)
            similar_docs = self.vector_store.search_documents(query, k)
            
            retrieval_time = time.time() - start_time
            
            logger.info(f"Retrieved {len(similar_docs)} documents for query in {retrieval_time:.3f}s")
            
            return RetrievalResult(
                documents=similar_docs,
                query=query,
                total_found=len(similar_docs),
                retrieval_time=retrieval_time
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {str(e)}")
            raise
    
    def retrieve_with_filtering(self, 
                              query: str, 
                              source_file: str = None, 
                              min_similarity: float = 0.0,
                              k: int = None) -> RetrievalResult:
        """Retrieve documents with additional filtering."""
        try:
            # Get initial results
            result = self.retrieve(query, k)
            
            # Apply filters
            filtered_docs = []
            for doc in result.documents:
                # Filter by source file if specified
                if source_file and doc.get('source_file') != source_file:
                    continue
                
                # Filter by minimum similarity
                if doc.get('similarity_score', 0) < min_similarity:
                    continue
                
                filtered_docs.append(doc)
            
            # Update result
            result.documents = filtered_docs
            result.total_found = len(filtered_docs)
            
            logger.info(f"Filtered results: {len(filtered_docs)} documents")
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve filtered documents: {str(e)}")
            raise
    
    def get_context_for_query(self, query: str, max_context_length: int = 4000) -> str:
        """Get formatted context string for a query."""
        try:
            result = self.retrieve(query)
            
            if not result.documents:
                return "No relevant context found."
            
            context_parts = []
            current_length = 0
            
            for i, doc in enumerate(result.documents):
                # Format document info
                doc_info = f"[Document {i+1}] "
                if doc.get('source_file'):
                    doc_info += f"From: {doc['source_file']} "
                if doc.get('page_number'):
                    doc_info += f"(Page {doc['page_number']}) "
                doc_info += f"(Relevance: {doc.get('similarity_score', 0):.3f})\n"
                
                doc_text = doc.get('text', '')
                full_doc = doc_info + doc_text + "\n\n"
                
                # Check if adding this document would exceed max length
                if current_length + len(full_doc) > max_context_length and context_parts:
                    break
                
                context_parts.append(full_doc)
                current_length += len(full_doc)
            
            context = "".join(context_parts)
            logger.info(f"Generated context of {len(context)} characters from {len(context_parts)} documents")
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get context for query: {str(e)}")
            return "Error retrieving context."
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics."""
        vector_stats = self.vector_store.get_stats()
        return {
            'vector_store_stats': vector_stats,
            'top_k_documents': self.top_k,
            'has_documents': not self.vector_store.is_empty()
        }

class ContextFormatter:
    """Formats retrieved context for different use cases."""
    
    @staticmethod
    def format_for_chat(documents: List[Dict[str, Any]], query: str) -> str:
        """Format context for chat completion."""
        if not documents:
            return "No relevant context found in the uploaded documents."
        
        context_parts = [
            f"Based on the uploaded PDF documents, here is the relevant context for your question: '{query}'\n\n"
        ]
        
        for i, doc in enumerate(documents):
            context_parts.append(f"**Context {i+1}** (Relevance: {doc.get('similarity_score', 0):.3f})")
            
            if doc.get('source_file'):
                context_parts.append(f"Source: {doc['source_file']}")
            
            if doc.get('page_number'):
                context_parts.append(f"Page: {doc['page_number']}")
            
            context_parts.append(f"Content: {doc.get('text', '')}\n")
        
        return "\n".join(context_parts)
    
    @staticmethod
    def format_for_display(documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Format context for UI display."""
        formatted_docs = []
        
        for i, doc in enumerate(documents):
            formatted_doc = {
                'title': f"Result {i+1}",
                'source': doc.get('source_file', 'Unknown'),
                'page': str(doc.get('page_number', 'N/A')),
                'relevance': f"{doc.get('similarity_score', 0):.3f}",
                'content': doc.get('text', '')[:500] + ('...' if len(doc.get('text', '')) > 500 else ''),
                'full_content': doc.get('text', '')
            }
            formatted_docs.append(formatted_doc)
        
        return formatted_docs

# Convenience functions
def retrieve_documents(query: str, k: int = None) -> RetrievalResult:
    """Retrieve documents for a query using default retriever."""
    retriever = DocumentRetriever()
    return retriever.retrieve(query, k)

def get_context_for_query(query: str, max_length: int = 4000) -> str:
    """Get formatted context for a query."""
    retriever = DocumentRetriever()
    return retriever.get_context_for_query(query, max_length)
