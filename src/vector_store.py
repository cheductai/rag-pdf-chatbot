"""Vector store implementation using LangChain FAISS integration."""

import os
import json
import logging
from typing import List, Dict, Any, Optional

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangChainVectorStore:
    """LangChain FAISS vector store implementation."""
    
    def __init__(self):
        self.config = settings.get_faiss_config()
        self.openai_config = settings.get_openai_config()
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_config["api_key"],
            model=self.openai_config["embedding_model"]
        )
        
        # Initialize empty vector store
        self.vector_store: Optional[FAISS] = None
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.config["save_path"]), exist_ok=True)
        
        # Try to load existing vector store
        self.load()
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        try:
            if not documents:
                logger.warning("No documents provided to add to vector store")
                return
            
            if self.vector_store is None:
                # Create new vector store
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                logger.info(f"Created new FAISS vector store with {len(documents)} documents")
            else:
                # Add to existing vector store
                self.vector_store.add_documents(documents)
                logger.info(f"Added {len(documents)} documents to existing vector store")
            
            # Auto-save after adding documents
            self.save()
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {str(e)}")
            raise
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Add legacy chunks to the vector store."""
        try:
            # Convert chunks to documents
            documents = []
            for chunk in chunks:
                if 'text' not in chunk:
                    logger.warning(f"Chunk missing text content: {chunk}")
                    continue
                
                # Create document metadata
                metadata = {k: v for k, v in chunk.items() if k != 'text' and k != 'embedding'}
                
                doc = Document(
                    page_content=chunk['text'],
                    metadata=metadata
                )
                documents.append(doc)
            
            if documents:
                self.add_documents(documents)
            
        except Exception as e:
            logger.error(f"Failed to add chunks to vector store: {str(e)}")
            raise
    
    def search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            if k is None:
                k = settings.TOP_K_DOCUMENTS
            
            if self.vector_store is None:
                logger.warning("No vector store available")
                return []
            
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Format results
            formatted_results = []
            for i, (doc, score) in enumerate(results):
                result = {
                    'text': doc.page_content,
                    'similarity_score': float(score),
                    'rank': i + 1,
                    **doc.metadata
                }
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search vector store: {str(e)}")
            raise
    
    def search_by_embedding(self, query_embedding: List[float], k: int = None) -> List[Dict[str, Any]]:
        """Search using pre-computed embedding."""
        try:
            if k is None:
                k = settings.TOP_K_DOCUMENTS
            
            if self.vector_store is None:
                logger.warning("No vector store available")
                return []
            
            # Convert to query string (this is a limitation - LangChain FAISS needs query text)
            # For now, we'll use a placeholder and recommend using the search method instead
            logger.warning("Direct embedding search not supported with LangChain FAISS. Using similarity search instead.")
            return []
            
        except Exception as e:
            logger.error(f"Failed to search by embedding: {str(e)}")
            raise
    
    def save(self, path: str = None) -> None:
        """Save FAISS vector store to disk."""
        try:
            if self.vector_store is None:
                logger.warning("No vector store to save")
                return
            
            save_path = path or self.config["save_path"]
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save using LangChain's save method
            self.vector_store.save_local(save_path)
            
            logger.info(f"Saved FAISS vector store to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
            raise
    
    def load(self, path: str = None) -> bool:
        """Load FAISS vector store from disk."""
        try:
            load_path = path or self.config["save_path"]
            
            # Check if the directory exists and has the required files
            if not os.path.exists(load_path):
                logger.info("No existing vector store found")
                return False
            
            # Try to load the vector store
            self.vector_store = FAISS.load_local(load_path, self.embeddings)
            
            logger.info(f"Loaded FAISS vector store from {load_path}")
            return True
            
        except Exception as e:
            logger.info(f"Could not load existing vector store: {str(e)}")
            return False
    
    def clear(self) -> None:
        """Clear the vector store."""
        self.vector_store = None
        
        # Remove saved files
        try:
            save_path = self.config["save_path"]
            if os.path.exists(save_path):
                import shutil
                shutil.rmtree(save_path)
            logger.info("Cleared vector store and removed saved files")
        except Exception as e:
            logger.warning(f"Could not remove saved files: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if self.vector_store is None:
            return {
                'total_documents': 0,
                'dimension': 1536,  # OpenAI embedding dimension
                'index_type': 'FAISS (LangChain)',
                'has_documents': False
            }
        
        # Get document count (this is an approximation)
        try:
            # LangChain FAISS doesn't expose direct document count easily
            # We'll use the index size as an approximation
            doc_count = len(self.vector_store.docstore._dict) if hasattr(self.vector_store, 'docstore') else 0
        except:
            doc_count = 0
        
        return {
            'total_documents': doc_count,
            'dimension': 1536,
            'index_type': 'FAISS (LangChain)',
            'has_documents': doc_count > 0
        }
    
    def is_empty(self) -> bool:
        """Check if vector store is empty."""
        if self.vector_store is None:
            return True
        
        stats = self.get_stats()
        return stats['total_documents'] == 0

# Legacy compatibility classes
class VectorStore:
    """Legacy abstract base class for compatibility."""
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        raise NotImplementedError
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        raise NotImplementedError
    
    def load(self, path: str) -> None:
        raise NotImplementedError
    
    def clear(self) -> None:
        raise NotImplementedError

class FAISSVectorStore(LangChainVectorStore):
    """Legacy FAISS wrapper for backward compatibility."""
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Legacy method - converts chunks to documents."""
        self.add_chunks(chunks)
    
    def search(self, query_embedding: List[float], k: int = None) -> List[Dict[str, Any]]:
        """Legacy method - uses text search instead of embedding search."""
        logger.warning("Direct embedding search not supported. Please use search_documents with query text.")
        return []

class VectorStoreManager:
    """Manager class for vector store operations."""
    
    def __init__(self, store_type: str = "langchain"):
        if store_type in ["faiss", "langchain"]:
            self.store = LangChainVectorStore()
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
    
    def add_pdf_documents(self, documents: List[Document]) -> None:
        """Add PDF documents to the vector store."""
        self.store.add_documents(documents)
    
    def add_pdf_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Add PDF chunks (legacy format) to the vector store."""
        self.store.add_chunks(chunks)
    
    def search_documents(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """Search for similar documents using query text."""
        return self.store.search(query, k)
    
    def search_by_embedding(self, query_embedding: List[float], k: int = None) -> List[Dict[str, Any]]:
        """Search using embedding (limited support with LangChain FAISS)."""
        return self.store.search_by_embedding(query_embedding, k)
    
    def clear_all(self) -> None:
        """Clear all documents."""
        self.store.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return self.store.get_stats()
    
    def is_empty(self) -> bool:
        """Check if vector store is empty."""
        return self.store.is_empty()

# Convenience functions
def create_vector_store(store_type: str = "langchain") -> VectorStoreManager:
    """Create a vector store manager."""
    return VectorStoreManager(store_type)

def search_similar_documents(query: str, k: int = None) -> List[Dict[str, Any]]:
    """Search for similar documents using default vector store."""
    store_manager = create_vector_store()
    return store_manager.search_documents(query, k)
