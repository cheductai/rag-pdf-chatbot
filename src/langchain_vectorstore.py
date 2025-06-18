"""LangChain-based vector store implementation."""

from pathlib import Path
from typing import List, Optional, Tuple

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from loguru import logger

from config.settings import settings


class LangChainVectorStore:
    """FAISS vector store using LangChain components."""
    
    def __init__(self):
        self.embeddings = None
        self.vectorstore: Optional[FAISS] = None
        self.index_path = Path(settings.faiss_index_path)
        self.documents: List[Document] = []
        self._initialize_embeddings()
    
    def _initialize_embeddings(self) -> None:
        """Initialize OpenAI embeddings."""
        try:
            if not settings.openai_api_key:
                logger.error("OpenAI API key not provided")
                return
            
            logger.info(f"Initializing OpenAI embeddings with model: {settings.embedding_model}")
            self.embeddings = OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=settings.openai_api_key
            )
            logger.info("OpenAI embeddings initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def create_vectorstore(self, documents: List[Document]) -> bool:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.embeddings:
            logger.error("Embeddings not initialized")
            return False
        
        if not documents:
            logger.error("No documents provided")
            return False
        
        try:
            logger.info(f"Creating FAISS vector store from {len(documents)} documents")
            
            # Create FAISS vector store
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            self.documents = documents
            logger.info("FAISS vector store created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            return False
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to existing vector store.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return False
        
        try:
            logger.info(f"Adding {len(documents)} documents to vector store")
            self.vectorstore.add_documents(documents)
            self.documents.extend(documents)
            logger.info("Documents added successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """
        Perform similarity search in the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List[Tuple[Document, float]]: List of (document, similarity_score) tuples
        """
        if not self.vectorstore:
            logger.warning("Vector store not initialized")
            return []
        
        if k is None:
            k = settings.top_k_results
        
        try:
            logger.info(f"Performing similarity search for: {query[:100]}...")
            
            # Perform similarity search with scores
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # Filter by similarity threshold (note: FAISS returns distance, lower is better)
            filtered_results = []
            for doc, score in results:
                # Convert distance to similarity (approximate)
                similarity = 1.0 / (1.0 + score)
                if similarity >= settings.similarity_threshold:
                    filtered_results.append((doc, similarity))
            
            logger.info(f"Found {len(filtered_results)} relevant results")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def save_vectorstore(self) -> bool:
        """
        Save the vector store to disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.vectorstore:
            logger.error("No vector store to save")
            return False
        
        try:
            # Create directory if it doesn't exist
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            self.vectorstore.save_local(str(self.index_path))
            
            logger.info(f"Vector store saved to {self.index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            return False
    
    def load_vectorstore(self) -> bool:
        """
        Load vector store from disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.embeddings:
            logger.error("Embeddings not initialized")
            return False
        
        try:
            if not self.index_path.exists():
                logger.info("Vector store files not found")
                return False
            
            logger.info(f"Loading vector store from {self.index_path}")
            
            # Load FAISS index
            self.vectorstore = FAISS.load_local(
                str(self.index_path),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            logger.info("Vector store loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False
    
    def clear_vectorstore(self) -> None:
        """Clear the current vector store."""
        self.vectorstore = None
        self.documents = []
        logger.info("Vector store cleared")
    
    def get_retriever(self, search_kwargs: dict = None):
        """
        Get a retriever for the vector store.
        
        Args:
            search_kwargs: Additional search parameters
            
        Returns:
            VectorStoreRetriever: LangChain retriever object
        """
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return None
        
        if search_kwargs is None:
            search_kwargs = {"k": settings.top_k_results}
        
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def get_status(self) -> dict:
        """
        Get vector store status information.
        
        Returns:
            dict: Status information
        """
        return {
            "initialized": self.vectorstore is not None,
            "embeddings_ready": self.embeddings is not None,
            "total_documents": len(self.documents),
            "embedding_model": settings.embedding_model
        }
