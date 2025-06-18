"""FAISS vector store implementation for similarity search."""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from loguru import logger

from config.settings import settings


class FAISSVectorStore:
    """FAISS-based vector store for efficient similarity search."""
    
    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.texts: List[str] = []
        self.index_path = Path(settings.faiss_index_path)
        self.texts_path = self.index_path.parent / "texts.pkl"
        self.dimension: Optional[int] = None
    
    def create_index(self, dimension: int) -> None:
        """
        Create a new FAISS index.
        
        Args:
            dimension: Dimension of the embedding vectors
        """
        try:
            self.dimension = dimension
            # Use IndexFlatIP for cosine similarity (after normalization)
            self.index = faiss.IndexFlatIP(dimension)
            logger.info(f"Created FAISS index with dimension {dimension}")
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            raise
    
    def add_embeddings(self, embeddings: np.ndarray, texts: List[str]) -> None:
        """
        Add embeddings and corresponding texts to the index.
        
        Args:
            embeddings: Array of embedding vectors
            texts: List of corresponding text chunks
        """
        if self.index is None:
            logger.error("Index not created. Call create_index() first.")
            return
        
        if len(embeddings) != len(texts):
            logger.error("Number of embeddings and texts must match")
            return
        
        try:
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.index.add(embeddings.astype(np.float32))
            self.texts.extend(texts)
            
            logger.info(f"Added {len(embeddings)} embeddings to index")
            logger.info(f"Total vectors in index: {self.index.ntotal}")
            
        except Exception as e:
            logger.error(f"Failed to add embeddings to index: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = None) -> List[Tuple[str, float]]:
        """
        Search for similar vectors in the index.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List[Tuple[str, float]]: List of (text, similarity_score) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty or not created")
            return []
        
        if k is None:
            k = min(settings.top_k_results, len(self.texts))
        
        try:
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_embedding)
            
            # Search
            similarities, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.texts) and similarity >= settings.similarity_threshold:
                    results.append((self.texts[idx], float(similarity)))
            
            logger.info(f"Found {len(results)} relevant results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def save_index(self) -> bool:
        """
        Save the FAISS index and texts to disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.index is None:
            logger.error("No index to save")
            return False
        
        try:
            # Create directory if it doesn't exist
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save texts
            with open(self.texts_path, 'wb') as f:
                pickle.dump(self.texts, f)
            
            logger.info(f"Saved index to {self.index_path}")
            logger.info(f"Saved texts to {self.texts_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index(self) -> bool:
        """
        Load the FAISS index and texts from disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.index_path.exists() or not self.texts_path.exists():
                logger.info("Index files not found")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            
            # Load texts
            with open(self.texts_path, 'rb') as f:
                self.texts = pickle.load(f)
            
            self.dimension = self.index.d
            
            logger.info(f"Loaded index from {self.index_path}")
            logger.info(f"Loaded {len(self.texts)} texts")
            logger.info(f"Index contains {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def clear_index(self) -> None:
        """Clear the current index and texts."""
        self.index = None
        self.texts = []
        self.dimension = None
        logger.info("Cleared index and texts")
