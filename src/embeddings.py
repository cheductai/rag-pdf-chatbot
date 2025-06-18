"""Embedding generation module using sentence transformers."""

from typing import List, Optional

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from config.settings import settings


class EmbeddingGenerator:
    """Handles text embedding generation using sentence transformers."""
    
    def __init__(self):
        self.model_name = settings.embedding_model
        self.model: Optional[SentenceTransformer] = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Optional[np.ndarray]: Array of embeddings or None if generation fails
        """
        if not texts:
            logger.warning("No texts provided for embedding generation")
            return None
        
        if not self.model:
            logger.error("Embedding model not loaded")
            return None
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    def generate_single_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Optional[np.ndarray]: Embedding vector or None if generation fails
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding generation")
            return None
        
        embeddings = self.generate_embeddings([text])
        return embeddings[0] if embeddings is not None else None
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model.
        
        Returns:
            int: Embedding dimension
        """
        if not self.model:
            logger.error("Embedding model not loaded")
            return 0
        
        # Generate a dummy embedding to get dimension
        dummy_embedding = self.generate_single_embedding("test")
        return len(dummy_embedding) if dummy_embedding is not None else 0
