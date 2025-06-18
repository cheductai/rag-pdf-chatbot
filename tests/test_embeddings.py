"""Tests for embeddings module."""

import numpy as np
import pytest

from src.embeddings import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test cases for EmbeddingGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = EmbeddingGenerator()
    
    def test_generate_single_embedding(self):
        """Test single embedding generation."""
        text = "This is a test sentence."
        embedding = self.generator.generate_single_embedding(text)
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1
        assert embedding.shape[0] > 0
    
    def test_generate_embeddings_multiple(self):
        """Test multiple embeddings generation."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = self.generator.generate_embeddings(texts)
        
        assert embeddings is not None
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] > 0
    
    def test_generate_embedding_empty(self):
        """Test embedding generation with empty text."""
        embedding = self.generator.generate_single_embedding("")
        assert embedding is None
        
        embedding = self.generator.generate_single_embedding("   ")
        assert embedding is None
    
    def test_get_embedding_dimension(self):
        """Test getting embedding dimension."""
        dimension = self.generator.get_embedding_dimension()
        assert isinstance(dimension, int)
        assert dimension > 0
