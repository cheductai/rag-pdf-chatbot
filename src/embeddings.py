"""Embeddings generation module using LangChain OpenAIEmbeddings."""

import logging
from typing import List, Dict, Any, Optional
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangChainEmbeddingsGenerator:
    """Handles embeddings generation using LangChain OpenAIEmbeddings."""
    
    def __init__(self):
        self.config = settings.get_openai_config()
        
        # Initialize LangChain OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.config["api_key"],
            model=self.config["embedding_model"]
        )
        
        # Validate configuration
        settings.validate()
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            
            # Clean text
            text = text.strip().replace('\n', ' ').replace('\r', ' ')
            
            # Generate embedding using LangChain
            embedding = self.embeddings.embed_query(text)
            
            logger.debug(f"Generated embedding for text of length {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            if not texts:
                return []
            
            # Clean texts
            cleaned_texts = []
            for text in texts:
                if text and text.strip():
                    cleaned_text = text.strip().replace('\n', ' ').replace('\r', ' ')
                    cleaned_texts.append(cleaned_text)
                else:
                    logger.warning("Skipping empty text in batch")
                    cleaned_texts.append(" ")  # Add placeholder for empty text
            
            # Generate embeddings using LangChain
            embeddings = self.embeddings.embed_documents(cleaned_texts)
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings batch: {str(e)}")
            raise
    
    def generate_embeddings_for_documents(self, documents: List[Document]) -> List[Document]:
        """Generate embeddings for LangChain documents."""
        try:
            if not documents:
                return []
            
            # Extract texts from documents
            texts = [doc.page_content for doc in documents]
            
            # Generate embeddings
            embeddings = self.generate_embeddings_batch(texts)
            
            # Add embeddings to document metadata
            enriched_documents = []
            for doc, embedding in zip(documents, embeddings):
                enriched_doc = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        'embedding': embedding,
                        'embedding_model': self.config["embedding_model"]
                    }
                )
                enriched_documents.append(enriched_doc)
            
            logger.info(f"Added embeddings to {len(enriched_documents)} documents")
            return enriched_documents
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings for documents: {str(e)}")
            raise
    
    def generate_embeddings_for_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for legacy chunk format."""
        try:
            if not chunks:
                return []
            
            # Extract texts
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.generate_embeddings_batch(texts)
            
            # Add embeddings to chunks
            enriched_chunks = []
            for chunk, embedding in zip(chunks, embeddings):
                enriched_chunk = chunk.copy()
                enriched_chunk['embedding'] = embedding
                enriched_chunk['embedding_model'] = self.config["embedding_model"]
                enriched_chunks.append(enriched_chunk)
            
            logger.info(f"Added embeddings to {len(enriched_chunks)} chunks")
            return enriched_chunks
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings for chunks: {str(e)}")
            raise

# Legacy compatibility classes
class EmbeddingsGenerator(LangChainEmbeddingsGenerator):
    """Legacy wrapper for backward compatibility."""
    pass

class CachedEmbeddingsGenerator(LangChainEmbeddingsGenerator):
    """Legacy cached embeddings generator - caching is now handled by LangChain."""
    
    def __init__(self):
        super().__init__()
        logger.info("Note: Caching is now handled internally by LangChain")

class EmbeddingCache:
    """Legacy cache class - kept for compatibility but not used."""
    
    def __init__(self):
        logger.warning("EmbeddingCache is deprecated - LangChain handles caching internally")
        self._cache = {}
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        return None
    
    def set(self, text: str, model: str, embedding: List[float]) -> None:
        pass
    
    def clear(self) -> None:
        pass
    
    def size(self) -> int:
        return 0

# Convenience functions
def create_embeddings_generator(use_cache: bool = True) -> LangChainEmbeddingsGenerator:
    """Create embeddings generator - caching is handled by LangChain."""
    return LangChainEmbeddingsGenerator()

def generate_query_embedding(query: str) -> List[float]:
    """Generate embedding for a search query."""
    generator = create_embeddings_generator()
    return generator.generate_embedding(query)
