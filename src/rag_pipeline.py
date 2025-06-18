"""RAG pipeline implementation combining all components."""

from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from src.embeddings import EmbeddingGenerator
from src.pdf_processor import PDFProcessor
from src.vector_store import FAISSVectorStore
from src.openai_generator import OpenAIResponseGenerator


class RAGPipeline:
    """Complete RAG pipeline for PDF question answering."""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = FAISSVectorStore()
        self.is_initialized = False
        self.current_pdf_name: Optional[str] = None
        self.response_generator = OpenAIResponseGenerator()
    
    def initialize(self) -> bool:
        """
        Initialize the RAG pipeline.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Try to load existing index
            if self.vector_store.load_index():
                self.is_initialized = True
                logger.info("RAG pipeline initialized with existing index")
                return True
            
            logger.info("No existing index found. Ready to process new PDF.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            return False
    
    def process_pdf(self, pdf_path: Path) -> bool:
        """
        Process a PDF file and build the vector index.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Starting PDF processing: {pdf_path.name}")
            
            # Clear existing index
            self.vector_store.clear_index()
            
            # Process PDF
            chunks = self.pdf_processor.process_pdf(pdf_path)
            if not chunks:
                logger.error("Failed to process PDF")
                return False
            
            # Generate embeddings
            embeddings = self.embedding_generator.generate_embeddings(chunks)
            if embeddings is None:
                logger.error("Failed to generate embeddings")
                return False
            
            # Create and populate vector store
            dimension = embeddings.shape[1]
            self.vector_store.create_index(dimension)
            self.vector_store.add_embeddings(embeddings, chunks)
            
            # Save index
            if not self.vector_store.save_index():
                logger.warning("Failed to save index to disk")
            
            self.is_initialized = True
            self.current_pdf_name = pdf_path.name
            logger.info(f"Successfully processed PDF: {pdf_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return False
    
    def query(self, question: str) -> Optional[str]:
        """
        Query the RAG system with a question.
        
        Args:
            question: User question
            
        Returns:
            Optional[str]: Generated answer or None if query fails
        """
        if not self.is_initialized:
            logger.error("RAG pipeline not initialized")
            return None
        
        if not question or not question.strip():
            logger.warning("Empty question provided")
            return None
        
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_single_embedding(question)
            if query_embedding is None:
                logger.error("Failed to generate query embedding")
                return None
            
            # Search for relevant chunks
            results = self.vector_store.search(query_embedding)
            if not results:
                return "I couldn't find relevant information in the document to answer your question."
            
            # Generate answer based on retrieved context
            answer = self._generate_answer(question, results)
            logger.info("Successfully generated answer")
            return answer
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return None
    
    def _generate_answer(self, question: str, context_results: List[Tuple[str, float]]) -> str:
        """
        Generate an answer using OpenAI based on the question and retrieved context.
        
        Args:
            question: User question
            context_results: List of (text, similarity_score) tuples
            
        Returns:
            str: Generated answer
        """
        return self.response_generator.generate_response(question, context_results)
    
    def get_status(self) -> dict:
        """
        Get the current status of the RAG pipeline.
        
        Returns:
            dict: Status information
        """
        openai_info = self.response_generator.get_model_info()
        
        return {
            "initialized": self.is_initialized,
            "current_pdf": self.current_pdf_name,
            "total_chunks": len(self.vector_store.texts) if self.vector_store.texts else 0,
            "index_size": self.vector_store.index.ntotal if self.vector_store.index else 0,
            "openai_enabled": openai_info["enabled"],
            "openai_model": openai_info["model"]
        }
