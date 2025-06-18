"""LangChain-based document processing module."""

import tempfile
from pathlib import Path
from typing import List, Optional

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from loguru import logger

from config.settings import settings


class LangChainDocumentProcessor:
    """Handles PDF processing using LangChain components."""
    
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.max_file_size = settings.max_file_size_mb * 1024 * 1024  # Convert to bytes
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def validate_pdf(self, file_path: Path) -> bool:
        """
        Validate PDF file size and format.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check file size
            if file_path.stat().st_size > self.max_file_size:
                logger.error(f"File size exceeds limit: {file_path.stat().st_size} bytes")
                return False
            
            # Check if it's a valid PDF by trying to load it
            loader = PyMuPDFLoader(str(file_path))
            # Try to load first page to validate
            docs = loader.load()
            if not docs:
                logger.error("PDF appears to be empty or corrupted")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"PDF validation failed: {e}")
            return False
    
    def load_and_split_pdf(self, file_path: Path) -> Optional[List[Document]]:
        """
        Load PDF and split into documents using LangChain.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Optional[List[Document]]: List of document chunks or None if processing fails
        """
        try:
            if not self.validate_pdf(file_path):
                return None
            
            logger.info(f"Loading PDF with PyMuPDFLoader: {file_path}")
            
            # Load PDF using PyMuPDFLoader
            loader = PyMuPDFLoader(str(file_path))
            documents = loader.load()
            
            if not documents:
                logger.error("No documents loaded from PDF")
                return None
            
            logger.info(f"Loaded {len(documents)} pages from PDF")
            
            # Split documents into chunks
            logger.info("Splitting documents into chunks")
            split_docs = self.text_splitter.split_documents(documents)
            
            if not split_docs:
                logger.error("No chunks created from documents")
                return None
            
            logger.info(f"Created {len(split_docs)} chunks from PDF")
            
            # Add metadata to chunks
            for i, doc in enumerate(split_docs):
                doc.metadata.update({
                    'chunk_id': i,
                    'source_file': file_path.name,
                    'total_chunks': len(split_docs)
                })
            
            return split_docs
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return None
    
    def process_pdf_from_bytes(self, pdf_bytes: bytes, filename: str) -> Optional[List[Document]]:
        """
        Process PDF from bytes (useful for uploaded files).
        
        Args:
            pdf_bytes: PDF file content as bytes
            filename: Original filename
            
        Returns:
            Optional[List[Document]]: List of document chunks or None if processing fails
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_file.write(pdf_bytes)
                temp_path = Path(temp_file.name)
            
            # Process the temporary file
            result = self.load_and_split_pdf(temp_path)
            
            # Clean up temporary file
            temp_path.unlink(missing_ok=True)
            
            # Update metadata with original filename
            if result:
                for doc in result:
                    doc.metadata['source_file'] = filename
            
            return result
            
        except Exception as e:
            logger.error(f"PDF processing from bytes failed: {e}")
            return None
    
    def get_text_from_documents(self, documents: List[Document]) -> List[str]:
        """
        Extract text content from Document objects.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List[str]: List of text chunks
        """
        return [doc.page_content for doc in documents if doc.page_content.strip()]
