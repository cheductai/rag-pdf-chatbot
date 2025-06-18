"""PDF processing module for text extraction and chunking."""

import tempfile
from pathlib import Path
from typing import List, Optional

import PyPDF2
from loguru import logger

from config.settings import settings


class PDFProcessor:
    """Handles PDF text extraction and chunking operations."""
    
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.max_file_size = settings.max_file_size_mb * 1024 * 1024  # Convert to bytes
    
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
            
            # Check if it's a valid PDF
            with open(file_path, 'rb') as file:
                PyPDF2.PdfReader(file)
            
            return True
            
        except Exception as e:
            logger.error(f"PDF validation failed: {e}")
            return False
    
    def extract_text_from_pdf(self, file_path: Path) -> Optional[str]:
        """
        Extract text content from PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Optional[str]: Extracted text or None if extraction fails
        """
        try:
            if not self.validate_pdf(file_path):
                return None
            
            text_content = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(f"[Page {page_num + 1}]\n{page_text}")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue
            
            full_text = "\n\n".join(text_content)
            logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
            
            return full_text if full_text.strip() else None
            
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            return None
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for processing.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        if not text or not text.strip():
            return []
        
        # Simple sentence-aware chunking
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add sentence to current chunk
            test_chunk = f"{current_chunk}. {sentence}" if current_chunk else sentence
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Handle overlap
                if len(chunks) > 0 and self.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = f"{overlap_text} {sentence}"
                else:
                    current_chunk = sentence
        
        # Add the last chunk
        if current_chunk and current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def process_pdf(self, file_path: Path) -> Optional[List[str]]:
        """
        Complete PDF processing pipeline.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Optional[List[str]]: List of text chunks or None if processing fails
        """
        logger.info(f"Processing PDF: {file_path}")
        
        # Extract text
        text = self.extract_text_from_pdf(file_path)
        if not text:
            logger.error("Failed to extract text from PDF")
            return None
        
        # Chunk text
        chunks = self.chunk_text(text)
        if not chunks:
            logger.error("Failed to create chunks from extracted text")
            return None
        
        logger.info(f"Successfully processed PDF into {len(chunks)} chunks")
        return chunks
