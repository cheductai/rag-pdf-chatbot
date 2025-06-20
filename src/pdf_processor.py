"""PDF processing module using LangChain for text extraction and chunking."""

import os
import logging
from typing import List, Dict, Any
import tempfile

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF text extraction using LangChain's PyMuPDFLoader."""
    
    def __init__(self):
        self.config = settings.get_pdf_config()
        self.max_file_size = self.config["max_file_size_mb"] * 1024 * 1024  # Convert to bytes
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def validate_pdf(self, file_path: str) -> bool:
        """Validate PDF file size and format."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                raise ValueError(f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum allowed size ({self.config['max_file_size_mb']}MB)")
            
            # Check file extension
            if not file_path.lower().endswith('.pdf'):
                raise ValueError("Only PDF files are supported")
            
            return True
        except Exception as e:
            logger.error(f"PDF validation failed: {str(e)}")
            raise
    
    def extract_documents_from_pdf(self, file_path: str) -> List[Document]:
        """Extract documents from PDF file using LangChain."""
        try:
            self.validate_pdf(file_path)
            
            # Load PDF using PyMuPDFLoader
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            
            if not documents:
                raise ValueError("No content found in PDF")
            
            logger.info(f"Successfully loaded {len(documents)} pages from PDF")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to extract documents from PDF: {str(e)}")
            raise
    
    def extract_documents_from_bytes(self, pdf_bytes: bytes, filename: str = "uploaded.pdf") -> List[Document]:
        """Extract documents from PDF bytes using temporary file."""
        try:
            # Check file size
            if len(pdf_bytes) > self.max_file_size:
                raise ValueError(f"File size ({len(pdf_bytes) / 1024 / 1024:.1f}MB) exceeds maximum allowed size ({self.config['max_file_size_mb']}MB)")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_bytes)
                temp_path = temp_file.name
            
            try:
                # Load using temporary file
                loader = PyMuPDFLoader(temp_path)
                documents = loader.load()
                
                # Update metadata with original filename
                for doc in documents:
                    doc.metadata['source'] = filename
                    doc.metadata['original_source'] = filename
                
                if not documents:
                    raise ValueError("No content found in PDF")
                
                logger.info(f"Successfully loaded {len(documents)} pages from PDF bytes")
                return documents
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            
        except Exception as e:
            logger.error(f"Failed to extract documents from PDF bytes: {str(e)}")
            raise
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks using RecursiveCharacterTextSplitter."""
        try:
            if not documents:
                logger.warning("No documents provided for splitting")
                return []
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_index'] = i
                chunk.metadata['total_chunks'] = len(chunks)
                
                # Ensure we have consistent metadata keys
                if 'source' not in chunk.metadata and 'original_source' in chunk.metadata:
                    chunk.metadata['source'] = chunk.metadata['original_source']
            
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to split documents: {str(e)}")
            raise

class TextChunker:
    """Legacy wrapper for backward compatibility - now uses RecursiveCharacterTextSplitter."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Convert LangChain documents to legacy format for compatibility."""
        try:
            # Split documents
            chunks = self.text_splitter.split_documents(documents)
            
            # Convert to legacy format
            legacy_chunks = []
            for i, chunk in enumerate(chunks):
                legacy_chunk = {
                    'text': chunk.page_content,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'page_number': chunk.metadata.get('page', 1),
                    'source_file': chunk.metadata.get('source', 'unknown'),
                    'file_path': chunk.metadata.get('source', 'unknown')
                }
                legacy_chunks.append(legacy_chunk)
            
            logger.info(f"Created {len(legacy_chunks)} chunks from {len(documents)} documents")
            return legacy_chunks
            
        except Exception as e:
            logger.error(f"Failed to chunk documents: {str(e)}")
            raise
    
    def chunk_by_tokens(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Legacy method - converts text to Document and chunks."""
        doc = Document(page_content=text, metadata=metadata)
        return self.chunk_documents([doc])
    
    def chunk_pages(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Legacy method for backward compatibility."""
        documents = []
        for page_data in pages_data:
            doc = Document(
                page_content=page_data['text'],
                metadata={
                    'page': page_data.get('page_number', 1),
                    'source': page_data.get('source_file', 'unknown'),
                    'file_path': page_data.get('file_path', 'unknown')
                }
            )
            documents.append(doc)
        
        return self.chunk_documents(documents)

class PDFProcessorFactory:
    """Factory for creating PDF processors with different configurations."""
    
    @staticmethod
    def create_processor(processor_type: str = "default") -> PDFProcessor:
        """Create PDF processor based on type."""
        if processor_type == "default":
            return PDFProcessor()
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")
    
    @staticmethod
    def create_chunker(strategy: str = "recursive", **kwargs) -> TextChunker:
        """Create text chunker based on strategy."""
        if strategy in ["recursive", "token_based"]:
            return TextChunker(**kwargs)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

# Convenience functions
def process_pdf_file(file_path: str) -> List[Document]:
    """Process PDF file and return LangChain documents."""
    processor = PDFProcessorFactory.create_processor()
    documents = processor.extract_documents_from_pdf(file_path)
    chunks = processor.split_documents(documents)
    return chunks

def process_pdf_bytes(pdf_bytes: bytes, filename: str = "uploaded.pdf") -> List[Document]:
    """Process PDF bytes and return LangChain documents."""
    processor = PDFProcessorFactory.create_processor()
    documents = processor.extract_documents_from_bytes(pdf_bytes, filename)
    chunks = processor.split_documents(documents)
    return chunks

def process_pdf_file_legacy(file_path: str) -> List[Dict[str, Any]]:
    """Legacy function that returns dict format for backward compatibility."""
    processor = PDFProcessorFactory.create_processor()
    documents = processor.extract_documents_from_pdf(file_path)
    chunker = TextChunker()
    return chunker.chunk_documents(documents)

def process_pdf_bytes_legacy(pdf_bytes: bytes, filename: str = "uploaded.pdf") -> List[Dict[str, Any]]:
    """Legacy function that returns dict format for backward compatibility."""
    processor = PDFProcessorFactory.create_processor()
    documents = processor.extract_documents_from_bytes(pdf_bytes, filename)
    chunker = TextChunker()
    return chunker.chunk_documents(documents)
