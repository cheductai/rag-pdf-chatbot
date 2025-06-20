"""Configuration management for RAG PDF Chatbot."""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings using Singleton pattern."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # OpenAI Configuration
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
        self.TOP_K_DOCUMENTS = int(os.getenv("TOP_K_DOCUMENTS", "5"))
        
        # PDF Processing Configuration
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
        
        # FAISS Configuration
        self.FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index/")
        self.FAISS_METADATA_PATH = os.getenv("FAISS_METADATA_PATH", "data/faiss_index/metadata.json")
        
        # Application Configuration
        self.APP_TITLE = os.getenv("APP_TITLE", "RAG PDF Chatbot")
        self.APP_DESCRIPTION = os.getenv("APP_DESCRIPTION", "Ask questions about your PDF documents")
        self.DEBUG = os.getenv("DEBUG", "False").lower() == "true"
        
        # Paths
        self.DATA_DIR = "data"
        self.UPLOADS_DIR = "data/uploads"
        self.PROCESSED_DIR = "data/processed"
        
        self._initialized = True
    
    def validate(self) -> bool:
        """Validate required configuration."""
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file.")
        return True
    
    def get_pdf_config(self) -> Dict[str, Any]:
        """Get PDF processing configuration."""
        return {
            "chunk_size": self.CHUNK_SIZE,
            "chunk_overlap": self.CHUNK_OVERLAP,
            "max_file_size_mb": self.MAX_FILE_SIZE_MB,
            "supported_formats": [".pdf"]
        }
    
    def get_faiss_config(self) -> Dict[str, Any]:
        """Get FAISS configuration."""
        return {
            "index_type": "IndexFlatIP",
            "dimension": 1536,  # OpenAI embedding dimension
            "save_path": self.FAISS_INDEX_PATH,
            "metadata_path": self.FAISS_METADATA_PATH
        }
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration."""
        return {
            "api_key": self.OPENAI_API_KEY,
            "embedding_model": self.EMBEDDING_MODEL,
            "chat_model": self.CHAT_MODEL,
            "max_tokens": self.MAX_TOKENS,
            "temperature": self.TEMPERATURE,
            "top_k_documents": self.TOP_K_DOCUMENTS
        }
    
    def get_gradio_config(self) -> Dict[str, Any]:
        """Get Gradio configuration."""
        return {
            "theme": "soft",
            "title": self.APP_TITLE,
            "description": self.APP_DESCRIPTION,
            "max_file_size": self.MAX_FILE_SIZE_MB * 1024 * 1024,  # Convert to bytes
            "allow_flagging": "never"
        }

# Global settings instance
settings = Settings()
