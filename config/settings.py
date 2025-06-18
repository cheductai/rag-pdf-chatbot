"""Configuration management for the RAG PDF Chatbot."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    max_tokens: int = Field(default=1000, description="Maximum tokens for response generation")
    temperature: float = Field(default=0.7, description="Temperature for response generation")
    
    # Processing Configuration
    chunk_size: int = Field(default=1000, description="Text chunk size for processing")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    max_file_size_mb: int = Field(default=50, description="Maximum PDF file size in MB")
    
    # FAISS Configuration
    faiss_index_path: str = Field(default="data/faiss_index", description="Path to FAISS index")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold for retrieval")
    top_k_results: int = Field(default=5, description="Number of top results to retrieve")
    
    # Gradio Configuration
    gradio_port: int = Field(default=7860, description="Gradio server port")
    gradio_share: bool = Field(default=False, description="Create public Gradio link")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default="logs/app.log", description="Log file path")
    
    # Directories
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")
    temp_dir: Path = Field(default=Path("temp"), description="Temporary files directory")
    
    class Config:
        env_file = ".env"
        env_prefix = "RAG_"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()
        # Set OpenAI API key as environment variable for LangChain
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [self.data_dir, self.logs_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
