"""Main application entry point for the LangChain RAG PDF Chatbot."""

import sys
from pathlib import Path

from loguru import logger

from config.settings import settings
from src.gradio_interface import GradioInterface


def setup_logging() -> None:
    """Configure logging for the application."""
    # Remove default logger
    logger.remove()
    
    # Add console logging
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file logging if specified
    if settings.log_file:
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            level=settings.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days"
        )


def main() -> None:
    """Main application function."""
    # Setup logging
    setup_logging()
    
    logger.info("Starting LangChain RAG PDF Chatbot")
    
    # Check for OpenAI API key
    if not settings.openai_api_key:
        logger.error("OpenAI API key not provided. Please set RAG_OPENAI_API_KEY in your .env file")
        sys.exit(1)
    
    logger.info(f"Configuration: OpenAI Model={settings.openai_model}, Embedding Model={settings.embedding_model}")
    
    try:
        # Create and launch the Gradio interface
        app = GradioInterface()
        
        # Log system status
        status = app.rag_pipeline.get_status()
        logger.info(f"System Status: LLM Ready={status['llm_ready']}, Embeddings Ready={status['embeddings_ready']}")
        
        app.launch()
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
