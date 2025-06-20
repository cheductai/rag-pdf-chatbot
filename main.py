"""Main entry point for RAG PDF Chatbot application."""

import os
import sys
import logging
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chatbot.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to launch the RAG PDF Chatbot."""
    try:
        # Import after path setup
        from src.ui import launch_app
        from config.settings import settings
        
        logger.info("Starting RAG PDF Chatbot...")
        
        # Validate configuration
        try:
            settings.validate()
            logger.info("Configuration validated successfully")
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            print(f"\n❌ Configuration Error: {e}")
            print("\nPlease check your .env file and ensure all required variables are set.")
            print("You can copy .env.example to .env and fill in your values.")
            return 1
        
        # Create necessary directories
        os.makedirs("data/uploads", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("data/faiss_index", exist_ok=True)
        
        # Launch the application
        logger.info("Launching Gradio interface...")
        print(f"\n🚀 Starting {settings.APP_TITLE}...")
        print(f"📄 {settings.APP_DESCRIPTION}")
        print("\n🔧 Configuration:")
        print(f"   • Chat Model: {settings.CHAT_MODEL}")
        print(f"   • Embedding Model: {settings.EMBEDDING_MODEL}")
        print(f"   • Max Tokens: {settings.MAX_TOKENS}")
        print(f"   • Temperature: {settings.TEMPERATURE}")
        print(f"   • Top-K Documents: {settings.TOP_K_DOCUMENTS}")
        print(f"   • Chunk Size: {settings.CHUNK_SIZE}")
        print(f"   • Chunk Overlap: {settings.CHUNK_OVERLAP}")
        print(f"   • Max File Size: {settings.MAX_FILE_SIZE_MB}MB")
        print("\n🌐 The web interface will open in your browser...")
        print("💡 Upload a PDF file and start asking questions about its content!")
        
        # Launch with default settings
        launch_app(
            share=False,  # Set to True to create a public link
            debug=settings.DEBUG,
            server_name="0.0.0.0",  # Allow access from other devices on network
            server_port=7860
        )
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        print(f"\n❌ Error starting application: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
