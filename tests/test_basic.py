"""Basic tests for RAG PDF Chatbot."""

import unittest
import sys
from pathlib import Path

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

class TestConfiguration(unittest.TestCase):
    """Test configuration loading."""
    
    def test_settings_import(self):
        """Test that settings can be imported."""
        try:
            from config.settings import settings
            self.assertIsNotNone(settings)
        except ImportError:
            self.fail("Could not import settings")
    
    def test_settings_attributes(self):
        """Test that required settings attributes exist."""
        from config.settings import settings
        
        required_attrs = [
            'CHUNK_SIZE', 'CHUNK_OVERLAP', 'MAX_FILE_SIZE_MB',
            'EMBEDDING_MODEL', 'CHAT_MODEL', 'MAX_TOKENS',
            'TEMPERATURE', 'TOP_K_DOCUMENTS'
        ]
        
        for attr in required_attrs:
            self.assertTrue(hasattr(settings, attr), f"Missing attribute: {attr}")

class TestModuleImports(unittest.TestCase):
    """Test that all modules can be imported."""
    
    def test_pdf_processor_import(self):
        """Test PDF processor import."""
        try:
            from src.pdf_processor import PDFProcessor, TextChunker
            self.assertIsNotNone(PDFProcessor)
            self.assertIsNotNone(TextChunker)
        except ImportError as e:
            self.skipTest(f"Skipping PDF processor test due to missing dependencies: {e}")
    
    def test_chatbot_import(self):
        """Test chatbot import."""
        try:
            from src.chatbot import RAGChatbot
            self.assertIsNotNone(RAGChatbot)
        except ImportError as e:
            self.skipTest(f"Skipping chatbot test due to missing dependencies: {e}")

if __name__ == '__main__':
    unittest.main()
