# LangChain RAG PDF Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) chatbot built with LangChain components. Upload PDF documents and ask questions about their content using state-of-the-art language models and vector search.

## Features

- 📚 **Advanced PDF Processing**: PyMuPDFLoader for robust PDF text extraction
- 🔍 **Semantic Search**: OpenAI embeddings with FAISS vector store
- 💬 **Intelligent Responses**: ChatOpenAI with RetrievalQA chain
- 💾 **Persistent Storage**: Save and load FAISS indices for faster startup
- ⚙️ **LangChain Integration**: Built with production-ready LangChain components
- 📊 **Comprehensive Monitoring**: Real-time status and logging

## LangChain Components Used

- **PyMuPDFLoader**: Advanced PDF document loading and parsing
- **RecursiveCharacterTextSplitter**: Intelligent text chunking
- **OpenAIEmbeddings**: High-quality text embeddings
- **FAISS**: Efficient vector similarity search
- **ChatOpenAI**: OpenAI language model integration
- **RetrievalQA**: Question-answering chain with retrieval

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:
\`\`\`bash
git clone <repository-url>
cd rag-pdf-chatbot
\`\`\`

2. Create a virtual environment:
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

4. Set up environment variables:
\`\`\`bash
cp .env.example .env
# Edit .env and add your OpenAI API key
\`\`\`

### Usage

1. Start the application:
\`\`\`bash
python main.py
\`\`\`

2. Open your browser and navigate to `http://localhost:7860`

3. Upload a PDF file and wait for processing

4. Start asking questions about your document!

## Configuration

### Required Settings
- `RAG_OPENAI_API_KEY`: Your OpenAI API key (required)

### Optional Settings
- `RAG_OPENAI_MODEL`: OpenAI model (default: gpt-4o-mini)
- `RAG_EMBEDDING_MODEL`: OpenAI embedding model (default: text-embedding-3-small)
- `RAG_CHUNK_SIZE`: Text chunk size (default: 1000)
- `RAG_CHUNK_OVERLAP`: Chunk overlap (default: 200)
- `RAG_TEMPERATURE`: Response creativity (default: 0.7)

### Getting OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account or sign in
3. Navigate to API Keys section
4. Create a new API key
5. Add it to your `.env` file as `RAG_OPENAI_API_KEY`

## Architecture

\`\`\`
src/
├── langchain_processor.py      # PyMuPDFLoader + text splitting
├── langchain_vectorstore.py    # OpenAI embeddings + FAISS
├── langchain_rag_pipeline.py   # ChatOpenAI + RetrievalQA
└── gradio_interface.py         # Web interface
\`\`\`

## Technical Details

- **Document Loading**: PyMuPDFLoader handles complex PDF layouts and metadata
- **Text Splitting**: RecursiveCharacterTextSplitter with smart chunking
- **Embeddings**: OpenAI's text-embedding-3-small for high-quality vectors
- **Vector Store**: FAISS with cosine similarity search
- **LLM**: ChatOpenAI with configurable temperature and token limits
- **QA Chain**: RetrievalQA with custom prompts for accurate responses

## Development

### Running Tests
\`\`\`bash
pytest tests/
\`\`\`

### Code Formatting
\`\`\`bash
black src/ tests/
isort src/ tests/
\`\`\`

## License

MIT License - see LICENSE file for details.
\`\`\`

Create tests for the LangChain components:

```python file="tests/test_langchain_components.py"
"""Tests for LangChain components."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.langchain_processor import LangChainDocumentProcessor
from src.langchain_vectorstore import LangChainVectorStore
from src.langchain_rag_pipeline import LangChainRAGPipeline


class TestLangChainDocumentProcessor:
    """Test cases for LangChainDocumentProcessor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = LangChainDocumentProcessor()
    
    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor.chunk_size > 0
        assert self.processor.chunk_overlap >= 0
        assert self.processor.text_splitter is not None
    
    def test_validate_pdf_nonexistent(self):
        """Test PDF validation with non-existent file."""
        fake_path = Path("nonexistent.pdf")
        assert not self.processor.validate_pdf(fake_path)
    
    def test_get_text_from_documents(self):
        """Test text extraction from documents."""
        # Mock documents
        mock_docs = [
            Mock(page_content="First document content"),
            Mock(page_content="Second document content"),
            Mock(page_content="")  # Empty content should be filtered
        ]
        
        texts = self.processor.get_text_from_documents(mock_docs)
        
        assert len(texts) == 2
        assert "First document content" in texts
        assert "Second document content" in texts


class TestLangChainVectorStore:
    """Test cases for LangChainVectorStore."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('src.langchain_vectorstore.OpenAIEmbeddings'):
            self.vector_store = LangChainVectorStore()
    
    def test_initialization(self):
        """Test vector store initialization."""
        assert self.vector_store.index_path is not None
        assert isinstance(self.vector_store.documents, list)
    
    def test_get_status(self):
        """Test status information retrieval."""
        status = self.vector_store.get_status()
        
        assert isinstance(status, dict)
        assert "initialized" in status
        assert "embeddings_ready" in status
        assert "total_documents" in status
        assert "embedding_model" in status
    
    @patch('src.langchain_vectorstore.FAISS')
    def test_create_vectorstore(self, mock_faiss):
        """Test vector store creation."""
        # Mock embeddings
        self.vector_store.embeddings = Mock()
        
        # Mock documents
        mock_docs = [Mock(page_content="Test content")]
        
        # Mock FAISS.from_documents
        mock_faiss.from_documents.return_value = Mock()
        
        result = self.vector_store.create_vectorstore(mock_docs)
        
        assert result is True
        mock_faiss.from_documents.assert_called_once()


class TestLangChainRAGPipeline:
    """Test cases for LangChainRAGPipeline."""
    
    @patch('src.langchain_rag_pipeline.ChatOpenAI')
    @patch('src.langchain_rag_pipeline.LangChainVectorStore')
    @patch('src.langchain_rag_pipeline.LangChainDocumentProcessor')
    def setup_method(self, mock_processor, mock_vector_store, mock_chat_openai):
        """Set up test fixtures."""
        self.pipeline = LangChainRAGPipeline()
    
    def test_initialization(self):
        """Test pipeline initialization."""
        assert self.pipeline.document_processor is not None
        assert self.pipeline.vector_store is not None
        assert hasattr(self.pipeline, 'is_initialized')
        assert hasattr(self.pipeline, 'current_pdf_name')
    
    def test_get_status(self):
        """Test status information retrieval."""
        # Mock vector store status
        self.pipeline.vector_store.get_status = Mock(return_value={
            "initialized": False,
            "embeddings_ready": False,
            "total_documents": 0,
            "embedding_model": "test-model"
        })
        
        status = self.pipeline.get_status()
        
        assert isinstance(status, dict)
        assert "initialized" in status
        assert "current_pdf" in status
        assert "total_documents" in status
        assert "llm_ready" in status
        assert "qa_chain_ready" in status
    
    def test_query_not_initialized(self):
        """Test query when pipeline is not initialized."""
        self.pipeline.is_initialized = False
        
        result = self.pipeline.query("Test question")
        
        assert result is None
    
    def test_query_empty_question(self):
        """Test query with empty question."""
        self.pipeline.is_initialized = True
        
        result = self.pipeline.query("")
        assert result is None
        
        result = self.pipeline.query("   ")
        assert result is None
