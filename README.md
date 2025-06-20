# RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on PDF content using **LangChain**, **OpenAI models**, **FAISS vector search**, and a clean **Gradio interface**.

## 🚀 Features

- **LangChain Integration**: Built with LangChain for robust document processing and retrieval chains
- **PDF Processing**: Advanced PDF text extraction using PyMuPDF with intelligent chunking
- **Vector Storage**: Efficient FAISS vector storage with LangChain integration
- **RetrievalQA Chain**: Uses LangChain's RetrievalQA for optimized question-answering
- **Modern UI**: Clean, responsive Gradio web interface
- **Persistent Storage**: FAISS indices are saved to disk for faster reloading
- **Smart Retrieval**: Context-aware document retrieval with citation support

## 📋 Tech Stack

- **LangChain**: Document processing, text splitting, embeddings, and QA chains
- **OpenAI**: GPT models for chat and text embeddings
- **FAISS**: Facebook AI Similarity Search for vector storage
- **PyMuPDF**: Advanced PDF text extraction
- **Gradio**: Modern web interface
- **Python 3.8+**: Core runtime

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd rag-pdf-chatbot
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env file and add your OpenAI API key
   ```

5. **Run the application:**
   ```bash
   python main.py
   ```

## ⚙️ Configuration

Create a `.env` file from the provided example and configure the following:

### Required Settings
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Optional Settings (with defaults)
```env
# Model Configuration
EMBEDDING_MODEL=text-embedding-ada-002
CHAT_MODEL=gpt-3.5-turbo
MAX_TOKENS=1000
TEMPERATURE=0.1
TOP_K_DOCUMENTS=5

# PDF Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_FILE_SIZE_MB=50

# Application
APP_TITLE=RAG PDF Chatbot
APP_DESCRIPTION=Ask questions about your PDF documents
DEBUG=False
```

## 🎯 Usage

1. **Start the application:**
   ```bash
   python main.py
   ```

2. **Open your browser** and navigate to `http://localhost:7860`

3. **Upload a PDF document** using the file uploader

4. **Wait for processing** - you'll see a confirmation message

5. **Ask questions** about the PDF content in the chat interface

6. **Get AI-powered answers** with citations and source references

### Example Questions
- "What is the main topic of this document?"
- "Can you summarize the key points?"
- "What does the document say about [specific topic]?"
- "Are there any recommendations mentioned?"

## 🏗️ Architecture

```
rag-pdf-chatbot/
├── src/
│   ├── pdf_processor.py      # PDF text extraction and chunking
│   ├── embeddings.py         # OpenAI embeddings generation
│   ├── vector_store.py       # FAISS vector operations
│   ├── retriever.py          # Document retrieval logic
│   ├── chatbot.py           # Main chatbot implementation
│   └── ui.py                # Gradio web interface
├── config/
│   └── settings.py          # Configuration management
├── data/
│   ├── uploads/             # Uploaded PDF files
│   ├── processed/           # Processed text chunks
│   └── faiss_index/         # FAISS index files
├── main.py                  # Application entry point
└── requirements.txt         # Python dependencies
```

## 🔧 Technical Details

### LangChain Integration
- **Document Loaders**: PyMuPDFLoader for superior PDF text extraction
- **Text Splitters**: RecursiveCharacterTextSplitter for intelligent chunking
- **Embeddings**: OpenAIEmbeddings with automatic caching
- **Vector Stores**: FAISS integration with persistence
- **Chains**: RetrievalQA chain for optimized question-answering

### PDF Processing
- Uses PyMuPDF (via LangChain) for robust text extraction
- Recursive character text splitting with configurable overlap
- Preserves document metadata (page numbers, source files)
- Handles complex document layouts and formatting

### Vector Storage
- LangChain FAISS integration for seamless vector operations
- Automatic embedding generation during document ingestion
- Persistent storage with save/load functionality
- Optimized similarity search with score ranking

### Retrieval System
- Uses LangChain's RetrievalQA chain for structured retrieval
- Configurable top-k document retrieval
- Context-aware responses with source citations
- Automatic prompt engineering for RAG scenarios

### Chat Interface
- Built with Gradio for modern, responsive UI
- Real-time document processing with progress feedback
- Conversation history management
- System status monitoring and diagnostics

## 🔒 Security & Performance

- **API Key Management**: Secure environment variable handling
- **File Validation**: PDF file validation and size limits
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Caching**: Embedding caching for improved performance
- **Memory Management**: Efficient FAISS operations for large documents

## 📊 Performance Tips

1. **Chunk Size**: Adjust `CHUNK_SIZE` based on your documents (500-2000 tokens)
2. **Overlap**: Use `CHUNK_OVERLAP` to maintain context between chunks
3. **Top-K**: Increase `TOP_K_DOCUMENTS` for more comprehensive answers
4. **File Size**: Keep PDFs under the configured size limit for best performance

## 🚨 Troubleshooting

### Common Issues

1. **"Import could not be resolved" errors**:
   - Make sure you've installed all dependencies: `pip install -r requirements.txt`

2. **OpenAI API errors**:
   - Check your API key in the `.env` file
   - Ensure you have sufficient API credits

3. **FAISS installation issues**:
   - Install FAISS CPU version: `pip install faiss-cpu`
   - For GPU support: `pip install faiss-gpu` (requires CUDA)

4. **PDF processing errors**:
   - Ensure PDFs contain extractable text (not just images)
   - Check file size limits in configuration

5. **Memory issues**:
   - Reduce chunk size or number of documents
   - Consider using a machine with more RAM

### Logs
Application logs are saved to `chatbot.log` for debugging.

## 🧪 Development

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/
```

### Code Quality
```bash
# Install development dependencies
pip install black flake8 mypy

# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs in `chatbot.log`
3. Open an issue on GitHub

## 🙏 Acknowledgments

- **OpenAI** for the GPT models and embeddings API
- **Facebook AI Research** for FAISS vector search
- **Gradio** for the user interface framework
- **PyPDF2** for PDF processing capabilities

---

**Happy chatting with your PDFs! 🤖📄**
