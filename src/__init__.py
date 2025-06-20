"""RAG PDF Chatbot package."""

from .chatbot import RAGChatbot, process_pdf_file, ask_question, get_system_status, clear_all_data
from .ui import ChatbotUI, launch_app, create_demo
from .pdf_processor import process_pdf_bytes, PDFProcessor, TextChunker
from .embeddings import EmbeddingsGenerator, generate_query_embedding
from .vector_store import VectorStoreManager, create_vector_store
from .retriever import DocumentRetriever, retrieve_documents, get_context_for_query

__version__ = "1.0.0"
__author__ = "RAG PDF Chatbot"
__description__ = "A Retrieval-Augmented Generation chatbot for PDF documents"

__all__ = [
    # Main chatbot
    'RAGChatbot',
    'process_pdf_file',
    'ask_question',
    'get_system_status',
    'clear_all_data',
    
    # UI
    'ChatbotUI',
    'launch_app',
    'create_demo',
    
    # PDF processing
    'process_pdf_bytes',
    'PDFProcessor',
    'TextChunker',
    
    # Embeddings
    'EmbeddingsGenerator',
    'generate_query_embedding',
    
    # Vector store
    'VectorStoreManager',
    'create_vector_store',
    
    # Retrieval
    'DocumentRetriever',
    'retrieve_documents',
    'get_context_for_query'
]
