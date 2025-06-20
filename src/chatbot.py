"""Main chatbot logic using LangChain RetrievalQA chain."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from .pdf_processor import process_pdf_bytes
from .vector_store import create_vector_store
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatResponse:
    """Data class for chat responses."""
    answer: str
    context_used: List[Dict[str, Any]]
    query: str
    processing_time: float
    error: Optional[str] = None

class LangChainRAGChatbot:
    """RAG chatbot implementation using LangChain RetrievalQA."""
    
    def __init__(self):
        self.config = settings.get_openai_config()
        self.vector_store_manager = create_vector_store()
        
        # Initialize LangChain ChatOpenAI
        self.llm = ChatOpenAI(
            openai_api_key=self.config["api_key"],
            model_name=self.config["chat_model"],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"]
        )
        
        # Custom prompt template for RAG
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant that answers questions based on the provided PDF document context.

Instructions:
1. Use ONLY the information provided in the context to answer questions
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Always cite which document or page your information comes from when possible
4. Be precise and factual
5. If asked about something not in the documents, politely explain that you can only answer based on the uploaded PDF content

Context from PDF documents:
{context}

Question: {question}

Answer:"""
        )
        
        # RetrievalQA chain will be created when documents are available
        self.qa_chain = None
    
    def _create_qa_chain(self):
        """Create or recreate the RetrievalQA chain."""
        try:
            if self.vector_store_manager.is_empty():
                self.qa_chain = None
                return
            
            # Get the LangChain vector store
            vector_store = self.vector_store_manager.store.vector_store
            
            if vector_store is None:
                self.qa_chain = None
                return
            
            # Create retriever from vector store
            retriever = vector_store.as_retriever(
                search_kwargs={"k": self.config["top_k_documents"]}
            )
            
            # Create RetrievalQA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": self.prompt_template},
                return_source_documents=True
            )
            
            logger.info("Created RetrievalQA chain")
            
        except Exception as e:
            logger.error(f"Failed to create QA chain: {str(e)}")
            self.qa_chain = None
    
    def process_pdf(self, pdf_bytes: bytes, filename: str = "uploaded.pdf") -> Dict[str, Any]:
        """Process a PDF file and add to vector store."""
        try:
            import time
            start_time = time.time()
            
            logger.info(f"Processing PDF: {filename}")
            
            # Extract and chunk documents using LangChain
            documents = process_pdf_bytes(pdf_bytes, filename)
            
            if not documents:
                return {
                    'success': False,
                    'message': 'No text content found in PDF',
                    'chunks_count': 0,
                    'processing_time': time.time() - start_time
                }
            
            # Add documents to vector store (embeddings generated automatically)
            self.vector_store_manager.add_pdf_documents(documents)
            
            # Recreate QA chain with new documents
            self._create_qa_chain()
            
            processing_time = time.time() - start_time
            
            logger.info(f"Successfully processed PDF with {len(documents)} documents in {processing_time:.3f}s")
            
            return {
                'success': True,
                'message': f'Successfully processed {filename}',
                'chunks_count': len(documents),
                'processing_time': processing_time,
                'filename': filename
            }
            
        except Exception as e:
            logger.error(f"Failed to process PDF: {str(e)}")
            return {
                'success': False,
                'message': f'Error processing PDF: {str(e)}',
                'chunks_count': 0,
                'processing_time': 0
            }
    
    def chat(self, query: str, conversation_history: List[Dict[str, str]] = None) -> ChatResponse:
        """Generate response for a user query using RetrievalQA."""
        try:
            import time
            start_time = time.time()
            
            if not query or not query.strip():
                return ChatResponse(
                    answer="Please provide a question.",
                    context_used=[],
                    query=query,
                    processing_time=0,
                    error="Empty query"
                )
            
            # Check if vector store has documents
            if self.vector_store_manager.is_empty():
                return ChatResponse(
                    answer="No PDF documents have been uploaded yet. Please upload a PDF file first.",
                    context_used=[],
                    query=query,
                    processing_time=time.time() - start_time,
                    error="No documents"
                )
            
            # Ensure QA chain is created
            if self.qa_chain is None:
                self._create_qa_chain()
            
            if self.qa_chain is None:
                return ChatResponse(
                    answer="Error: Could not create QA chain. Please try uploading your PDF again.",
                    context_used=[],
                    query=query,
                    processing_time=time.time() - start_time,
                    error="QA chain creation failed"
                )
            
            # Run the RetrievalQA chain
            result = self.qa_chain({"query": query})
            
            answer = result.get("result", "I'm sorry, I couldn't generate a response.")
            source_documents = result.get("source_documents", [])
            
            # Convert source documents to context format
            context_used = []
            for i, doc in enumerate(source_documents):
                context_doc = {
                    'text': doc.page_content,
                    'rank': i + 1,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'page': doc.metadata.get('page', 'N/A'),
                    **doc.metadata
                }
                context_used.append(context_doc)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Generated response using {len(context_used)} source documents in {processing_time:.3f}s")
            
            return ChatResponse(
                answer=answer,
                context_used=context_used,
                query=query,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Failed to generate chat response: {str(e)}")
            return ChatResponse(
                answer=f"I'm sorry, I encountered an error while processing your question: {str(e)}",
                context_used=[],
                query=query,
                processing_time=time.time() - start_time if 'start_time' in locals() else 0,
                error=str(e)
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        vector_stats = self.vector_store_manager.get_stats()
        return {
            'vector_store': vector_stats,
            'has_documents': not self.vector_store_manager.is_empty(),
            'qa_chain_ready': self.qa_chain is not None,
            'model': self.config["chat_model"],
            'embeddings_model': self.config["embedding_model"],
            'max_tokens': self.config["max_tokens"],
            'temperature': self.config["temperature"],
            'top_k_documents': self.config["top_k_documents"]
        }
    
    def clear_documents(self) -> Dict[str, Any]:
        """Clear all documents from the system."""
        try:
            self.vector_store_manager.clear_all()
            self.qa_chain = None  # Clear QA chain since no documents remain
            logger.info("Cleared all documents from vector store")
            return {
                'success': True,
                'message': 'All documents cleared successfully'
            }
        except Exception as e:
            logger.error(f"Failed to clear documents: {str(e)}")
            return {
                'success': False,
                'message': f'Error clearing documents: {str(e)}'
            }

# Legacy compatibility
class RAGChatbot(LangChainRAGChatbot):
    """Legacy wrapper for backward compatibility."""
    pass

class ConversationManager:
    """Manages conversation history and context."""
    
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # Trim history if it gets too long
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history.copy()
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
    
    def format_for_display(self) -> List[Tuple[str, str]]:
        """Format conversation history for Gradio display."""
        formatted = []
        for i in range(0, len(self.conversation_history), 2):
            user_msg = self.conversation_history[i] if i < len(self.conversation_history) else None
            assistant_msg = self.conversation_history[i + 1] if i + 1 < len(self.conversation_history) else None
            
            user_content = user_msg['content'] if user_msg and user_msg['role'] == 'user' else ""
            assistant_content = assistant_msg['content'] if assistant_msg and assistant_msg['role'] == 'assistant' else ""
            
            if user_content or assistant_content:
                formatted.append((user_content, assistant_content))
        
        return formatted

# Global chatbot instance
chatbot = LangChainRAGChatbot()
conversation_manager = ConversationManager()

# Convenience functions
def process_pdf_file(pdf_bytes: bytes, filename: str = "uploaded.pdf") -> Dict[str, Any]:
    """Process PDF file using global chatbot instance."""
    return chatbot.process_pdf(pdf_bytes, filename)

def ask_question(query: str, use_history: bool = True) -> ChatResponse:
    """Ask a question using global chatbot instance."""
    history = conversation_manager.get_history() if use_history else None
    response = chatbot.chat(query, history)
    
    # Add to conversation history
    if use_history:
        conversation_manager.add_message("user", query)
        conversation_manager.add_message("assistant", response.answer)
    
    return response

def get_system_status() -> Dict[str, Any]:
    """Get system status using global chatbot instance."""
    return chatbot.get_system_status()

def clear_all_data() -> Dict[str, Any]:
    """Clear all documents and conversation history."""
    result = chatbot.clear_documents()
    conversation_manager.clear_history()
    return result
