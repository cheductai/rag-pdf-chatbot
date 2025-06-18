"""LangChain-based RAG pipeline implementation."""

from pathlib import Path
from typing import Optional

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from loguru import logger

from config.settings import settings
from src.langchain_processor import LangChainDocumentProcessor
from src.langchain_vectorstore import LangChainVectorStore


class LangChainRAGPipeline:
    """Complete RAG pipeline using LangChain components."""
    
    def __init__(self):
        self.document_processor = LangChainDocumentProcessor()
        self.vector_store = LangChainVectorStore()
        self.llm = None
        self.qa_chain = None
        self.is_initialized = False
        self.current_pdf_name: Optional[str] = None
        self._initialize_llm()
    
    def _initialize_llm(self) -> None:
        """Initialize the ChatOpenAI model."""
        try:
            if not settings.openai_api_key:
                logger.error("OpenAI API key not provided")
                return
            
            logger.info(f"Initializing ChatOpenAI with model: {settings.openai_model}")
            self.llm = ChatOpenAI(
                model_name=settings.openai_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                openai_api_key=settings.openai_api_key
            )
            logger.info("ChatOpenAI initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _create_qa_chain(self) -> bool:
        """Create the RetrievalQA chain."""
        if not self.llm or not self.vector_store.vectorstore:
            logger.error("LLM or vector store not initialized")
            return False
        
        try:
            # Create custom prompt template
            prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context provided, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer: """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Get retriever from vector store
            retriever = self.vector_store.get_retriever()
            if not retriever:
                logger.error("Failed to get retriever from vector store")
                return False
            
            # Create RetrievalQA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            logger.info("RetrievalQA chain created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create QA chain: {e}")
            return False
    
    def initialize(self) -> bool:
        """
        Initialize the RAG pipeline.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Try to load existing vector store
            if self.vector_store.load_vectorstore():
                if self._create_qa_chain():
                    self.is_initialized = True
                    logger.info("RAG pipeline initialized with existing vector store")
                    return True
            
            logger.info("No existing vector store found. Ready to process new PDF.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            return False
    
    def process_pdf(self, pdf_path: Path) -> bool:
        """
        Process a PDF file and build the vector store.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Starting PDF processing: {pdf_path.name}")
            
            # Clear existing vector store
            self.vector_store.clear_vectorstore()
            
            # Process PDF into documents
            documents = self.document_processor.load_and_split_pdf(pdf_path)
            if not documents:
                logger.error("Failed to process PDF")
                return False
            
            # Create vector store
            if not self.vector_store.create_vectorstore(documents):
                logger.error("Failed to create vector store")
                return False
            
            # Create QA chain
            if not self._create_qa_chain():
                logger.error("Failed to create QA chain")
                return False
            
            # Save vector store
            if not self.vector_store.save_vectorstore():
                logger.warning("Failed to save vector store to disk")
            
            self.is_initialized = True
            self.current_pdf_name = pdf_path.name
            logger.info(f"Successfully processed PDF: {pdf_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return False
    
    def process_pdf_from_bytes(self, pdf_bytes: bytes, filename: str) -> bool:
        """
        Process PDF from bytes (for uploaded files).
        
        Args:
            pdf_bytes: PDF file content as bytes
            filename: Original filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Starting PDF processing from bytes: {filename}")
            
            # Clear existing vector store
            self.vector_store.clear_vectorstore()
            
            # Process PDF into documents
            documents = self.document_processor.process_pdf_from_bytes(pdf_bytes, filename)
            if not documents:
                logger.error("Failed to process PDF from bytes")
                return False
            
            # Create vector store
            if not self.vector_store.create_vectorstore(documents):
                logger.error("Failed to create vector store")
                return False
            
            # Create QA chain
            if not self._create_qa_chain():
                logger.error("Failed to create QA chain")
                return False
            
            # Save vector store
            if not self.vector_store.save_vectorstore():
                logger.warning("Failed to save vector store to disk")
            
            self.is_initialized = True
            self.current_pdf_name = filename
            logger.info(f"Successfully processed PDF: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"PDF processing from bytes failed: {e}")
            return False
    
    def query(self, question: str) -> Optional[str]:
        """
        Query the RAG system with a question.
        
        Args:
            question: User question
            
        Returns:
            Optional[str]: Generated answer or None if query fails
        """
        if not self.is_initialized or not self.qa_chain:
            logger.error("RAG pipeline not initialized")
            return None
        
        if not question or not question.strip():
            logger.warning("Empty question provided")
            return None
        
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Run the QA chain
            result = self.qa_chain({"query": question})
            
            answer = result.get("result", "").strip()
            source_docs = result.get("source_documents", [])
            
            if not answer:
                return "I couldn't generate an answer to your question. Please try rephrasing it."
            
            # Optionally add source information
            if source_docs:
                logger.info(f"Answer generated using {len(source_docs)} source documents")
            
            logger.info("Successfully generated answer")
            return answer
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return "I encountered an error while processing your question. Please try again."
    
    def get_status(self) -> dict:
        """
        Get the current status of the RAG pipeline.
        
        Returns:
            dict: Status information
        """
        vector_status = self.vector_store.get_status()
        
        return {
            "initialized": self.is_initialized,
            "current_pdf": self.current_pdf_name,
            "total_documents": vector_status["total_documents"],
            "embeddings_ready": vector_status["embeddings_ready"],
            "llm_ready": self.llm is not None,
            "qa_chain_ready": self.qa_chain is not None,
            "openai_model": settings.openai_model,
            "embedding_model": settings.embedding_model
        }
