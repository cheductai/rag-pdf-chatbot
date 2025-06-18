"""Gradio web interface for the RAG PDF chatbot."""

import tempfile
from pathlib import Path
from typing import List, Tuple

import gradio as gr
from loguru import logger

from config.settings import settings
from src.langchain_rag_pipeline import LangChainRAGPipeline


class GradioInterface:
    """Gradio web interface for the RAG PDF chatbot."""
    
    def __init__(self):
        self.rag_pipeline = LangChainRAGPipeline()
        self.chat_history: List[Tuple[str, str]] = []
        
        # Initialize the pipeline
        if not self.rag_pipeline.initialize():
            logger.error("Failed to initialize RAG pipeline")
    
    def upload_pdf(self, file) -> Tuple[str, str]:
        """
        Handle PDF file upload and processing.
        
        Args:
            file: Uploaded file object from Gradio
            
        Returns:
            Tuple[str, str]: (status_message, chat_interface_visibility)
        """
        if file is None:
            return "Please select a PDF file to upload.", gr.update(visible=False)
        
        try:
            logger.info(f"Processing uploaded file: {file.name}")
            
            # Read file content
            pdf_bytes = file.read() if hasattr(file, 'read') else Path(file.name).read_bytes()
            
            # Process the PDF
            success = self.rag_pipeline.process_pdf_from_bytes(pdf_bytes, file.name)
            
            if success:
                self.chat_history = []  # Clear previous chat history
                status_msg = f"✅ Successfully processed '{file.name}'. You can now ask questions about the document!"
                return status_msg, gr.update(visible=True)
            else:
                return "❌ Failed to process the PDF. Please check the file and try again.", gr.update(visible=False)
                
        except Exception as e:
            logger.error(f"PDF upload failed: {e}")
            return f"❌ Error processing PDF: {str(e)}", gr.update(visible=False)
    
    def chat_with_pdf(self, message: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
        """
        Handle chat messages and generate responses.
        
        Args:
            message: User message
            history: Chat history
            
        Returns:
            Tuple[str, List[List[str]]]: (empty_string, updated_history)
        """
        if not message or not message.strip():
            return "", history
        
        if not self.rag_pipeline.is_initialized:
            error_msg = "Please upload and process a PDF file first."
            history.append([message, error_msg])
            return "", history
        
        try:
            # Get response from RAG pipeline
            response = self.rag_pipeline.query(message)
            
            if response is None:
                response = "I'm sorry, I encountered an error while processing your question. Please try again."
            
            # Update history
            history.append([message, response])
            
            return "", history
            
        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            error_response = "I'm sorry, I encountered an error while processing your question. Please try again."
            history.append([message, error_response])
            return "", history
    
    def clear_chat(self) -> List:
        """Clear the chat history."""
        return []
    
    def get_pipeline_status(self) -> str:
        """Get the current status of the RAG pipeline."""
        status = self.rag_pipeline.get_status()
        
        if not status["initialized"]:
            return "❌ No PDF loaded. Please upload a PDF file to start."
        
        embeddings_status = "✅ Ready" if status["embeddings_ready"] else "❌ Not Ready"
        llm_status = "✅ Ready" if status["llm_ready"] else "❌ Not Ready"
        qa_chain_status = "✅ Ready" if status["qa_chain_ready"] else "❌ Not Ready"
        
        return f"""✅ **System Status:**
- **Current PDF:** {status['current_pdf'] or 'None'}
- **Total Documents:** {status['total_documents']}
- **OpenAI Embeddings:** {embeddings_status} ({status['embedding_model']})
- **OpenAI LLM:** {llm_status} ({status['openai_model']})
- **QA Chain:** {qa_chain_status}
- **Ready for Questions:** {'Yes' if status['initialized'] else 'No'}"""
    
    def create_interface(self) -> gr.Blocks:
        """
        Create and configure the Gradio interface.
        
        Returns:
            gr.Blocks: Configured Gradio interface
        """
        with gr.Blocks(
            title="LangChain RAG PDF Chatbot",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .chat-container {
                height: 500px !important;
            }
            """
        ) as interface:
            
            gr.Markdown(
                """
                # 📚 LangChain RAG PDF Chatbot
                
                Upload a PDF document and ask questions about its content. This system uses LangChain components including:
                - **PyMuPDFLoader** for robust PDF processing
                - **OpenAI Embeddings** for high-quality text embeddings
                - **FAISS Vector Store** for efficient similarity search
                - **ChatOpenAI** with **RetrievalQA** for intelligent responses
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    # File upload section
                    gr.Markdown("### 📁 Upload PDF")
                    pdf_file = gr.File(
                        label="Select PDF File",
                        file_types=[".pdf"],
                        file_count="single"
                    )
                    
                    upload_btn = gr.Button("Process PDF", variant="primary", size="lg")
                    upload_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=3
                    )
                    
                    # System status
                    gr.Markdown("### 📊 System Status")
                    status_btn = gr.Button("Refresh Status", size="sm")
                    status_display = gr.Markdown(self.get_pipeline_status())
                
                with gr.Column(scale=2):
                    # Chat interface
                    chat_interface = gr.Column(visible=False)
                    
                    with chat_interface:
                        gr.Markdown("### 💬 Chat with your PDF")
                        
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=400,
                            show_label=False,
                            container=True,
                            bubble_full_width=False
                        )
                        
                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="Ask a question",
                                placeholder="Type your question about the PDF here...",
                                lines=2,
                                scale=4
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)
                        
                        with gr.Row():
                            clear_btn = gr.Button("Clear Chat", size="sm")
            
            # Example questions
            with gr.Row():
                gr.Markdown(
                    """
                    ### 💡 Example Questions
                    - "What is the main topic of this document?"
                    - "Can you summarize the key points?"
                    - "What does the document say about [specific topic]?"
                    - "Are there any conclusions or recommendations?"
                    - "What are the key findings mentioned in the document?"
                    """
                )
            
            # Event handlers
            upload_btn.click(
                fn=self.upload_pdf,
                inputs=[pdf_file],
                outputs=[upload_status, chat_interface]
            )
            
            msg_input.submit(
                fn=self.chat_with_pdf,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )
            
            send_btn.click(
                fn=self.chat_with_pdf,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )
            
            clear_btn.click(
                fn=self.clear_chat,
                outputs=[chatbot]
            )
            
            status_btn.click(
                fn=self.get_pipeline_status,
                outputs=[status_display]
            )
        
        return interface
    
    def launch(self) -> None:
        """Launch the Gradio interface."""
        interface = self.create_interface()
        
        logger.info(f"Launching Gradio interface on port {settings.gradio_port}")
        
        interface.launch(
            server_port=settings.gradio_port,
            share=settings.gradio_share,
            show_error=True,
            quiet=False
        )
