"""Gradio user interface for RAG PDF Chatbot."""

import logging
import gradio as gr
from typing import List, Tuple, Optional
import time

from .chatbot import (
    process_pdf_file, 
    ask_question, 
    get_system_status, 
    clear_all_data,
    conversation_manager
)
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotUI:
    """Gradio interface for the RAG PDF Chatbot."""
    
    def __init__(self):
        self.config = settings.get_gradio_config()
        self.app = None
        
    def upload_pdf(self, file) -> Tuple[str, str]:
        """Handle PDF file upload."""
        try:
            if file is None:
                return "❌ No file uploaded", ""
            
            # Read file bytes
            with open(file.name, 'rb') as f:
                pdf_bytes = f.read()
            
            # Process PDF
            result = process_pdf_file(pdf_bytes, file.name.split('/')[-1])
            
            if result['success']:
                message = f"✅ {result['message']}\n"
                message += f"📄 Processed {result['chunks_count']} text chunks\n"
                message += f"⏱️ Processing time: {result['processing_time']:.2f}s"
                
                status = self._get_status_info()
                return message, status
            else:
                return f"❌ {result['message']}", ""
                
        except Exception as e:
            logger.error(f"Error uploading PDF: {str(e)}")
            return f"❌ Error uploading file: {str(e)}", ""
    
    def chat_with_pdf(self, message: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
        """Handle chat interaction."""
        try:
            if not message or not message.strip():
                return history, ""
            
            # Get response from chatbot
            response = ask_question(message.strip())
            
            # Update history
            history.append((message, response.answer))
            
            return history, ""
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            error_response = f"I encountered an error: {str(e)}"
            history.append((message, error_response))
            return history, ""
    
    def clear_chat(self) -> Tuple[List[Tuple[str, str]], str]:
        """Clear chat history."""
        conversation_manager.clear_history()
        return [], ""
    
    def clear_all(self) -> Tuple[List[Tuple[str, str]], str, str]:
        """Clear all data including documents."""
        try:
            result = clear_all_data()
            if result['success']:
                message = "✅ All documents and chat history cleared"
            else:
                message = f"❌ {result['message']}"
            
            status = self._get_status_info()
            return [], message, status
            
        except Exception as e:
            logger.error(f"Error clearing data: {str(e)}")
            return [], f"❌ Error clearing data: {str(e)}", ""
    
    def _get_status_info(self) -> str:
        """Get current system status information."""
        try:
            status = get_system_status()
            
            info = f"📊 **System Status**\n\n"
            info += f"📚 Documents loaded: {'Yes' if status['has_documents'] else 'No'}\n"
            info += f"📄 Total chunks: {status['vector_store']['total_documents']}\n"
            info += f"🤖 Chat model: {status['model']}\n"
            info += f"🔍 Embedding model: {status['embeddings_model']}\n"
            info += f"📊 Top-K results: {status['top_k_documents']}\n"
            info += f"🌡️ Temperature: {status['temperature']}\n"
            info += f"📝 Max tokens: {status['max_tokens']}"
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            return f"❌ Error getting status: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(
            theme=gr.themes.Soft(),
            title=self.config["title"],
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .chat-container {
                height: 500px !important;
            }
            """
        ) as interface:
            
            gr.Markdown(f"""
            # 🤖 {self.config['title']}
            
            {self.config['description']}
            
            ## How to use:
            1. **Upload a PDF** using the file uploader below
            2. **Wait for processing** - you'll see a confirmation message
            3. **Ask questions** about the PDF content in the chat
            4. **Get answers** based on the document content with citations
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Chat interface
                    chatbot_ui = gr.Chatbot(
                        label="💬 Chat with your PDF",
                        height=500,
                        show_label=True,
                        container=True,
                        bubble_full_width=False
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask a question about your PDF...",
                            label="Your question",
                            scale=4,
                            container=False
                        )
                        submit_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_chat_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                        clear_all_btn = gr.Button("🗑️ Clear All", variant="stop")
                
                with gr.Column(scale=1):
                    # File upload and status
                    gr.Markdown("### 📄 PDF Upload")
                    
                    file_upload = gr.File(
                        label="Upload PDF",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                    
                    upload_status = gr.Textbox(
                        label="Upload Status",
                        interactive=False,
                        max_lines=5
                    )
                    
                    gr.Markdown("### 📊 System Status")
                    
                    system_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        max_lines=10,
                        value=self._get_status_info()
                    )
                    
                    refresh_status_btn = gr.Button("🔄 Refresh Status", variant="secondary")
            
            # Event handlers
            msg_input.submit(
                fn=self.chat_with_pdf,
                inputs=[msg_input, chatbot_ui],
                outputs=[chatbot_ui, msg_input]
            )
            
            submit_btn.click(
                fn=self.chat_with_pdf,
                inputs=[msg_input, chatbot_ui],
                outputs=[chatbot_ui, msg_input]
            )
            
            file_upload.upload(
                fn=self.upload_pdf,
                inputs=[file_upload],
                outputs=[upload_status, system_status]
            )
            
            clear_chat_btn.click(
                fn=self.clear_chat,
                inputs=[],
                outputs=[chatbot_ui, msg_input]
            )
            
            clear_all_btn.click(
                fn=self.clear_all,
                inputs=[],
                outputs=[chatbot_ui, upload_status, system_status]
            )
            
            refresh_status_btn.click(
                fn=lambda: self._get_status_info(),
                inputs=[],
                outputs=[system_status]
            )
            
            # Add examples
            gr.Markdown("""
            ### 💡 Example Questions
            Try asking questions like:
            - "What is the main topic of this document?"
            - "Can you summarize the key points?"
            - "What does the document say about [specific topic]?"
            - "Are there any recommendations mentioned?"
            """)
            
            # Footer
            gr.Markdown("""
            ---
            **Note:** This chatbot only answers based on the content of uploaded PDF documents. 
            Make sure to upload a PDF file before asking questions.
            """)
        
        self.app = interface
        return interface
    
    def launch(self, **kwargs) -> None:
        """Launch the Gradio interface."""
        if self.app is None:
            self.create_interface()
        
        # Default launch parameters
        launch_params = {
            'server_name': '0.0.0.0',
            'server_port': 7860,
            'share': False,
            'debug': settings.DEBUG
        }
        
        # Update with any provided parameters
        launch_params.update(kwargs)
        
        logger.info(f"Launching Gradio interface on {launch_params['server_name']}:{launch_params['server_port']}")
        self.app.launch(**launch_params)

# Convenience functions
def create_ui() -> ChatbotUI:
    """Create a new chatbot UI instance."""
    return ChatbotUI()

def launch_app(**kwargs) -> None:
    """Launch the chatbot application."""
    ui = create_ui()
    ui.launch(**kwargs)

# Demo function for testing
def create_demo() -> gr.Blocks:
    """Create a demo interface for testing."""
    ui = ChatbotUI()
    return ui.create_interface()

if __name__ == "__main__":
    # Launch the app if this file is run directly
    launch_app(share=True)
