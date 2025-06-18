"""OpenAI response generation module using AI SDK."""

from typing import List, Optional, Tuple

from loguru import logger

from config.settings import settings


class OpenAIResponseGenerator:
    """Handles response generation using OpenAI models via AI SDK."""
    
    def __init__(self):
        self.api_key = settings.openai_api_key
        self.model_name = settings.openai_model
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature
        self.use_openai = settings.use_openai and bool(self.api_key)
        
        if not self.use_openai:
            logger.warning("OpenAI integration disabled - API key not provided or use_openai=False")
    
    def generate_response(self, question: str, context_results: List[Tuple[str, float]]) -> Optional[str]:
        """
        Generate a response using OpenAI based on the question and retrieved context.
        
        Args:
            question: User question
            context_results: List of (text, similarity_score) tuples from vector search
            
        Returns:
            Optional[str]: Generated response or None if generation fails
        """
        if not self.use_openai:
            return self._generate_fallback_response(question, context_results)
        
        try:
            # Import AI SDK components (dynamic import to handle missing dependencies)
            from ai import generateText
            from ai.openai import openai
            
            # Prepare context from retrieved results
            context_text = self._prepare_context(context_results)
            
            # Create the prompt
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(question, context_text)
            
            logger.info(f"Generating OpenAI response for question: {question[:100]}...")
            
            # Generate response using AI SDK
            result = generateText({
                'model': openai(self.model_name),
                'system': system_prompt,
                'prompt': user_prompt,
                'maxTokens': self.max_tokens,
                'temperature': self.temperature
            })
            
            response = result.get('text', '').strip()
            
            if response:
                logger.info("Successfully generated OpenAI response")
                return response
            else:
                logger.warning("OpenAI returned empty response")
                return self._generate_fallback_response(question, context_results)
                
        except ImportError as e:
            logger.error(f"AI SDK not available: {e}")
            return self._generate_fallback_response(question, context_results)
        except Exception as e:
            logger.error(f"OpenAI response generation failed: {e}")
            return self._generate_fallback_response(question, context_results)
    
    def _prepare_context(self, context_results: List[Tuple[str, float]]) -> str:
        """
        Prepare context text from retrieved results.
        
        Args:
            context_results: List of (text, similarity_score) tuples
            
        Returns:
            str: Formatted context text
        """
        if not context_results:
            return "No relevant context found."
        
        context_parts = []
        for i, (text, score) in enumerate(context_results[:5], 1):  # Use top 5 results
            # Clean and truncate text if too long
            clean_text = text.strip()
            if len(clean_text) > 800:  # Limit context length
                clean_text = clean_text[:800] + "..."
            
            context_parts.append(f"Context {i} (Relevance: {score:.2f}):\n{clean_text}")
        
        return "\n\n".join(context_parts)
    
    def _create_system_prompt(self) -> str:
        """
        Create the system prompt for OpenAI.
        
        Returns:
            str: System prompt
        """
        return """You are a helpful AI assistant that answers questions based on provided document context. 

Your responsibilities:
1. Answer questions using ONLY the information provided in the context
2. If the context doesn't contain enough information to answer the question, clearly state this
3. Provide accurate, concise, and well-structured responses
4. Quote relevant parts of the context when appropriate
5. If multiple context sections are relevant, synthesize the information coherently
6. Maintain a helpful and professional tone

Important guidelines:
- Do NOT make up information not present in the context
- Do NOT use your general knowledge to supplement the answer
- If the question cannot be answered from the context, suggest what type of information would be needed
- Always be honest about the limitations of the provided context"""
    
    def _create_user_prompt(self, question: str, context: str) -> str:
        """
        Create the user prompt combining question and context.
        
        Args:
            question: User question
            context: Retrieved context text
            
        Returns:
            str: Complete user prompt
        """
        return f"""Based on the following context from the document, please answer this question:

QUESTION: {question}

CONTEXT:
{context}

Please provide a comprehensive answer based solely on the information provided in the context above."""
    
    def _generate_fallback_response(self, question: str, context_results: List[Tuple[str, float]]) -> str:
        """
        Generate a fallback response when OpenAI is not available.
        
        Args:
            question: User question
            context_results: List of (text, similarity_score) tuples
            
        Returns:
            str: Fallback response
        """
        if not context_results:
            return "I couldn't find relevant information in the document to answer your question."
        
        # Simple fallback response
        answer_parts = [
            f"Based on the document content, here's what I found regarding: '{question}'\n"
        ]
        
        for i, (text, score) in enumerate(context_results[:3], 1):
            answer_parts.append(f"\n**Relevant Section {i}** (Similarity: {score:.2f}):")
            # Truncate long text for readability
            display_text = text[:400] + "..." if len(text) > 400 else text
            answer_parts.append(f"{display_text}\n")
        
        if len(context_results) > 3:
            answer_parts.append(f"\n*Note: Found {len(context_results)} total relevant sections. OpenAI integration is currently unavailable for enhanced response generation.*")
        else:
            answer_parts.append(f"\n*Note: OpenAI integration is currently unavailable. This is a basic response based on document similarity search.*")
        
        return "".join(answer_parts)
    
    def is_available(self) -> bool:
        """
        Check if OpenAI integration is available and configured.
        
        Returns:
            bool: True if OpenAI can be used, False otherwise
        """
        return self.use_openai
    
    def get_model_info(self) -> dict:
        """
        Get information about the current OpenAI configuration.
        
        Returns:
            dict: Model configuration information
        """
        return {
            "enabled": self.use_openai,
            "model": self.model_name if self.use_openai else "Not configured",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "api_key_configured": bool(self.api_key)
        }
