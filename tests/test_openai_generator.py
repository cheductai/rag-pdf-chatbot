"""Tests for OpenAI response generator module."""

import pytest
from unittest.mock import Mock, patch

from src.openai_generator import OpenAIResponseGenerator


class TestOpenAIResponseGenerator:
    """Test cases for OpenAIResponseGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = OpenAIResponseGenerator()
    
    def test_prepare_context(self):
        """Test context preparation from search results."""
        context_results = [
            ("This is the first relevant text.", 0.9),
            ("This is the second relevant text.", 0.8),
            ("This is the third relevant text.", 0.7)
        ]
        
        context = self.generator._prepare_context(context_results)
        
        assert "Context 1" in context
        assert "Context 2" in context
        assert "Context 3" in context
        assert "0.9" in context
        assert "first relevant text" in context
    
    def test_prepare_context_empty(self):
        """Test context preparation with empty results."""
        context = self.generator._prepare_context([])
        assert context == "No relevant context found."
    
    def test_create_system_prompt(self):
        """Test system prompt creation."""
        prompt = self.generator._create_system_prompt()
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "context" in prompt.lower()
        assert "document" in prompt.lower()
    
    def test_create_user_prompt(self):
        """Test user prompt creation."""
        question = "What is the main topic?"
        context = "This document discusses artificial intelligence."
        
        prompt = self.generator._create_user_prompt(question, context)
        
        assert question in prompt
        assert context in prompt
        assert "QUESTION:" in prompt
        assert "CONTEXT:" in prompt
    
    def test_fallback_response(self):
        """Test fallback response generation."""
        question = "Test question"
        context_results = [
            ("Test context text", 0.8)
        ]
        
        response = self.generator._generate_fallback_response(question, context_results)
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert question in response
        assert "Test context text" in response
    
    def test_fallback_response_empty_context(self):
        """Test fallback response with empty context."""
        response = self.generator._generate_fallback_response("Test question", [])
        
        assert "couldn't find relevant information" in response
    
    def test_get_model_info(self):
        """Test model information retrieval."""
        info = self.generator.get_model_info()
        
        assert isinstance(info, dict)
        assert "enabled" in info
        assert "model" in info
        assert "max_tokens" in info
        assert "temperature" in info
        assert "api_key_configured" in info
    
    @patch('src.openai_generator.generateText')
    @patch('src.openai_generator.openai')
    def test_generate_response_with_openai(self, mock_openai, mock_generate):
        """Test response generation with OpenAI (mocked)."""
        # Mock the OpenAI response
        mock_generate.return_value = {'text': 'This is a test response from OpenAI.'}
        
        # Set up generator with API key
        self.generator.api_key = "test_key"
        self.generator.use_openai = True
        
        question = "What is AI?"
        context_results = [("AI is artificial intelligence.", 0.9)]
        
        response = self.generator.generate_response(question, context_results)
        
        assert response == 'This is a test response from OpenAI.'
