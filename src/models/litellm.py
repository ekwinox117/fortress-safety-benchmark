from openai import OpenAI
from typing import Any, Optional
from .base import ModelProvider
import os
OPENAI_REASONING_MODELS = ["openai/o1-mini-2024-09-12", "openai/o1-2024-12-17", "openai/o3-mini-2025-01-31", "openai/o3-2025-04-16", "openai/o4-mini-2025-04-16"]

class LiteLLMModel(ModelProvider):
    """LiteLLM model provider"""
    def __init__(self, model: str, temp: float = None, max_tokens: Optional[int] = None, response_format: Optional[Any] = None):
        """Initialize LiteLLM API with the necessary parameters.
        
        Parameters:
        - model_path: Path to the model.
        - temp: Temperature for sampling
        - max_tokens: Maximum number of tokens to generate.
        - response_format: Format of the response (optional).
        """
        base_url = os.getenv("BASE_URL")
        api_key = os.getenv("API_KEY")
        
        if not base_url:
            raise ValueError("BASE_URL is not set in the .env file.")
        if not api_key:
            raise ValueError("API_KEY is not set in the .env file.")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        self.model = model
        self.temp = float(temp) if temp is not None else None
        self.max_tokens = int(max_tokens) if max_tokens else None
        self.response_format = response_format
    
    def generate(self, prompt: str) -> str:
        """Generate a response using the LiteLLM API."""

        # Create the request that will go to the model provider
        request_dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        if self.temp is not None:
            request_dict['temperature'] = self.temp

        if self.response_format:
            request_dict['response_format'] = self.response_format

        if self.max_tokens:
            request_dict['max_tokens'] = self.max_tokens

        if self.model in OPENAI_REASONING_MODELS:
            request_dict['messages'] = [x for x in request_dict['messages'] if x['role'] != 'system']
            request_dict = {k:v for k,v in request_dict.items() if k in ['model','messages']}
        
        try:
            if self.response_format:
                response = self.client.beta.chat.completions.parse(**request_dict)
                model_response = response.choices[0].message.parsed
                
            else:
                response = self.client.chat.completions.create(**request_dict)
                model_response = response.choices[0].message.content

            return model_response
        
        except Exception as e:
            raise Exception(f"Error generating response from LiteLLM: {str(e)}")