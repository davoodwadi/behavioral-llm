# src/models.py
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from openai import OpenAI
import anthropic
from google import genai
from google.genai import types
import streamlit as st
import pprint

@dataclass
class LLMResponse:
    """Standardized response object for all LLM queries."""
    content: str
    response_time_ms: float
    raw_response: Any  # Keep the original provider response for debugging or extended info


class LLM(ABC):
    """Abstract Base Class for LLM models."""
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def query(self, system_prompt: str, user_prompt: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> LLMResponse:
        """
        Queries the model with a standardized interface.
        Returns a structured LLMResponse object.
        """
        pass

class OpenAIModel(LLM):
    """Wrapper for OpenAI's chat models."""
    def __init__(self, model_name: str, api_key):
        super().__init__(model_name)
        
        final_api_key = api_key if api_key else os.environ.get("OPENAI_API_KEY", '')
        self.client = OpenAI(api_key=final_api_key)

    def query(self, system_prompt: str, user_prompt: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> LLMResponse:
        # st.write(conversation_history)
        # messages = [{"role": "system", "content": system_prompt}]
        
        # messages.append({"role": "user", "content": user_prompt})
        
        # if conversation_history -> use this instead of system_prompt and user_prompt
        if conversation_history:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_history)

        # pprint.pprint(messages)
        # pprint.pprint('*'*10)
        # return LLMResponse(
        #         content='',
        #         response_time_ms=10,
        #         raw_response='response'
        #     )
    
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            end_time = time.time()
            
            content = response.choices[0].message.content
            response_time_ms = (end_time - start_time) * 1000
            
            return LLMResponse(
                content=content,
                response_time_ms=response_time_ms,
                raw_response=response
            )
        except Exception as e:
            print(f"Error querying {self.model_name}: {e}")
            return LLMResponse(content=f"API_ERROR: {e}", response_time_ms=0, raw_response=e)

class AnthropicModel(LLM):
    """Wrapper for Anthropic's chat models (Claude)."""
    def __init__(self, model_name: str, api_key):
        super().__init__(model_name)
        final_api_key = api_key if api_key else os.environ.get("ANTHROPIC_API_KEY", '')

        self.client = anthropic.Anthropic(api_key=final_api_key)

    def query(self, system_prompt: str, user_prompt: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> LLMResponse:
        # Anthropic API has a dedicated 'system' parameter
        # messages = []
        # if conversation_history:
        #     messages.extend(conversation_history)
        # messages.append({"role": "user", "content": user_prompt})

        if conversation_history:
            messages = conversation_history

        # pprint.pprint('anthropic')        
        # pprint.pprint(system_prompt)
        # pprint.pprint(messages)
        # pprint.pprint('*'*10)
        # return LLMResponse(
        #         content='',
        #         response_time_ms=10,
        #         raw_response='response'
        #     )

        try:
            start_time = time.time()
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=messages,
                max_tokens=1024, # Anthropic requires max_tokens
            )
            end_time = time.time()
            
            content = response.content[0].text
            response_time_ms = (end_time - start_time) * 1000

            return LLMResponse(
                content=content,
                response_time_ms=response_time_ms,
                raw_response=response
            )
        except Exception as e:
            print(f"Error querying {self.model_name}: {e}")
            return LLMResponse(content=f"API_ERROR: {e}", response_time_ms=0, raw_response=e)

class GeminiModel(OpenAIModel):
    """Wrapper for Google's Gemini models."""
    def __init__(self, model_name: str, api_key):
        LLM.__init__(self, model_name) 
        final_api_key = api_key if api_key else os.environ.get("GEMINI_API_KEY", '')
        self.client = OpenAI(
            api_key=final_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    
   
class DeepInfraModel(OpenAIModel):
    """
    Wrapper for models on DeepInfra, which uses an OpenAI-compatible API.
    This demonstrates extending by inheritance for API-compatible services.
    """
    def __init__(self, model_name: str, api_key):
        # We don't call super().__init__() of OpenAIModel directly
        LLM.__init__(self, model_name) 
        final_api_key = api_key if api_key else os.environ.get("DEEPINFRA_API_KEY", '')
        
        self.client = OpenAI(
            api_key=final_api_key,
            base_url="https://api.deepinfra.com/v1/openai",
        )
        # The query() method from OpenAIModel is inherited and works perfectly.
class TestModel(LLM):
    """Wrapper for OpenAI's chat models."""
    def __init__(self, model_name: str, api_key):
        super().__init__(model_name)
        
        self.final_api_key = api_key if api_key else os.environ.get("OPENAI_API_KEY", '')
        # self.client = OpenAI(api_key=final_api_key)

    def query(self, system_prompt: str, user_prompt: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> LLMResponse:
        # messages = [{"role": "system", "content": system_prompt}]
        # if conversation_history:
        #     messages.extend(conversation_history)
        # messages.append({"role": "user", "content": user_prompt})

        if conversation_history:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_history)
        # st.write(messages)
        # print(messages) 

        try:
            start_time = time.time() 
            response = '[1,2,4,3,5,6,7]'
            end_time = time.time()
            # print('*****************')
            # for m in messages:
            #     print('+++++++++++++')
            #     print(m)
            #     print('+++++++++++++')
            # print('*****************')
            if self.final_api_key=='123':
                raise Exception
            content = '[1,2,4,3,5,6,7]'
            response_time_ms = (end_time - start_time) * 1000
            
            return LLMResponse(
                content=content,
                response_time_ms=response_time_ms,
                raw_response=response
            )
        except Exception as e:
            print(f"Error querying {self.model_name}: {e}")
            return LLMResponse(content=f"API_ERROR: {e}", response_time_ms=0, raw_response=e)
# A registry mapping a user-friendly provider key to the corresponding class.
MODEL_REGISTRY = {
    "openai": OpenAIModel,
    "anthropic": AnthropicModel,
    "google": GeminiModel,
    "deepinfra": DeepInfraModel,
    'test': TestModel
}

def get_model(user_model_name: str, api_keys) -> LLM:
    """
    Factory function to get the correct model instance based on a user-provided
    name in the format 'provider-model_name'.

    Examples:
        - 'openai-gpt-4o'
        - 'anthropic-claude-3-sonnet-20240229'
        - 'google-gemini-1.5-pro-latest'
        - 'deepinfra-meta-llama/Llama-3-8B-instruct'
    """
    # Split the string by the *first* hyphen only. This is crucial for model
    # names that contain hyphens themselves (like claude-3-sonnet...).
    parts = user_model_name.split('-', 1)

    assert len(parts) == 2, 'You should prefix model name with provider_name-'

    provider_key, api_model_name = parts

    # Look up the provider's class in the registry
    model_class = MODEL_REGISTRY.get(provider_key)
    api_key = api_keys.get(provider_key)
    if model_class is None:
        raise ValueError(
            f"Provider '{provider_key}' is not supported. "
            f"Available providers: {list(MODEL_REGISTRY.keys())}"
        )

    # Instantiate the class with the specific model name required by the API
    return model_class(api_model_name, api_key)