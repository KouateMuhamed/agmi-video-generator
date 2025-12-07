"""
LLM Provider abstraction for multi-LLM support.
"""

from typing import Protocol, TypeVar, Type, Optional, Any
from abc import ABC, abstractmethod
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(Protocol):
    """Protocol defining the interface for LLM providers."""
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: Type[T],
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> T:
        """
        Generate structured output from an LLM.
        
        Args:
            system_prompt: System instruction prompt
            user_prompt: User input prompt
            output_schema: Pydantic model class for structured output
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter (0.0-1.0)
        
        Returns:
            Instance of output_schema with parsed LLM response
        """
        ...


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize the LLM provider.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for authentication (if required)
        """
        self.model_name = model_name
        self.api_key = api_key
    
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: Type[T],
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> T:
        """Generate structured output from the LLM."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider implementation."""
    
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize OpenAI provider.
        
        Args:
            model_name: OpenAI model name (e.g., "gpt-4o", "gpt-4-turbo")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        
        Raises:
            ValueError: If API key is not provided
        """
        import os
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        super().__init__(model_name, api_key)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: Type[T],
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> T:
        """Generate structured output using OpenAI API."""
        import json
        
        try:
            # Try using structured outputs (beta API)
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=output_schema,
                temperature=temperature,
                top_p=top_p,
            )
            
            # Parse response
            parsed = response.choices[0].message.parsed
            if parsed is not None:
                return parsed
            
            # Fallback: try to parse raw content
            content = response.choices[0].message.content
            if content:
                data = json.loads(content)
                return output_schema(**data)
            raise ValueError("Failed to parse LLM response")
        except Exception as e:
            # Fallback to regular chat completion if beta API fails
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt + "\n\nOutput valid JSON matching this schema: " + json.dumps(output_schema.model_json_schema())},
                    ],
                    temperature=temperature,
                    top_p=top_p,
                )
                content = response.choices[0].message.content
                if content:
                    # Try to extract JSON from markdown code blocks
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                    data = json.loads(content)
                    return output_schema(**data)
            except Exception as fallback_error:
                raise ValueError(f"Failed to generate structured output: {e}, fallback also failed: {fallback_error}")
            raise ValueError(f"Failed to generate structured output: {e}")


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider implementation."""
    
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        """
        Initialize Anthropic provider.
        
        Args:
            model_name: Anthropic model name (e.g., "claude-3-5-sonnet-20241022")
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        
        Raises:
            ValueError: If API key is not provided
        """
        import os
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key is required. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
            )
        super().__init__(model_name, api_key)
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: Type[T],
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> T:
        """Generate structured output using Anthropic API."""
        import json
        
        # Get JSON schema from Pydantic model
        json_schema = output_schema.model_json_schema()
        
        # Anthropic uses tools for structured outputs
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                temperature=temperature,
                top_p=top_p,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                tools=[{
                    "name": "generate_output",
                    "description": f"Generate output matching the {output_schema.__name__} schema",
                    "input_schema": json_schema,
                }],
                tool_choice={"type": "tool", "name": "generate_output"},
            )
            
            # Extract tool use content
            if response.content and len(response.content) > 0:
                tool_use = response.content[0]
                if hasattr(tool_use, "input"):
                    data = tool_use.input
                    return output_schema(**data)
                elif hasattr(tool_use, "text"):
                    # Fallback: parse JSON from text
                    data = json.loads(tool_use.text)
                    return output_schema(**data)
            
            raise ValueError("Failed to parse Anthropic response: no tool use content")
        except Exception as e:
            # Fallback: try regular message with JSON instruction
            try:
                enhanced_prompt = user_prompt + "\n\nOutput valid JSON matching this schema:\n" + json.dumps(json_schema, indent=2)
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    temperature=temperature,
                    top_p=top_p,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": enhanced_prompt},
                    ],
                )
                
                if response.content and len(response.content) > 0:
                    text_content = response.content[0].text
                    # Try to extract JSON from markdown code blocks
                    if "```json" in text_content:
                        text_content = text_content.split("```json")[1].split("```")[0].strip()
                    elif "```" in text_content:
                        text_content = text_content.split("```")[1].split("```")[0].strip()
                    data = json.loads(text_content)
                    return output_schema(**data)
            except Exception as fallback_error:
                raise ValueError(f"Failed to generate structured output: {e}, fallback also failed: {fallback_error}")
            raise ValueError(f"Failed to generate structured output: {e}")


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider implementation."""
    
    def __init__(self, model_name: str = "gemini-2.0-flash-exp", api_key: Optional[str] = None):
        """
        Initialize Gemini provider.
        
        Args:
            model_name: Gemini model name (e.g., "gemini-2.0-flash-exp", "gemini-1.5-pro")
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
        
        Raises:
            ValueError: If API key is not provided
        """
        import os
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key is required. "
                "Set GOOGLE_API_KEY environment variable or pass api_key parameter."
            )
        super().__init__(model_name, api_key)
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
            self.model = genai.GenerativeModel(self.model_name)
        except ImportError:
            raise ImportError("google-generativeai package is required. Install with: pip install google-generativeai")
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: Type[T],
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> T:
        """Generate structured output using Gemini API."""
        import json
        
        # Get JSON schema from Pydantic model
        json_schema = output_schema.model_json_schema()
        
        # Combine system and user prompts for Gemini
        # Gemini doesn't have separate system/user roles, so we combine them
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        try:
            # Try using Gemini's JSON mode
            from google.generativeai.types import GenerationConfig
            
            # Add schema instruction to prompt
            enhanced_prompt = full_prompt + "\n\nOutput valid JSON matching this schema:\n" + json.dumps(json_schema, indent=2)
            
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                response_mime_type="application/json",
            )
            
            # Create model with generation config
            model = self.genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
            )
            
            # Generate content
            response = model.generate_content(enhanced_prompt)
            
            # Parse JSON response
            if response.text:
                # Clean up response text (remove markdown code blocks if present)
                text = response.text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                
                data = json.loads(text)
                return output_schema(**data)
            
            raise ValueError("Failed to parse Gemini response: empty response")
        except Exception as e:
            # Fallback: try without JSON mode (regular generation)
            try:
                from google.generativeai.types import GenerationConfig
                
                enhanced_prompt = full_prompt + "\n\nOutput valid JSON matching this schema:\n" + json.dumps(json_schema, indent=2)
                
                generation_config = GenerationConfig(
                    temperature=temperature,
                    top_p=top_p,
                )
                
                model = self.genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=generation_config,
                )
                
                response = model.generate_content(enhanced_prompt)
                
                if response.text:
                    text = response.text.strip()
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0].strip()
                    elif "```" in text:
                        text = text.split("```")[1].split("```")[0].strip()
                    
                    data = json.loads(text)
                    return output_schema(**data)
            except Exception as fallback_error:
                raise ValueError(f"Failed to generate structured output: {e}, fallback also failed: {fallback_error}")
            raise ValueError(f"Failed to generate structured output: {e}")


def create_provider_from_model(model: str, api_key: Optional[str] = None) -> BaseLLMProvider:
    """
    Automatically create the appropriate LLM provider based on model name.
    
    Args:
        model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash")
        api_key: Optional API key (if not provided, uses environment variables)
        
    Returns:
        Appropriate provider instance
        
    Raises:
        ValueError: If model name doesn't match any known provider pattern
    """
    model_lower = model.lower()
    
    # OpenAI models: gpt-*, o1-*
    if model_lower.startswith("gpt-") or model_lower.startswith("o1-"):
        return OpenAIProvider(model_name=model, api_key=api_key)
    
    # Anthropic models: claude-*
    elif model_lower.startswith("claude-"):
        return AnthropicProvider(model_name=model, api_key=api_key)
    
    # Gemini models: gemini-*
    elif model_lower.startswith("gemini-"):
        return GeminiProvider(model_name=model, api_key=api_key)
    
    else:
        raise ValueError(
            f"Unknown model: {model}. "
            "Supported model prefixes: 'gpt-', 'o1-' (OpenAI), 'claude-' (Anthropic), 'gemini-' (Google)"
        )

