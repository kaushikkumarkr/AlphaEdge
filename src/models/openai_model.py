from typing import Optional
from openai import AsyncOpenAI
from src.models.base import BaseModelInterface, ModelResponse
from src.config.settings import settings


class OpenAIModel(BaseModelInterface):
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.default_model
        api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> ModelResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )
        
        return ModelResponse(
            content=response.choices[0].message.content or "",
            model=response.model
        )
